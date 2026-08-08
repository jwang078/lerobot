#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset
from .utils import resolve_episode_indices


def unconsumed_camera_keys(
    policy_cfg, ds_meta: LeRobotDatasetMetadata, rename_map: dict | None = None
) -> list[str]:
    """Camera features (dtype video/image) in ``ds_meta`` that the policy does
    NOT consume — i.e. their post-rename_map name is absent from the policy's
    input_features. These are safe to exclude from dataloading entirely (no
    temporal window, no parquet-column materialization, no video decode).
    Returns [] when the policy declares no input_features (nothing provable).
    """
    consumed = set(getattr(policy_cfg, "input_features", None) or {})
    if not consumed:
        return []
    rename = dict(rename_map or {})
    return [
        key
        for key, ft in ds_meta.features.items()
        if ft.get("dtype") in ("video", "image") and rename.get(key, key) not in consumed
    ]


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig,
    ds_meta: LeRobotDatasetMetadata,
    exclude_keys: set[str] | None = None,
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            if exclude_keys and key in exclude_keys:
                continue  # feature not consumed by the policy — don't build a temporal window for it
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if cfg.dataset.repo_ids is None:
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
            repo_type=cfg.dataset.repo_type,
        )
        # Camera features the policy does NOT consume are excluded from
        # dataloading entirely: no temporal window, no parquet column
        # materialization, no video decode. Matters enormously for
        # image-mode datasets (camera frames embedded as PNG bytes in the
        # parquet rows): a state-only policy otherwise pays full per-row
        # image extraction for tensors it throws away (~2.8x slower steps
        # observed on the small_engine image-mode dataset). rename_map is
        # honored so image trainings whose dataset keys rename into the
        # policy's input_features (e.g. base_rgb_letterbox -> base_rgb)
        # keep their cameras.
        excluded_image_keys = unconsumed_camera_keys(
            cfg.trainable_config, ds_meta, getattr(cfg, "rename_map", None)
        )
        if excluded_image_keys:
            logging.info(
                f"make_dataset: excluding {len(excluded_image_keys)} camera feature(s) the policy "
                f"doesn't consume from dataloading: {excluded_image_keys}"
            )
        delta_timestamps = resolve_delta_timestamps(
            cfg.trainable_config, ds_meta, exclude_keys=set(excluded_image_keys) or None
        )
        episodes = resolve_episode_indices(
            cfg.dataset.episodes, ds_meta.total_episodes, cfg.dataset.exclude_episodes
        )
        if not cfg.dataset.streaming:
            if cfg.dataset.repo_type == "bucket":
                raise ValueError(
                    "repo_type='bucket' is streaming-only: set dataset.streaming=true to train from an HF Storage Bucket."
                )
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
                exclude_features=excluded_image_keys or None,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
                repo_type=cfg.dataset.repo_type,
            )
    else:
        # Multi-dataset weighted-sampling mode (see DatasetConfig docstring
        # for the activation surface). Build the per-sub-dataset metadata
        # using the FIRST sub-dataset's metadata as the reference for
        # delta_timestamps resolution — all sub-datasets share the same
        # robot/task schema by construction (TrainPipelineConfig.validate
        # enforces parallel `stats_paths` so they're compatible).
        #
        # These asserts are guaranteed by TrainPipelineConfig.validate, which
        # ran before make_dataset is reached — pyright can't see across that
        # validation boundary, so we re-assert here for type-narrowing AND
        # defense-in-depth in case make_dataset is called directly.
        assert cfg.dataset.repo_ids is not None, "validate() should have set repo_ids before make_dataset"
        assert cfg.dataset.stats_paths is not None, "validate() requires stats_paths when repo_ids is set"
        assert cfg.policy is not None, "make_dataset requires cfg.policy"
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_ids[0], root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        # Same unconsumed-camera exclusion as the single-source branch,
        # computed on the base sub-dataset's meta (the sub that carries the
        # camera features; passing a superset to camera-less interventions is
        # a no-op — each reader drops only columns it actually has).
        excluded_image_keys = unconsumed_camera_keys(cfg.policy, ds_meta, getattr(cfg, "rename_map", None))
        if excluded_image_keys:
            logging.info(
                f"make_dataset (multi-source): excluding {len(excluded_image_keys)} camera "
                f"feature(s) the policy doesn't consume from dataloading: {excluded_image_keys}"
            )
        delta_timestamps = resolve_delta_timestamps(
            cfg.policy, ds_meta, exclude_keys=set(excluded_image_keys) or None
        )
        # In feature-intersection mode, resolve_delta_timestamps() built entries
        # for EVERY observation.* key in the base sub-dataset's features — some
        # of which don't exist in the other sub-datasets (that's the whole
        # reason the user opted in). MultiLeRobotDataset forwards the same
        # delta_timestamps dict to every sub-dataset's LeRobotDataset, and each
        # sub-dataset's DatasetReader precomputes delta_indices and later
        # fetches every key at __getitem__ time — including keys that don't
        # exist in its underlying hf_dataset → ValueError("Column X doesn't
        # exist") on the first batch. Filter to the cross-sub-dataset feature
        # intersection so no sub-dataset ever gets asked for a column it lacks.
        # This mirrors MultiLeRobotDataset's own `disabled_features` intersection
        # (line ~76-91) but at the delta_timestamps level, where the fetch is
        # actually driven from.
        if cfg.dataset.multi_source_feature_intersection and delta_timestamps:
            other_metas = [
                LeRobotDatasetMetadata(rid, root=cfg.dataset.root, revision=cfg.dataset.revision)
                for rid in cfg.dataset.repo_ids[1:]
            ]
            feature_intersection = set(ds_meta.features)
            for m in other_metas:
                feature_intersection &= set(m.features)
            dropped = set(delta_timestamps) - feature_intersection
            if dropped:
                logging.warning(
                    "Multi-source: dropping %d delta_timestamps key(s) not present in "
                    "every sub-dataset: %s. (multi_source_feature_intersection=True)",
                    len(dropped),
                    sorted(dropped),
                )
                delta_timestamps = {k: v for k, v in delta_timestamps.items() if k in feature_intersection}
                if not delta_timestamps:
                    delta_timestamps = None
        multi = MultiLeRobotDataset(
            repo_ids=cfg.dataset.repo_ids,
            root=cfg.dataset.root,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
            exclude_features=excluded_image_keys or None,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(multi.repo_id_to_index, indent=2)}"
        )
        # Wrap MultiLeRobotDataset in MultiSourceNormalizingDataset so each
        # frame is normalized with its source dataset's stats sidecar before
        # leaving the dataloader. The policy's downstream NormalizerProcessorStep
        # is set to a no-op in lerobot_train.py for this mode.
        # Import here (not at top) to avoid a circular import with factory.py
        # callers that don't use multi mode.
        from lerobot.datasets.multi_source_normalizing_dataset import (
            MultiSourceNormalizingDataset,
        )

        dataset = MultiSourceNormalizingDataset(
            multi_dataset=multi,
            stats_paths=cfg.dataset.stats_paths,
            features={**cfg.policy.input_features, **cfg.policy.output_features},
            norm_map=cfg.policy.normalization_mapping,
            norm_mode=cfg.dataset.norm_mode,
            allow_key_mismatch=cfg.dataset.multi_source_feature_intersection,
        )

    if cfg.dataset.use_imagenet_stats:
        # Bind to a local so pyright can narrow away the `| None` half of
        # LeRobotDatasetMetadata.stats' declared type. In practice stats is
        # never None once the dataset (or MultiSourceNormalizingDataset) is
        # constructed, but the type says it can be.
        meta_stats = dataset.meta.stats
        if meta_stats is not None:
            for key in dataset.meta.camera_keys:
                if key in dataset.meta.depth_keys:
                    continue  # Exclude depth keys from ImageNet stats
                # In multi-source intersection mode, camera_keys forwards from
                # the base sub-dataset's meta and can include image keys that
                # were DROPPED from the aggregated stats because some other
                # source didn't have them. Skip those — there's no stats slot
                # to inject into, and the policy in that scenario doesn't
                # consume them anyway (that's why the user opted into
                # intersection mode).
                if key not in meta_stats:
                    continue
                for stats_type, stats in IMAGENET_STATS.items():
                    meta_stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
) -> tuple[LeRobotDataset | MultiLeRobotDataset, LeRobotDataset | None]:
    """Create train and optional eval datasets by splitting episodes based on eval_split.

    The last ceil(n_episodes * eval_split) episodes per task are held out for evaluation.
    If eval_split == 0.0, returns (full_dataset, None).
    """
    full_dataset = make_dataset(cfg)

    if cfg.dataset.eval_split == 0.0:
        return full_dataset, None

    base_episodes = (
        full_dataset.episodes if full_dataset.episodes is not None else list(range(full_dataset.num_episodes))
    )

    episode_tasks = full_dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in base_episodes:
        task_key = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])

    if not train_episodes:
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves 0 training episodes from {len(base_episodes)} total."
        )

    logging.info(
        f"Train/eval split: {len(train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, {len(task_to_episodes)} tasks)"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta)

    train_image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return train_dataset, eval_dataset
