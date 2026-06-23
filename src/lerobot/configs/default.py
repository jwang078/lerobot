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

from dataclasses import dataclass, field

from lerobot.transforms import ImageTransformsConfig
from lerobot.utils.import_utils import get_safe_default_codec


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory for a concrete local dataset tree (e.g. 'dataset/path'). If None, local datasets are
    # looked up under $HF_LEROBOT_HOME/repo_id and Hub downloads use a revision-safe cache under $HF_LEROBOT_HOME/hub.
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False
    # Optional path to a stats.json file to use instead of the dataset's built-in meta/stats.json.
    # Useful when training multiple policies (e.g. diffusion vs pi05) with different chunk sizes
    # for relative action stats, without needing to store the dataset twice.
    stats_path: str | None = None

    # ─── Multi-dataset weighted-sampling mode ─────────────────────────────────
    # When `repo_ids` is set, training switches to multi-dataset mode:
    #   * `repo_id` must be empty (mutually exclusive with `repo_ids`).
    #   * Each entry of `repo_ids` is loaded as its own LeRobotDataset and
    #     concatenated via MultiLeRobotDataset.
    #   * `sample_weights` controls per-source sampling probability — frames
    #     from sub-dataset i are drawn with target share `sample_weights[i]`
    #     regardless of how large sub-dataset i is. Must be the same length
    #     as `repo_ids` and sum to ~1.0.
    #   * `stats_paths` parallels `repo_ids`; how those stats are used at
    #     normalization time depends on `norm_mode` below.
    #   * `norm_mode` controls how the policy's normalizer + unnormalizer get
    #     their stats out of the per-source sidecars:
    #       - "aggregated" (default): min-of-mins / max-of-maxes / count-weighted
    #         mean/std over ALL sources. One stats set, single normalization
    #         pass, both train and eval use the same. The aggregated range
    #         stretches every time intervention data adds extreme values, so
    #         base data gets COMPRESSED into a narrower normalized range.
    #       - "base_only": just use source[0]'s sidecar (the base dataset).
    #         Intervention data normalized by base stats may fall outside
    #         [-1, 1] if its raw range exceeds base's — the normalizer does
    #         NOT clip, so the policy sees out-of-bounds normalized targets.
    #         Useful for A/B testing whether aggregated-mode's base-compression
    #         is hurting training; the trade-off is that you may instead hurt
    #         training via the out-of-bounds intervention targets.
    #       - "per_source": NOT IMPLEMENTED. Old design (pre-normalize each
    #         frame using its source's stats, no-op the policy's normalizer)
    #         was removed because the "no-op" trick was fragile across
    #         `load_state_dict` (see `_stats_explicitly_provided` in
    #         `lerobot.processor.normalize_processor`) and eval-time
    #         normalization was inherently asymmetric (the live env has no
    #         source attribution).
    # In single-dataset mode (the default — `repo_ids` is None), these fields
    # are ignored entirely and behavior is byte-identical to today.
    repo_ids: list[str] | None = None
    sample_weights: list[float] | None = None
    stats_paths: list[str] | None = None
    norm_mode: str = "aggregated"

    def __post_init__(self) -> None:
        if self.episodes is not None:
            if any(ep < 0 for ep in self.episodes):
                raise ValueError(
                    f"Episode indices must be non-negative, got: {[ep for ep in self.episodes if ep < 0]}"
                )
            if len(self.episodes) != len(set(self.episodes)):
                duplicates = sorted({ep for ep in self.episodes if self.episodes.count(ep) > 1})
                raise ValueError(f"Episode indices contain duplicates: {duplicates}")
        _allowed_norm_modes = {"aggregated", "base_only", "per_source"}
        if self.norm_mode not in _allowed_norm_modes:
            raise ValueError(
                f"norm_mode must be one of {sorted(_allowed_norm_modes)}, got '{self.norm_mode}'"
            )


@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'
    add_tags: bool = True  # If True, save configuration as tags in the WandB run.


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    # Set to 0 for auto-tuning based on available CPU cores and n_episodes.
    batch_size: int = 0
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    # Defaults to True; automatically downgraded to SyncVectorEnv when batch_size=1.
    use_async_envs: bool = True

    def __post_init__(self) -> None:
        if self.batch_size == 0:
            self.batch_size = self._auto_batch_size()
        if self.batch_size > self.n_episodes:
            self.batch_size = self.n_episodes

    def _auto_batch_size(self) -> int:
        """Pick batch_size based on CPU cores, capped by n_episodes."""
        import math
        import os

        cpu_cores = os.cpu_count() or 4
        # Each async env worker needs ~1 core; leave headroom for main process + inference.
        by_cpu = max(1, math.floor(cpu_cores * 0.7))
        return min(by_cpu, self.n_episodes, 64)


@dataclass
class PeftConfig:
    # PEFT offers many fine-tuning methods, layer adapters being the most common and currently also the most
    # effective methods so we'll focus on those in this high-level config interface.

    # Either a string (module name suffix or 'all-linear'), a list of module name suffixes or a regular expression
    # describing module names to target with the configured PEFT method. Some policies have a default value for this
    # so that you don't *have* to choose which layers to adapt but it might still be worthwhile depending on your case.
    target_modules: list[str] | str | None = None

    # Names/suffixes of modules to fully fine-tune and store alongside adapter weights. Useful for layers that are
    # not part of a pre-trained model (e.g., action state projections). Depending on the policy this defaults to layers
    # that are newly created in pre-trained policies. If you're fine-tuning an already trained policy you might want
    # to set this to `[]`. Corresponds to PEFT's `modules_to_save`.
    full_training_modules: list[str] | None = None

    # The PEFT (adapter) method to apply to the policy. Needs to be a valid PEFT type.
    method_type: str = "LORA"

    # Adapter initialization method. Look at the specific PEFT adapter documentation for defaults.
    init_type: str | None = None

    # We expect that all PEFT adapters are in some way doing rank-decomposition therefore this parameter specifies
    # the rank used for the adapter. In general a higher rank means more trainable parameters and closer to full
    # fine-tuning.
    r: int = 16
