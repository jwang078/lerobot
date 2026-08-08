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
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
import torch

from lerobot.configs import (
    DEFAULT_DEPTH_UNIT,
    DEPTH_METER_UNIT,
    DepthEncoderConfig,
)

from .dataset_metadata import LeRobotDatasetMetadata
from .depth_utils import MM_PER_METRE, dequantize_depth
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .utils import resolve_episode_indices
from .video_utils import decode_video_frames


class DatasetReader:
    """Encapsulates read-side state and methods for LeRobotDataset.

    Owns: hf_dataset, _absolute_to_relative_idx, delta_indices.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        exclude_features: list[str] | None = None,
    ):
        """Initialize the reader with metadata, filtering, and transform config.

        The HF dataset is not loaded here — call :meth:`try_load` or
        :meth:`load_and_activate` afterward.

        Args:
            meta: Dataset metadata instance.
            root: Local dataset root directory.
            episodes: Optional list of episode indices to select. ``None``
                means all episodes.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            video_backend: Video decoding backend identifier.
            delta_timestamps: Optional dict mapping feature keys to lists of
                relative timestamp offsets for temporal context windows.
            image_transforms: Optional torchvision v2 transform applied to
                visual features.
            return_uint8: If True, return RGB video frames as raw uint8 tensors
                instead of normalized float32.
            depth_output_unit: Physical unit depth maps are dequantized to
                (``"m"`` or ``"mm"``). Defaults to ``"mm"``.
            exclude_features: Optional dataset feature keys to fully skip at
                load time: parquet-embedded columns are dropped from the HF
                dataset view (so row access never materializes them — for
                image-mode datasets a single full-row read otherwise decodes
                every embedded PNG) and video keys are skipped in the decode
                path. Used by the train factory to drop camera features a
                state-only policy doesn't consume.
        """
        self._meta = meta
        self.root = root
        self.episodes = resolve_episode_indices(episodes, meta.total_episodes)
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self._image_transforms = image_transforms
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit
        self._exclude_features: set[str] = set(exclude_features or [])

        self.hf_dataset: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None

        # Setup delta_indices (doesn't depend on hf_dataset)
        self.delta_indices = None
        if delta_timestamps is not None:
            if self._exclude_features:
                delta_timestamps = {
                    k: v for k, v in delta_timestamps.items() if k not in self._exclude_features
                }
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

        self._depth_encoder_configs: dict[str, DepthEncoderConfig] = {
            vid_key: DepthEncoderConfig.from_video_info(self._meta.features[vid_key].get("info"))
            for vid_key in self._meta.depth_keys
        }

        # Get the input unit of each depth feature stored as raw images.
        self._image_depth_units: dict[str, str | None] = {
            key: (self._meta.features[key].get("info") or {}).get("depth_unit")
            for key in self._meta.depth_keys
            if key in self._meta.image_keys
        }

    def _apply_feature_exclusions(self) -> None:
        """Drop excluded parquet columns from the loaded HF dataset view.

        Cheap schema-level operation; the point is that a subsequent row
        access (``hf_dataset[idx]``) no longer materializes the excluded
        columns — the dominant cost for image-mode datasets whose camera
        frames are PNG bytes embedded in the parquet rows.
        """
        if self.hf_dataset is None or not self._exclude_features:
            return
        drop = [c for c in self.hf_dataset.column_names if c in self._exclude_features]
        if drop:
            self.hf_dataset = self.hf_dataset.remove_columns(drop)

    @property
    def _active_video_keys(self) -> list[str]:
        """Video keys minus the excluded ones (excluded videos are never decoded)."""
        if not self._exclude_features:
            return self._meta.video_keys
        return [k for k in self._meta.video_keys if k not in self._exclude_features]

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform applied to visual observations."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self._image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        """Remove the transform applied to visual observations."""
        self._image_transforms = None

    def try_load(self) -> bool:
        """Attempt to load from local cache. Returns True if data is sufficient."""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._apply_feature_exclusions()
        self._build_index_mapping()
        return True

    def load_and_activate(self) -> None:
        """Load HF dataset from disk and build index mapping. Call after data is on disk."""
        self.hf_dataset = self._load_hf_dataset()
        self._apply_feature_exclusions()
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """Build absolute-to-relative index mapping from loaded hf_dataset."""
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            indices = self.hf_dataset.data.column("index").to_numpy()
            self._absolute_to_relative_idx = dict(zip(indices.tolist(), range(len(indices)), strict=True))

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    def _load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self._meta.features)
        self._validate_language_columns_declared(features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _validate_language_columns_declared(self, features: datasets.Features) -> None:
        """Require language columns stored in Parquet to be declared in metadata."""
        # Leave empty datasets to fail through the normal loading path.
        try:
            sample = next((self.root / "data").glob("*/*.parquet"))
        except StopIteration:
            return

        from pyarrow import parquet as _pq  # noqa: PLC0415

        # LeRobot shards are schema-uniform, so one schema represents the dataset.
        schema_names = set(_pq.read_schema(sample).names)
        from .language import LANGUAGE_COLUMNS  # noqa: PLC0415

        missing = sorted(set(LANGUAGE_COLUMNS) & schema_names - set(features))
        if missing:
            raise ValueError(
                f"Dataset Parquet files contain language feature(s) missing from metadata: {missing}. "
                "Metadata must describe the stored data; add the entries returned by "
                "lerobot.datasets.language.language_feature_info() to meta/info.json['features'] "
                "or rerun the annotation pipeline's metadata synchronization."
            )

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """Return deduplicated file paths (data + video) for selected episodes.

        Used to build the ``allow_patterns`` list for ``snapshot_download``.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices for delta timestamps."""
        # Scalar reads, not a full row materialization — see the note in
        # `_query_videos` (a metadata row can be ~1 MB / ~19 ms).
        ep_start = self._meta.episode_scalar(ep_idx, "dataset_from_index")
        ep_end = self._meta.episode_scalar(ep_idx, "dataset_to_index")
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self._active_video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys."""
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault.
        """

        # NOTE: read per-episode scalars via `episode_scalar`, never
        # `self._meta.episodes[ep_idx]` — the latter materializes the entire
        # metadata row, which is ~1 MB (and ~19 ms) on datasets that store
        # large per-episode blobs (e.g. SplatSim's splatsim_robot_config).
        # That single line cost more than the video decode it precedes.
        def _decode_single(vid_key: str, query_ts: list[float]) -> tuple[str, torch.Tensor]:
            from_timestamp = self._meta.episode_scalar(ep_idx, f"videos/{vid_key}/from_timestamp")
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
                is_depth=vid_key in self._meta.depth_keys,
            )
            if vid_key in self._meta.depth_keys:
                depth_encoder = self._depth_encoder_configs[vid_key]
                frames = dequantize_depth(
                    frames,
                    depth_min=depth_encoder.depth_min,
                    depth_max=depth_encoder.depth_max,
                    shift=depth_encoder.shift,
                    use_log=depth_encoder.use_log,
                    output_unit=self._depth_output_unit,
                )
            return vid_key, frames.squeeze(0)

        items = list(query_timestamps.items())

        # Single camera: no threading overhead
        if len(items) <= 1:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        # Multi-camera. Fan out ONLY in the main process. Inside a DataLoader
        # worker the parallelism already exists across samples, and video
        # decoders are themselves internally multithreaded — nesting the two
        # oversubscribes the CPU badly (measured: 4 cameras took 182 ms
        # threaded vs 47 ms serial, ~3.9x slower, with 4 workers x 4 cameras
        # of AV1 decode competing for 24 cores). Serial per sample is the
        # right choice there; the workers keep the cores busy.
        if torch.utils.data.get_worker_info() is not None:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = [pool.submit(_decode_single, k, ts) for k, ts in items]
            return dict(f.result() for f in futures)

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded.

        ``idx`` is a *relative* index into the (possibly episode-filtered)
        HF dataset, **not** the absolute frame index stored in the ``index``
        column.  The absolute index is retrieved from the row itself.
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self._active_video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            for cam in self._meta.camera_keys:
                if cam in self._meta.depth_keys:
                    continue
                # Excluded camera features never enter the item (see
                # exclude_features) — skip rather than KeyError.
                if cam not in item:
                    continue
                item[cam] = self._image_transforms(item[cam])

        # Convert depth features to the output unit.
        for key, stored_unit in self._image_depth_units.items():
            if key in item and stored_unit is not None and stored_unit != self._depth_output_unit:
                item[key] = (
                    item[key] * MM_PER_METRE if stored_unit == DEPTH_METER_UNIT else item[key] / MM_PER_METRE
                )

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        return item
