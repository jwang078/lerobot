#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Slice observation feature tensors to a subset of last-axis dimensions.

Motivation: a constant / unused observation dim (e.g., the always-0 gripper
joint in the planar_3joint env) has min == max in the dataset stats, which
poisons MIN_MAX normalization (0/0 → NaN or a degenerate constant) AND
wastes a policy input slot on a dead feature. Dropping the dim at the
policy input boundary avoids both, without needing to re-record the dataset.

Placement: insert as the FIRST step in the preprocessor pipeline so
downstream steps (normalizer, encoders, ...) see the sliced shape.

`dims`: {observation_key: [dim_indices_to_keep_on_last_axis]}.
Keys not present in `dims` pass through untouched; keys listed in `dims`
but absent from the batch are silently skipped (so this step can be
declared once in a factory and stay inert until a caller populates dims).
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="select_observation_dims_processor")
class SelectObservationDimsProcessorStep(ObservationProcessorStep):
    """Slice specific observation-tensor keys to a subset of last-axis dims.

    Attributes:
        dims: Mapping from observation key (e.g., ``"observation.state"``) to
              a list of dim indices to keep on that tensor's LAST axis (so
              this works uniformly for [B, D], [B, T, D], etc). Empty dict is
              a no-op. Keys listed here that don't appear in a given batch
              are silently skipped.
    """

    dims: dict[str, list[int]] = field(default_factory=dict)

    def observation(self, observation):
        if not self.dims:
            return observation
        out = {}
        for key, value in observation.items():
            idx_list = self.dims.get(key)
            if idx_list is None or not isinstance(value, torch.Tensor):
                out[key] = value
                continue
            idx = torch.as_tensor(list(idx_list), dtype=torch.long, device=value.device)
            out[key] = value.index_select(dim=-1, index=idx)
        return out

    def get_config(self) -> dict[str, Any]:
        # Serialize as plain lists so the saved processor config round-trips
        # through JSON without dataclass-specific machinery.
        return {"dims": {k: list(v) for k, v in self.dims.items()}}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Shrink the schema shape of each sliced observation to match the
        runtime slice, so downstream steps (normalizer, policy) see the
        correct dim count.

        For each key present in both ``dims`` and the observation feature
        map, replaces the feature's ``shape`` last-dim with ``len(dims[key])``.
        Other axes are preserved verbatim.
        """
        if not self.dims:
            return features
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = features.copy()
        obs = features.get(PipelineFeatureType.OBSERVATION, {})
        new_obs = dict(obs)
        for key, idx_list in self.dims.items():
            if key not in new_obs:
                continue
            feat = new_obs[key]
            old_shape = tuple(feat.shape)
            if not old_shape:
                continue
            new_shape = (*old_shape[:-1], len(idx_list))
            new_obs[key] = PolicyFeature(type=feat.type, shape=new_shape)
        new_features[PipelineFeatureType.OBSERVATION] = new_obs
        return new_features
