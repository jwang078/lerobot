# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import ProcessorStep, ProcessorStepRegistry

# CSV logging for debugging anchor / rel-action desync at inference time.
# Activate by setting LEROBOT_REL_ACTION_DEBUG_LOG=/path/to/file.csv before running eval.
# Each call to the relative or absolute processor appends one row.
_DEBUG_LOG_PATH = os.environ.get("LEROBOT_REL_ACTION_DEBUG_LOG")
_DEBUG_LOG_FILE: Any = None
_DEBUG_LOG_INIT = False


def _debug_log(event: str, **fields: Any) -> None:
    """Append a row to the debug CSV. Lazy-opens the file on first call."""
    global _DEBUG_LOG_FILE, _DEBUG_LOG_INIT
    if _DEBUG_LOG_PATH is None:
        return
    if not _DEBUG_LOG_INIT:
        os.makedirs(os.path.dirname(_DEBUG_LOG_PATH) or ".", exist_ok=True)
        _DEBUG_LOG_FILE = open(_DEBUG_LOG_PATH, "w", buffering=1)  # noqa: SIM115 - long-lived log handle, closed by process exit
        _DEBUG_LOG_FILE.write(
            "ts,event,state,anchor,rel_action,abs_action,queue_empty,has_action,did_cache_update\n"
        )
        _DEBUG_LOG_INIT = True

    def _fmt(v):
        if v is None:
            return ""
        if torch.is_tensor(v):
            v = v.detach().cpu().flatten().tolist()
        if isinstance(v, (list, tuple)):
            return "[" + ";".join(f"{x:.6f}" for x in v) + "]"
        return str(v)

    row = [
        f"{time.time():.6f}",
        event,
        _fmt(fields.get("state")),
        _fmt(fields.get("anchor")),
        _fmt(fields.get("rel_action")),
        _fmt(fields.get("abs_action")),
        _fmt(fields.get("queue_empty")),
        _fmt(fields.get("has_action")),
        _fmt(fields.get("did_cache_update")),
    ]
    _DEBUG_LOG_FILE.write(",".join(row) + "\n")


# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
]


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


@ProcessorStepRegistry.register("delta_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    # Optional reference to the policy running inference. Used to detect chunk
    # boundaries (via the policy's action queue) so the cached anchor state is
    # only refreshed when a fresh chunk is about to be predicted. Not serialized.
    _policy: Any = field(default=None, init=False, repr=False)

    def attach_policy(self, policy: Any) -> None:
        """Attach the policy running inference so anchor caching can be gated on
        chunk boundaries. Safe to call multiple times; pass None to detach.
        """
        self._policy = policy

    def _policy_queue_empty(self) -> bool:
        """Return True if the attached policy has no buffered chunk actions,
        meaning a fresh chunk is about to be predicted on the next select_action.

        When no policy is attached (e.g. training, unit tests, one-off calls),
        returns True so the cache is always refreshed — backward-compatible.

        Supports both common queue conventions used across policy modules:
        - ``self._action_queue`` (deque): act, pi0, pi05, pi0_fast, groot
        - ``self._queues[ACTION]`` (deque): diffusion, smolvla, vqbet,
          wall_x, xvla, multi_task_dit, tdmpc

        Walks through wrappers exposing ``inner_policy`` (e.g.
        ``SharedAutonomyPolicyWrapper``, ``TemporalEnsemblePolicyWrapper``)
        and returns the queue state at the *outermost* level that has one.
        This lets wrappers that override chunk-boundary semantics (like TE,
        which exposes its own ``_action_queue`` proxy backed by an intra-
        chunk counter) intercept the check before the walk descends into
        the underlying policy. Backward-compatible: wrappers without a
        queue attribute continue to forward through.
        """
        policy = self._policy
        if policy is None:
            return True
        while True:
            action_queue = getattr(policy, "_action_queue", None)
            if action_queue is not None:
                return len(action_queue) == 0
            queues_dict = getattr(policy, "_queues", None)
            if queues_dict is not None:
                action_q = queues_dict.get(ACTION) if hasattr(queues_dict, "get") else None
                if action_q is not None:
                    return len(action_q) == 0
            if hasattr(policy, "inner_policy"):
                policy = policy.inner_policy
            else:
                return True

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        has_action = transition.get(TransitionKey.ACTION) is not None

        # Cache the anchor state for the paired AbsoluteActionsProcessorStep.
        # During training the transition carries an action and we always refresh
        # (harmless — postprocessor isn't called in that path). During chunked
        # inference the action is None; only refresh when the policy's action
        # queue is empty, i.e. a fresh chunk is about to be predicted. Otherwise
        # mid-chunk preprocessor calls would overwrite the anchor with stale
        # per-step state, producing wrong absolute actions for chunk[1..n-1].
        queue_empty = self._policy_queue_empty()
        did_cache_update = state is not None and (has_action or queue_empty)
        if did_cache_update:
            self._last_state = state

        # Log every preprocessor call so we can audit cache-update timing vs the
        # state the model is conditioning on this same call.
        _debug_log(
            "preproc",
            state=state[..., -1, :] if state is not None and state.ndim == 3 else state,
            anchor=self._last_state[..., -1, :]
            if self._last_state is not None and self._last_state.ndim == 3
            else self._last_state,
            queue_empty=queue_empty,
            has_action=has_action,
            did_cache_update=did_cache_update,
        )

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        # During training the dataset provides state as (B, n_obs_steps, state_dim).
        # to_relative_actions expects (B, state_dim); use the most recent timestep as the base,
        # consistent with UMI/OpenPI which anchor all actions relative to the current state.
        if state.ndim == 3:
            state = state[:, -1, :]

        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts relative actions back to absolute actions (action += state) for all dimensions.

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted relative offsets are converted back to absolute positions for execution.
    Reads the cached state from its paired RelativeActionsProcessorStep.

    Attributes:
        enabled: Whether to apply the absolute conversion.
        relative_step: Reference to the paired RelativeActionsProcessorStep that caches state.
    """

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep "
                "but relative_step is None. Ensure relative_step is set when constructing the postprocessor."
            )

        if self.relative_step._last_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        abs_action = to_absolute_actions(action, self.relative_step._last_state, mask)
        new_transition[TransitionKey.ACTION] = abs_action

        anchor = self.relative_step._last_state
        if anchor is not None and anchor.ndim == 3:
            anchor = anchor[..., -1, :]
        _debug_log(
            "postproc",
            anchor=anchor,
            rel_action=action,
            abs_action=abs_action,
        )

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
