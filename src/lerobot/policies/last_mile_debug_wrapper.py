#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Last-mile oracle override wrapper (DEBUG / diagnostic).

Tests whether the residual eval failure is *last-mile precision* by overriding
the policy's commanded joint targets toward ``oracle_env_config.task.q_goal_bias``
when the robot is close (in EE space) to the goal.

Architecturally this is a thin tracking shim around the inner policy:

* ``select_action`` is a passthrough that records per-episode EE distance to
  the goal and exposes the override target via ``pending_override``.
* ``apply_override(action_raw, raw_obs_state)`` is called by ``lerobot_eval``
  AFTER the postprocessor has converted ``action_norm`` to raw absolute joint
  commands. The override is applied in raw joint space — no fragile round-trip
  through inverse-normalization, which would silently misscale on delta-action
  policies (delta-normalization stats applied to an absolute joint vector).
* ``reset`` logs a per-episode summary so users can see why the override did
  or didn't fire even when no successes occurred.

This is a diagnostic, not a deployable mechanism — depends on oracle env info
(``current_ee_pos`` / ``target_ee_pos`` / ``q_goal_bias``) that only splatsim
exposes via ``get_env_config``. When the diagnostic has served its purpose,
delete this file, its config, the factory wiring, and the eval-script
integration block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy

if TYPE_CHECKING:
    from lerobot.configs.last_mile_debug import LastMileDebugConfig

logger = logging.getLogger(__name__)

# Key under which lerobot-eval stashes the raw (pre-normalization) joint state
# so this wrapper can read it without re-implementing inverse normalization.
RAW_STATE_KEY = "_raw_obs_state_for_lastmile"


@dataclass
class _PendingOverride:
    """What ``apply_override`` should do to the raw joint command this step.

    ``apply_override`` consumes this once per step (always — the wrapper
    re-computes it on every ``select_action`` so a stale value from a previous
    step is overwritten before it can be acted on).
    """

    q_goal_bias: Tensor  # shape (n_joints,), raw joint angles (radians)
    alpha: float  # 0.0 = no override, 1.0 = full override toward q_goal_bias
    ee_dist: float  # for logging only


class LastMileDebugWrapper(PreTrainedPolicy):
    """Track EE-space distance to the goal and stage an override for lerobot_eval.

    select_action is a passthrough returning the inner policy's normalized
    action. The wrapper exposes ``pending_override`` so lerobot_eval can apply
    the override after the postprocessor has produced raw joint commands.
    """

    config_class = PreTrainedConfig
    name = "last_mile_debug_wrapper"

    def __init__(
        self,
        inner_policy: PreTrainedPolicy,
        debug_cfg: LastMileDebugConfig,
    ) -> None:
        nn.Module.__init__(self)
        self.config: PreTrainedConfig = inner_policy.config
        self.inner_policy = inner_policy
        self.debug_cfg = debug_cfg

        # Set by select_action each step, consumed by apply_override.
        # None means "no override this step" (out of range, or missing inputs).
        self.pending_override: _PendingOverride | None = None

        # Per-episode counters and one-shot logging flags.
        self._fire_count: int = 0
        self._step_count: int = 0
        self._min_ee_dist: float = float("inf")
        self._min_ee_dist_step: int = -1
        self._missing_oracle_warned: bool = False
        self._missing_raw_state_warned: bool = False
        self._saw_oracle_logged: bool = False
        self._saw_raw_state_logged: bool = False

        logger.info(
            "LastMileDebugWrapper [DIAGNOSTIC]: enabled=%s, "
            "ee_distance_threshold=%.4f m, blend_alpha=%.2f. "
            "Triggers on EE position distance (oracle_env_config.current_ee_pos "
            "vs task.target_ee_pos). Override is applied in lerobot_eval AFTER "
            "the postprocessor (raw joint space). Remove this wrapper once the "
            "diagnostic is complete.",
            debug_cfg.enabled,
            debug_cfg.ee_distance_threshold,
            debug_cfg.blend_alpha,
        )

    # ------------------------------------------------------------------
    # Inference path
    # ------------------------------------------------------------------

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Passthrough: returns inner action_norm. Side effect: updates
        ``self.pending_override`` based on the current EE distance.

        Reads oracle info and raw obs.state from the batch; logs the first
        successful read per episode so the user can confirm the wrapper sees
        what it needs.
        """
        action_norm = self.inner_policy.select_action(batch, **kwargs)
        # Always clear pending override first — a stale one from a previous
        # step would otherwise leak into the next apply_override call if any
        # branch below early-returns.
        self.pending_override = None

        if not self.debug_cfg.enabled:
            return action_norm
        self._step_count += 1

        oracle_cfg = batch.get("oracle_env_config")
        q_goal_bias = self._extract_q_goal_bias(oracle_cfg)
        target_ee_pos = self._extract_target_ee_pos(oracle_cfg)
        current_ee_pos = self._extract_current_ee_pos(oracle_cfg)
        if q_goal_bias is None or target_ee_pos is None or current_ee_pos is None:
            if not self._missing_oracle_warned:
                missing = []
                if q_goal_bias is None:
                    missing.append("task.q_goal_bias")
                if target_ee_pos is None:
                    missing.append("task.target_ee_pos")
                if current_ee_pos is None:
                    missing.append("current_ee_pos (top-level)")
                logger.warning(
                    "LastMileDebugWrapper: oracle missing fields %s. "
                    "Override will not fire. Check env.get_env_config().",
                    missing,
                )
                self._missing_oracle_warned = True
            return action_norm
        if not self._saw_oracle_logged:
            logger.info(
                "LastMileDebugWrapper: ✓ oracle ready. q_goal_bias=%s, "
                "target_ee_pos=%s, initial current_ee_pos=%s.",
                q_goal_bias.detach().cpu().numpy().round(3).tolist(),
                target_ee_pos.detach().cpu().numpy().round(3).tolist(),
                current_ee_pos.detach().cpu().numpy().round(3).tolist(),
            )
            self._saw_oracle_logged = True

        raw_state = batch.get(RAW_STATE_KEY)
        if raw_state is None:
            if not self._missing_raw_state_warned:
                logger.warning(
                    "LastMileDebugWrapper: no raw obs.state in batch under "
                    f"'{RAW_STATE_KEY}'. lerobot-eval must inject it before "
                    "the policy preprocessor. Override will not fire."
                )
                self._missing_raw_state_warned = True
            return action_norm
        if not self._saw_raw_state_logged:
            logger.info(
                "LastMileDebugWrapper: ✓ raw obs.state received (shape=%s). "
                "Triggering on EE position distance (target_ee_pos − current_ee_pos).",
                tuple(raw_state.shape),
            )
            self._saw_raw_state_logged = True

        # EE-space trigger: meters between current and target EE position.
        ee_dist = float(
            torch.linalg.vector_norm(
                current_ee_pos.to(dtype=torch.float32) - target_ee_pos.to(dtype=torch.float32)
            ).item()
        )

        if ee_dist < self._min_ee_dist:
            self._min_ee_dist = ee_dist
            self._min_ee_dist_step = self._step_count

        if ee_dist >= self.debug_cfg.ee_distance_threshold:
            return action_norm

        # In range — stage the override for apply_override to consume.
        self.pending_override = _PendingOverride(
            q_goal_bias=q_goal_bias,
            alpha=float(self.debug_cfg.blend_alpha),
            ee_dist=ee_dist,
        )
        return action_norm

    def apply_override(self, action_raw: Tensor, raw_obs_state: Tensor | None) -> Tensor:
        """Apply the staged override (if any) to a raw absolute-joint action.

        Called by lerobot_eval AFTER the postprocessor. Operates entirely in
        raw joint space, so it works identically for delta-action and
        absolute-action policies (no inverse normalization needed).

        Includes a hard safety gate against kinematic-redundancy teleports:
        if the override would command a joint jump > 0.5 rad L2 under
        ``alpha > 0``, refuses and logs (because pybullet's integrator
        can't handle teleports and will SIGABRT mid-step).

        Returns the (possibly overridden) raw joint command.
        """
        pending = self.pending_override
        if pending is None:
            return action_raw
        if raw_obs_state is None:
            # Shouldn't happen if select_action checks pass, but be defensive.
            return action_raw

        n_joints = int(pending.q_goal_bias.shape[-1])
        alpha = pending.alpha
        q_goal_bias_t = pending.q_goal_bias.to(action_raw.device, dtype=action_raw.dtype)
        raw_state_joints = raw_obs_state.reshape(-1)[:n_joints].to(action_raw.device, dtype=action_raw.dtype)
        joint_jump = float(torch.linalg.vector_norm(raw_state_joints - q_goal_bias_t).item())

        # PYBULLET SAFETY GATE — refuse giant joint teleports. The policy
        # itself rarely commands more than ~0.5 rad L2 per step at chunk-
        # boundary speed; anything beyond that is a kinematic-redundancy
        # teleport that crashes pybullet's physics integrator (SIGABRT with
        # no Python traceback because the crash is C-level).
        max_safe_joint_jump = 0.5
        if alpha > 0.0 and joint_jump > max_safe_joint_jump:
            print(
                f"⚠ LastMileDebugWrapper: REFUSING override at step "
                f"{self._step_count} — joint_jump={joint_jump:.3f} rad > safety "
                f"cap {max_safe_joint_jump:.2f}. Current joint config is in a "
                f"different IK branch than q_goal_bias; alpha={alpha:.2f} "
                f"would cause a violent teleport that crashes pybullet. "
                f"EE distance was {pending.ee_dist:.4f} m (threshold "
                f"{self.debug_cfg.ee_distance_threshold:.4f}). Passing through "
                f"inner policy action unchanged.",
                flush=True,
            )
            return action_raw

        # Print BEFORE applying so the message reaches stderr even if env.step
        # on this commanded action triggers a downstream native abort.
        if self._fire_count == 0 or self._fire_count % 25 == 0:
            print(
                f"LastMileDebugWrapper: override applied step={self._step_count} "
                f"ee_dist={pending.ee_dist:.4f}m joint_jump={joint_jump:.3f}rad "
                f"alpha={alpha:.2f}",
                flush=True,
            )

        action_raw_overridden = action_raw.clone()
        action_raw_overridden[..., :n_joints] = (1.0 - alpha) * action_raw[
            ..., :n_joints
        ] + alpha * q_goal_bias_t
        self._fire_count += 1
        return action_raw_overridden

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        # Override is applied post-postprocessor on the executed action only.
        # Chunk forecasts returned to outer consumers (e.g. SA) are untouched.
        return self.inner_policy.predict_action_chunk(batch, **kwargs)

    def reset(self) -> None:
        self.inner_policy.reset()
        if self.debug_cfg.enabled and self._step_count > 0:
            if self._min_ee_dist == float("inf"):
                logger.info(
                    "LastMileDebugWrapper episode summary: %d steps, "
                    "EE distance never computed (oracle and/or raw state "
                    "missing — see warnings above).",
                    self._step_count,
                )
            else:
                verdict = "OVERRIDE APPLIED" if self._fire_count > 0 else "NEVER APPLIED"
                logger.info(
                    "LastMileDebugWrapper episode summary: %d steps, %s "
                    "(applied %d times). Closest EE approach: %.4f m at step %d "
                    "(threshold=%.4f m). %s",
                    self._step_count,
                    verdict,
                    self._fire_count,
                    self._min_ee_dist,
                    self._min_ee_dist_step,
                    self.debug_cfg.ee_distance_threshold,
                    (
                        "→ try raising --last-mile-debug-threshold above this min to see the override fire."
                        if self._fire_count == 0
                        else ""
                    ),
                )
        self._fire_count = 0
        self._step_count = 0
        self._min_ee_dist = float("inf")
        self._min_ee_dist_step = -1
        self._saw_oracle_logged = False
        self._saw_raw_state_logged = False
        self.pending_override = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_q_goal_bias(oracle_cfg) -> Tensor | None:
        if not isinstance(oracle_cfg, dict):
            return None
        task = oracle_cfg.get("task")
        if not isinstance(task, dict):
            return None
        q = task.get("q_goal_bias")
        if q is None:
            return None
        return torch.as_tensor(q, dtype=torch.float32)

    @staticmethod
    def _extract_target_ee_pos(oracle_cfg) -> Tensor | None:
        if not isinstance(oracle_cfg, dict):
            return None
        task = oracle_cfg.get("task")
        if not isinstance(task, dict):
            return None
        p = task.get("target_ee_pos")
        if p is None:
            return None
        return torch.as_tensor(p, dtype=torch.float32)

    @staticmethod
    def _extract_current_ee_pos(oracle_cfg) -> Tensor | None:
        if not isinstance(oracle_cfg, dict):
            return None
        p = oracle_cfg.get("current_ee_pos")
        if p is None:
            return None
        return torch.as_tensor(p, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def forward(self, batch, **kwargs):
        return self.inner_policy.forward(batch, **kwargs)

    def get_optim_params(self):
        return self.inner_policy.get_optim_params()

    def train(self, mode: bool = True):
        self.inner_policy.train(mode)
        return self

    def eval(self):
        self.inner_policy.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self.inner_policy.parameters(recurse)

    def to(self, *args, **kwargs):
        self.inner_policy.to(*args, **kwargs)
        return self

    def use_original_modules(self):
        if hasattr(self.inner_policy, "use_original_modules"):
            self.inner_policy.use_original_modules()
