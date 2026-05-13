#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Last-mile oracle override (DEBUG / diagnostic).

Wraps a chunk-predicting policy with an eval-time override that pulls the
commanded joint targets toward ``oracle_env_config.task.q_goal_bias`` when the
robot's joint state is within ``joint_distance_threshold`` of the goal. Lets us
test the hypothesis that the residual eval failure is *last-mile precision*
specifically (off-trajectory near-goal covariate shift) rather than approach.

This is a diagnostic, not a deployable mechanism — it depends on oracle env
info that won't be available outside the splatsim eval-benchmark setup. When
the diagnostic has served its purpose, delete this file, its wrapper, the
factory wiring, and the policy-config fields that reference it.
"""

from dataclasses import dataclass


@dataclass
class LastMileDebugConfig:
    """Configuration for the oracle last-mile-override debug wrapper.

    When enabled, on every ``select_action`` the wrapper:
      1. Reads ``target_ee_pos`` and ``current_ee_pos`` from
         ``batch["oracle_env_config"]`` (and the ``q_goal_bias`` joint config
         under ``task``).
      2. Computes the L2 *end-effector position* distance (meters) between
         the current and target EE poses. This matches the simulator's own
         success-criterion metric (closer to the actual notion of "near goal"
         than joint-space distance, which can be huge even when the EE is
         near goal due to kinematic redundancy — multiple joint configs
         producing the same EE pose).
      3. If ``ee_distance < ee_distance_threshold``, blends the inner
         policy's commanded joint targets toward ``q_goal_bias`` with weight
         ``blend_alpha`` (1.0 = full override, 0.0 = pure policy). The
         gripper dimension is *never* overridden — it always reflects the
         policy's command.
      4. Otherwise returns the inner policy's action unchanged.

    Caveat: when ``blend_alpha`` is high and the policy's current joint
    config is in a different IK branch than ``q_goal_bias``, the override
    will command a large joint-space jump (potential teleport). The wrapper
    logs a warning when this is detected so it's visible in the eval log.
    """

    enabled: bool = False
    # EE position distance (meters) below which the override fires. Default
    # matches a few × the typical success tolerance (e.g. 3 cm tol → 0.05 m
    # threshold) so the override fires shortly before the success region.
    ee_distance_threshold: float = 0.05
    # 0.0 = no override (sanity check), 1.0 = command exactly q_goal_bias on the
    # joints whenever in-range. Intermediate values blend.
    blend_alpha: float = 1.0
