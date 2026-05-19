#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Intervention-driven policy rollout config.

Drives `lerobot-eval`'s intervention path: per-scenario automated supervision
of an SA-wrapped policy via the `InterventionController` state machine.
Setting `EvalPipelineConfig.intervention` to a non-None instance switches
lerobot-eval from passive evaluation into intervention-recording mode.

Two intervention methods:
* `"rrt"` — uses the SA wrapper's RRTGuidanceSource planner. Records as
  `FrameSource.RRT`.
* `"oracle_goal"` — uses OracleGoalGuidanceSource, a straight-line joint-space
  interpolation from q_start to the oracle's q_goal_bias, played back
  verbatim. Records as `FrameSource.BLEND_INTERVENTION_100`.

Both methods produce dataset frames the recorder commits; the controller's
stall/collision triggers, plan-failure backoff, and per-scenario advance
are method-agnostic. See `lerobot.scripts.intervention_controller` for the
state machine.
"""

from dataclasses import dataclass


@dataclass
class InterventionConfig:
    """Knobs for the `InterventionController` state machine."""

    # Intervention method. "rrt" uses the SA wrapper's RRT-to-goal planner
    # (the default for back-compat with existing DAgger runs). "oracle_goal"
    # uses OracleGoalGuidanceSource: a straight-line joint-space interpolation
    # from q_start to q_goal_bias, played back verbatim. Both produce frames
    # the recorder commits to the dataset (RRT → FrameSource.RRT, oracle_goal
    # → FrameSource.BLEND_INTERVENTION_100).
    method: str = "rrt"
    # Stall threshold for the FIRST intervention trigger of each scenario.
    # Policy gets this many select_action calls before the controller fires.
    policy_steps_before_rrt: int = 400
    # After an intervention cycle has actually executed, check the policy's
    # progress more often: pick a random threshold in
    # [policy_steps_between_rrt_min, policy_steps_between_rrt_max] for each
    # subsequent trigger. Set min == max to disable randomization.
    policy_steps_between_rrt_min: int = 80
    policy_steps_between_rrt_max: int = 120
    # Random number of waypoints to play back per intervention cycle, drawn
    # from [rrt_steps_min, rrt_steps_max]. After this many steps the
    # controller auto-cancels and hands control back to the policy.
    rrt_steps_min: int = 60
    rrt_steps_max: int = 200
    # Used iff method == "oracle_goal": number of waypoints in the
    # q_start → q_goal_bias linear interpolation chunk. The controller's
    # rrt_steps_min/max picks a target inside this chunk; choosing
    # target < chunk_steps means partial playback before cancel.
    oracle_goal_chunk_steps: int = 80
    # Hard cap on intervention cycles per scenario. Advance once hit.
    max_cycles_per_scenario: int = 10
    # Only meaningful for method == "rrt" (oracle_goal interpolation never
    # fails — there's no planner). After this many consecutive failed plans
    # the controller backs off; after max_backoff_rounds_per_scenario backoffs,
    # the scenario is abandoned.
    max_plan_failures: int = 5
    max_backoff_rounds_per_scenario: int = 3
