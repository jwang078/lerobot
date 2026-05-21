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
    # No-progress trigger (in addition to the step-count stall + collision
    # triggers). Fires when the env's `info["position_error_m"]` — distance
    # from the EE to the goal pose — hasn't improved by at least
    # `no_progress_min_decrease_m` for `no_progress_window_steps` consecutive
    # policy steps. Catches the "policy is drifting confidently in the wrong
    # direction" failure mode much earlier than the time-based stall trigger.
    #
    # Internally uses the same anchor-based algorithm as the last_mile
    # wrapper's NoEEProgressDetector (shared via
    # lerobot.policies.last_mile.detectors.EEDistanceProgressTracker), so
    # tuning the params here uses the same semantics: anchor moves down with
    # the EE on progress, resets if the robot enters a repositioning epoch.
    #
    # Disabled by default (window=0). Recommended starting values when
    # enabling: window=50, min_decrease=0.005m, warmup=30. Set
    # `no_progress_warmup_steps` higher (e.g. 100) if the policy needs time
    # to start moving from rest.
    #
    # Silently no-ops if the env doesn't surface `position_error_m` in info.
    no_progress_window_steps: int = 0
    no_progress_min_decrease_m: float = 0.005
    no_progress_warmup_steps: int = 30
    no_progress_reposition_grace_steps: int = 30
    no_progress_reposition_turnaround_m: float = 0.01

    # Orientation-axis no-progress trigger. Mirrors the position trigger but
    # watches `info["orientation_error_deg"]` instead. Catches the
    # "wrist twisting wrong" failure mode that position-only triggers miss
    # — e.g., a policy that gets to the right position but can't align the
    # gripper for a precision grasp. Both triggers can be enabled together
    # (independent state, OR'd into the same intervention fire).
    #
    # Disabled by default (window=0). Recommended starting values when
    # enabling: window=50, min_decrease=1.0deg, warmup=30. Silently no-ops
    # if the env doesn't surface `orientation_error_deg` in info.
    no_progress_orientation_window_steps: int = 0
    no_progress_orientation_min_decrease_deg: float = 1.0
    no_progress_orientation_warmup_steps: int = 30
    no_progress_orientation_reposition_grace_steps: int = 30
    no_progress_orientation_reposition_turnaround_deg: float = 2.0
    # Hard cap on intervention cycles per scenario. Advance once hit.
    max_cycles_per_scenario: int = 10
    # Only meaningful for method == "rrt" (oracle_goal interpolation never
    # fails — there's no planner). After this many consecutive failed plans
    # the controller backs off; after max_backoff_rounds_per_scenario backoffs,
    # the scenario is abandoned.
    max_plan_failures: int = 5
    max_backoff_rounds_per_scenario: int = 3
