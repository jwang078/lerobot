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
    # When True, SCHEDULED cadence triggers ("time stall": the
    # policy_steps_before/between_rrt budget elapsing, NOT an actual stall)
    # take the no-lookback path: RRT plans from the LIVE state with ruckig
    # seeded from the robot's recent velocity, so the recorded correction
    # starts velocity-continuous (decelerate-and-redirect) instead of the
    # rewind+teleport+cold-start of the lookback path. The scheduled trigger
    # fires mid-healthy-motion, where the rewind serves no purpose — it only
    # erases the moving-handoff supervision. Genuine stalls
    # (joint_stall / no_progress*) still rewind: the robot is at rest there,
    # so there is no velocity to preserve and rewinding to a pre-mistake
    # state is the desired semantics. Default False = historical behavior.
    scheduled_trigger_no_lookback: bool = False
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
    # Stuck-detection gate for the planner-side `in_collision` signal.
    # When the controller's planner-clearance check (using
    # `rrt_in_progress_obstacle_clearance` / self_collision_clearance from
    # SharedAutonomyConfig) reports the robot is in (or near) collision,
    # the intervention trigger only fires if the robot is ALSO stuck —
    # i.e., the joint-L2 |Δstate| over the last `stuck_consecutive_ticks`
    # ticks has stayed below `stuck_threshold_rad_per_tick` every tick.
    # Together the two conditions distinguish:
    #   * WEDGE (true positive)  : in_collision + can't move → fire retry.
    #     PD controller pushing but contact stops the joint; |Δq| ≈ 0
    #     for many ticks. Recording this without intervention leaks
    #     "stuck pose" frames into the dataset.
    #   * APPROACH-NEAR-OBSTACLE (false positive) : in_collision + moving.
    #     Trajectory passes near scene geometry (often the GOAL is itself
    #     within clearance of the lever) but the robot is still tracking
    #     the command. No intervention needed.
    # Gate applies to BOTH: (a) the new-cycle "obstacle_collision" trigger
    # (controller observes policy heading into collision) and (b) the
    # mid-RRT-execution retry (controller observes RRT chunk colliding).
    # Set `stuck_threshold_rad_per_tick = 0` to disable the gate and
    # restore the legacy "fire on every in_collision tick" behavior.
    # Defaults: 0.005 rad/tick ≈ 0.15 rad/s at 30 Hz (an order of magnitude
    # below normal commanded motion ~0.02-0.05 rad/tick), 3 ticks ≈ 100 ms
    # (quick to react when actually stuck without burst-firing on a single
    # slow frame).
    stuck_threshold_rad_per_tick: float = 0.005
    stuck_consecutive_ticks: int = 3

    # Joint-stall trigger: fires an intervention when the robot's joint state
    # has barely moved for `joint_stall_window_steps` consecutive policy-mode
    # ticks (per-tick joint-L2 |Δstate| < joint_stall_threshold_rad). Purely
    # kinematic — no reference to goal / EE distance — so it fires even when
    # the policy legitimately needs to move AWAY from the goal to find a valid
    # path around an obstacle. Complements the goal-relative no-progress
    # trigger (which is disabled by default; window=0).
    #
    # Fires on POLICY mode only (mode == IDLE). During RRT execution the
    # counter is reset — Ruckig's smooth deceleration near a waypoint would
    # otherwise misfire.
    #
    # Uses its own threshold (not stuck_threshold_rad_per_tick) so the wedge
    # gate and the stall trigger can be tuned independently — the wedge gate
    # cares about "not moving under command", stall cares about "policy
    # produces the same joint config forever".
    #
    # Disabled by default (window=0). Recommended starting values when
    # enabling: window=60 (≈2 s at 30 Hz — long enough to let Ruckig's
    # goal-tail deceleration and momentary pauses pass without misfiring),
    # threshold=0.005 rad/tick (same magnitude as the wedge gate default).
    joint_stall_window_steps: int = 0
    joint_stall_threshold_rad: float = 0.005
