#!/usr/bin/env python

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

"""
Shared Autonomy configuration for diffusion and flow-matching policies.

Based on "To the Noise and Back: Diffusion for Shared Autonomy"
Reference: https://arxiv.org/abs/2302.12244

Works with any noise/flow-based policy by controlling the starting point
and schedule of the denoising process. The forward_flow_ratio parameter
controls the trade-off between fidelity (preserving human intent) and
conformity (following learned behavior distribution):

- forward_flow_ratio = 0.0: No intervention, return human action directly
- forward_flow_ratio = 0.4: Moderate blending (recommended default)
- forward_flow_ratio = 1.0: Full model control, ignore human input
"""

from dataclasses import dataclass, field


@dataclass
class PreJumpLookbackConfig:
    """Tunables for the `rrt_collision_detection="pre_jump_lookback"` mode.

    Controls how far back (in env steps) the wrapper rewinds the recorded
    joint-state history before triggering RRT. The robot is teleported to
    the rewound config; RRT then plans from there. This is the historical
    behavior — useful when the policy ALREADY entered collision and you
    want the recorded recovery to start from a known-safe earlier state.
    """

    # Minimum lookback (in env steps) for the pre-RRT teleport. When
    # `steps_max` is None, this is the FIXED lookback used by every RRT
    # trigger. When `steps_max` is set, this is the LOWER bound of the
    # per-trigger random sampling interval.
    steps_min: int = 5
    # Optional upper bound. When set (must be >= steps_min), each RRT
    # trigger samples a random lookback from the closed [min, max]
    # interval, broadening the distribution of states from which the
    # recorded RRT corrections start.
    # The wrapper sizes its actual_q_history deque to fit `steps_max`
    # (or stays at `steps_min` when None), so the buffer always has
    # enough history for whatever lookback is sampled. Default None =
    # fixed-lookback at `steps_min`.
    steps_max: int | None = None


@dataclass
class FutureChunkConfig:
    """Tunables for the `rrt_collision_detection="future_chunk"` mode.

    In this mode the wrapper FK-checks the inner policy's already-cached
    action chunk against the obstacle world EVERY select_action tick. If
    any future waypoint would collide, RRT is triggered preemptively
    from the robot's CURRENT continuous-motion joint state — no rewind,
    no teleport. The recorded intervention episode therefore starts at
    velocity-continuous, in-distribution states.
    """

    # Optional: shorten how many chunk frames we FK-check. None = full
    # chunk (n_remaining_steps from inner_policy.get_pending_action_chunk()).
    # Set to e.g. 8 to cap chunk-check cost at ~8 FK queries per tick.
    horizon_frames: int | None = None
    # Optional: explicit clearance for FK shielding. None = inherit from
    # the top-level `rrt_obstacle_clearance` / `rrt_self_collision_clearance`
    # fields. Override only when you want the shield to be more (or less)
    # conservative than RRT's planner-time margins.
    obstacle_clearance: float | None = None
    self_collision_clearance: float | None = None


@dataclass
class SharedAutonomyConfig:
    """Configuration for Shared Autonomy inference.

    The key parameter is forward_flow_ratio (t_sw), which controls the
    trade-off between fidelity and conformity:

    For flow matching (PI0.5): x_tsw = t_sw * noise + (1-t_sw) * policy_guidance_action
    For diffusion (DDPM/DDIM): x_tsw = sqrt(alpha_bar_t) * policy_guidance_action + sqrt(1-alpha_bar_t) * noise
    """

    enabled: bool = False
    forward_flow_ratio: float = 0.4  # t_sw switching time (0.0-1.0)
    policy_guidance_action_buffer_size: int = 1
    apply_to_first_action_only: bool = True
    show_slider: bool = True  # launch a Tkinter slider to adjust forward_flow_ratio live
    start_paused: bool = False  # start with policy paused (unpause via GUI button)
    robot_name: str = "robot_iphone_w_engine_new"
    max_joint_delta: float = 0.016
    num_dofs: int = 6
    blend_mode: str = "every_step"  # "every_step" or "once_per_chunk"
    # Number of action steps at the start of each chunk to anchor exactly to guidance via inpainting.
    # 0 = current behavior (full-chunk blending only). k > 0 = clamp first k steps to guidance
    # inside the denoising loop, letting the model generate a coherent continuation from those steps.
    # Only applies to GuidanceBlendStrategy.DENOISE.
    n_anchor_steps: int = 0
    debug: bool = False
    debug_maxlen: int = 100
    # Control rate (Hz) used by the RRT-to-Goal mode for ruckig time parametrization.
    # Should match the env's fps. Only consulted when the GUI's "RRT to Goal" button
    # is pressed, so a slightly off value just changes the trajectory pacing.
    fps: int = 30
    # Selects which collision-detection mode drives RRT triggering.
    # Options:
    #   "pre_jump_lookback" (default) — historical reactive flow. No
    #     preemptive shielding. ALL triggers (in_collision, time stall,
    #     no_progress, no_progress_ori) sample a lookback from
    #     `pre_jump_lookback.steps_min/max` and teleport the robot to
    #     that earlier joint config before RRT plans. See
    #     `PreJumpLookbackConfig` docstring.
    #
    #   "future_chunk" — predictive shielding (no rewind). Every
    #     select_action call, the wrapper FK-checks the inner policy's
    #     already-cached action chunk against the obstacle world. If a
    #     future waypoint would collide, RRT is triggered NOW from the
    #     robot's CURRENT joint state. ALL triggers (shield-fired,
    #     in_collision, time stall, no_progress) use the no-lookback
    #     flow. See `FutureChunkConfig` docstring.
    #
    #   "hybrid" — best of both. Shield runs (preemptive FK check), AND
    #     collision-related triggers use no-lookback (rewind is pointless
    #     when the robot is already in collision or about to be), AND
    #     stall/no-progress triggers use lookback (rewinding to before
    #     the stall point gives RRT a fresh start, instead of planning
    #     from the dead-stop pose). Per-trigger dispatch is hard-coded
    #     in the InterventionController; the wrapper's shield is always
    #     no-lookback by nature (it's a collision-related trigger).
    #     Both `pre_jump_lookback` and `future_chunk` nested configs are
    #     consulted in this mode.
    rrt_collision_detection: str = "pre_jump_lookback"
    # Tunables for `rrt_collision_detection="pre_jump_lookback"` mode.
    # Ignored when mode is `"future_chunk"`.
    pre_jump_lookback: PreJumpLookbackConfig = field(default_factory=PreJumpLookbackConfig)
    # Tunables for `rrt_collision_detection="future_chunk"` mode.
    # Ignored when mode is `"pre_jump_lookback"`.
    future_chunk: FutureChunkConfig = field(default_factory=FutureChunkConfig)
    # When True, the wrapper teleports the env's robot to the pre-jump pose
    # before starting RRT execution (sim-only). Set to False for real-robot
    # runs where teleportation isn't possible.
    rrt_teleport_to_q_start: bool = True
    # When True (default), trigger_rrt_to_goal blocks until planning +
    # teleport finish and the wrapper is in EXECUTING. The env is never
    # stepped while the planner is working, so the recorded intervention
    # data begins on the very first RRT action — no frames of "policy still
    # driving the robot toward the collision while the planner thinks".
    # Set False only when you're driving the wrapper from a GUI thread that
    # can't afford to block (e.g. an interactive teleop control surface).
    rrt_blocking_plan: bool = True
    # How the planner picks among IK-goal-candidate paths. One of:
    #   * "ee_arc_length" (default) — minimize cartesian EE distance traversed.
    #     Penalizes wide swings; current behavior.
    #   * "joint_arc_length" — minimize joint-space L2 distance summed across
    #     waypoints. Legacy behavior; tends to pick paths that land near
    #     q_start in configuration space even if the EE swings wide.
    #   * "joint_velocity_match" — minimize L2 deviation between the
    #     candidate's initial joint velocity and the robot's recent joint
    #     velocity (averaged over the trailing samples of
    #     `_actual_q_history`). Picks the path that maintains the robot's
    #     current motion direction the most, minimizing the velocity
    #     discontinuity at the trigger moment. Requires enough history to
    #     derive a velocity (≥2 samples); raises if not.
    # None passes through to the planner's default (EE_ARC_LENGTH).
    rrt_path_selection: str | None = None
    # IK-goal-selection strategy: scores AMONG the IK candidates BEFORE
    # running RRT, based on goal-state geometry alone (no planned path
    # needed). When set, the planner sorts candidates by this score,
    # tries RRT in order, and takes the FIRST successful plan —
    # `rrt_path_selection` is unused in this mode because each candidate's
    # path goes to a different goal, so cross-path comparison is
    # meaningless once IK selection has committed to a goal.
    # Options:
    #   "joint_distance" — minimize ||q_candidate - q_start||. For
    #     redundant arms (7-DOF) this picks the IK branch that requires
    #     the LEAST joint reconfiguration. Use when training data is
    #     multimodal (multiple IK branches) and you want interventions
    #     to consistently commit to the policy's current mode.
    # None (default) → no IK pre-selection; multi-candidate path scoring
    # via `rrt_path_selection` runs as before.
    rrt_ik_goal_selection: str | None = None
    # Per-IK multi-path scoring knobs (ports SplatSim's
    # TrajectoryGenerator._generate_multiple_path_candidates pattern):
    #
    #   rrt_num_path_candidates_per_ik   (default 1) — when > 1, the
    #     planner generates N distinct RRT paths to each IK goal by
    #     perturbing q_start/q_goal between attempts, then uses
    #     `rrt_path_selection` to pick the best one for that IK. This is
    #     what makes path_selection non-trivial when ik_goal_selection
    #     is set: each IK gets several path candidates, the best one
    #     wins for that IK, and IK ordering decides which IK's best
    #     path is used. With the default 1, behavior is unchanged
    #     (single path per IK, scored only for cross-IK comparison).
    #
    #   rrt_max_path_attempts_per_ik    (default 5) — caps consecutive
    #     RRT calls between successes. Total RRT calls per IK is bounded
    #     by num_candidates × max_attempts in the worst case.
    #
    #   rrt_path_perturbation_scale     (default 0.001 rad) — magnitude
    #     of the random perturbation applied to q_start/q_goal in the
    #     2nd+ attempts. Matches SplatSim's RRT_PERTURBATION_SCALE.
    rrt_num_path_candidates_per_ik: int = 1
    rrt_max_path_attempts_per_ik: int = 5
    rrt_path_perturbation_scale: float = 0.001
    # Number of IK seed samples per EE-pose goal. The planner runs
    # pybullet's nullspace IK with N different random seeds, filters
    # collision-failing solutions, and uses the survivors as RRT goal
    # candidates. Higher = better chance of finding a collision-free goal
    # in cluttered scenes, at the cost of N IK solves per planning call.
    # 16 is the historical default (matches the previously-hardcoded value
    # in `rrt_source.py`); bump to 24-32 for scenarios with heavy
    # obstacle congestion (e.g. small_engine with multiple boxes near the
    # goal EE pose) where 16 often leaves zero collision-free candidates.
    rrt_num_ik_candidates: int = 16
    # RRT collision clearances, in meters. These are PLANNER-time margins;
    # they don't affect eval-time termination (which uses obstacle_clearance=0
    # at the env level — actual penetration only). Increase them to make RRT
    # plans more conservative, giving the policy room to drift along the path
    # without triggering eval-time collisions.
    #
    # SplatSim's default (`_COLLISION_CLEARANCE = 0.01`, `self_collision_clearance=0.0`)
    # leaves RRT plans 1 cm from obstacles and 0 cm from self-intersection —
    # the policy needs near-perfect path-following to avoid eval termination.
    # Empirically, scenarios 10 and 25 fail at eval with SELF-COLLISION even
    # though intervention recording produces hundreds of steps of clean
    # recovery, suggesting policy drift is shaving the 0 cm self-margin to
    # actual intersection.
    #
    # Trade-off: tight scenarios may become unplannable at high clearance —
    # RRT will return None paths more often, surfacing as `plan_failures`
    # in the per-scenario CSV and `rrt_steps_executed = 0` rows. Start small
    # (0.02-0.04 m obstacle, 0.01-0.02 m self) and only crank up if planning
    # success stays high.
    #
    # None = use SplatSim's default constants. A float overrides at the
    # RRTToGoalPlanner level and is threaded into every
    # `check_links_in_collision` / `get_path` call.
    rrt_obstacle_clearance: float | None = None
    rrt_self_collision_clearance: float | None = None
    # Non-adjacent robot-link pairs to EXCLUDE from self-collision checks.
    # Use for URDF link pairs that are structurally close at every valid
    # joint config (e.g. UR's base_link(0) vs upper_arm_link(2), ~4 mm apart
    # due to shoulder bracket geometry) — without skipping them, any
    # non-zero `rrt_self_collision_clearance` flags every valid IK solution
    # as a collision, breaking RRT planning entirely.
    #
    # Format: list of [link_a, link_b] pairs (draccus-friendly nested list,
    # not tuples). Order doesn't matter ((a,b) == (b,a)). None = no skips.
    # Robot-specific — set on the SA config that gets dispatched into the
    # SplatSim env. For the UR robot in the small-engine scene, the
    # canonical setting is `[[0, 2]]`.
    rrt_self_collision_skip_pairs: list[list[int]] | None = None
    # If True (default), ruckig time-parametrization splits the smoothed RRT
    # path at sharp-angle waypoints (angle > 45°) and runs ruckig per-segment
    # with zero velocity at every sharp boundary — historical "stop-and-go"
    # behavior. If False, ruckig is invoked once across the full path with
    # intermediate_positions, optimizing corner deceleration internally
    # without forced zero-velocity stops. Empirical comparison on
    # lever-grasp interventions (d5_fast_03dag vs d5jvm_g0_03dag,
    # 2026-06-10) showed no observable trajectory-duration difference
    # between the two modes — typical manipulation RRT plans don't have
    # enough sharp corners for the segmentation to matter. True is the
    # conservative default; flip to False only for stress-testing paths
    # with many sharp corners or matching legacy recordings.
    rrt_segment_at_sharp_corners: bool = True
    # Diagnostic dump for MIN_PAIR_CLEARANCE path scoring. Controls whether
    # `_path_min_pair_clearance` prints a table of all non-adjacent link
    # pairs (sorted by min distance, with min/max/range columns + a
    # STRUCTURAL? flag for pairs with sub-5mm range) to help identify
    # URDF-rigid pairs that should be added to `rrt_self_collision_skip_pairs`.
    # Modes:
    #   "off"    — no diagnostic, no overhead. Production default.
    #              Uses the lean 10cm getClosestPoints cap so the score
    #              loop runs in ~200 ms per call.
    #   "first"  — dump the table once per planner instance (first RRT
    #              trigger only). Subsequent calls revert to the lean
    #              10cm cap. Useful for catching structural offenders
    #              once and then staying out of the way.
    #   "always" — dump every RRT trigger with 5m cap + full per-pair
    #              tracking. ~2-5x slower (~500-1000ms per trigger) and
    #              ~100 log lines per trigger. Use when actively
    #              iterating on the skip list across many paths.
    # Has no effect when `rrt_path_selection != "min_pair_clearance"`.
    rrt_diagnostic_log_pairs: str = "off"

    def __post_init__(self) -> None:
        if not 0.0 <= self.forward_flow_ratio <= 1.0:
            raise ValueError(f"forward_flow_ratio must be in [0, 1], got {self.forward_flow_ratio}")
        if self.policy_guidance_action_buffer_size <= 0:
            raise ValueError(
                f"policy_guidance_action_buffer_size must be positive, got {self.policy_guidance_action_buffer_size}"
            )
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")
        valid_blend_modes = {"every_step", "once_per_chunk"}
        if self.blend_mode not in valid_blend_modes:
            raise ValueError(f"blend_mode must be one of {valid_blend_modes}, got '{self.blend_mode}'")
        valid_diag_modes = {"off", "first", "always"}
        if self.rrt_diagnostic_log_pairs not in valid_diag_modes:
            raise ValueError(
                f"rrt_diagnostic_log_pairs must be one of {valid_diag_modes}, "
                f"got '{self.rrt_diagnostic_log_pairs}'"
            )
        valid_collision_modes = {"pre_jump_lookback", "future_chunk", "hybrid"}
        if self.rrt_collision_detection not in valid_collision_modes:
            raise ValueError(
                f"rrt_collision_detection must be one of {valid_collision_modes}, "
                f"got '{self.rrt_collision_detection}'"
            )
        # Validate both nested configs when relevant. Hybrid consults BOTH,
        # so validate both. The other two modes only need one, but checking
        # both is harmless and keeps the rules consistent.
        needs_lookback = self.rrt_collision_detection in ("pre_jump_lookback", "hybrid")
        needs_future_chunk = self.rrt_collision_detection in ("future_chunk", "hybrid")
        if needs_lookback:
            lb = self.pre_jump_lookback
            if lb.steps_min < 0:
                raise ValueError(f"pre_jump_lookback.steps_min must be >= 0, got {lb.steps_min}")
            if lb.steps_max is not None and lb.steps_max < lb.steps_min:
                raise ValueError(
                    f"pre_jump_lookback.steps_max ({lb.steps_max}) must be >= steps_min ({lb.steps_min})"
                )
        if needs_future_chunk:
            fc = self.future_chunk
            if fc.horizon_frames is not None and fc.horizon_frames <= 0:
                raise ValueError(
                    f"future_chunk.horizon_frames must be positive when set, got {fc.horizon_frames}"
                )
