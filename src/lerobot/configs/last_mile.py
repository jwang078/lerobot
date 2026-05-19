#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Last-mile help wrapper config.

Wraps a chunk-predicting policy with an eval-time "help" mechanism that
intervenes when the inner policy is unable to reach the goal. Two pluggable
halves:

* Detect — when does the inner policy need help?
    - ``oracle_ee_distance``: EE position within threshold of goal (oracle).
    - ``stall``: joint angles haven't moved meaningfully in last N steps.
* Help — what to do about it?
    - ``blend_to_goal_bias``: blend commanded joints toward ``q_goal_bias``.
    - ``rrt_to_goal``: delegate to an outer SharedAutonomyPolicyWrapper's
      ``trigger_rrt_to_goal()`` and play the resulting plan.
    - ``swap_to_alt_policy``: unload the inner policy from GPU, load an
      alt policy from disk, run it to episode end, then swap back.

Backends are selected via ``detect_backend`` / ``help_backend`` strings.
Each backend has its own params dataclass; only the chosen backend's
fields are read by the factory.
"""

from dataclasses import dataclass, field

# Backend selectors. Typed as plain ``str`` (not ``Literal``) because draccus
# doesn't have a decoding function for typing.Literal. Validation happens in
# ``build_detector`` / ``build_helper`` which raise ValueError on unknown values.
# Allowed values:
#   detect_backend: "oracle_ee_distance" | "stall" | "no_ee_progress"
#   help_backend:   "blend_to_goal_bias" | "rrt_to_goal" | "swap_to_alt_policy"
DetectBackend = str
HelpBackend = str


@dataclass
class OracleEEDistanceParams:
    # EE position distance (meters) below which the override fires. Default
    # matches a few × the typical success tolerance (e.g. 3 cm tol → 0.05 m
    # threshold) so the override fires shortly before the success region.
    ee_distance_threshold: float = 0.05


@dataclass
class StallParams:
    # Rolling window over which "no progress" is measured.
    window: int = 75
    # Max pairwise joint L2 within the window below which the robot is
    # considered stalled. 0.02 rad ≈ 1.15°.
    joint_l2_threshold: float = 0.02
    # Don't fire stall in the first N steps of an episode (initial
    # acceleration phase legitimately has near-zero joint motion when
    # warming up from rest).
    min_warmup_steps: int = 50


@dataclass
class NoEEProgressParams:
    """Detector for 'policy is drifting / not making EE progress toward goal'.

    Unlike StallDetector (which measures joint range), this measures
    whether the policy is making progress toward the goal RELATIVE TO A
    LOCAL ANCHOR. The anchor is the EE distance at the end of the most
    recent progress phase, NOT the global best. After enough consecutive
    steps above the anchor, the anchor resets to the local minimum from
    the repositioning phase — so the robot can "back off to reposition,
    then make progress from a new starting point" without falsely firing.

    Requires oracle EE positions.
    """

    # Number of steps without ``min_decrease_m`` of EE progress (below anchor)
    # before firing.
    no_progress_window: int = 75
    # How much the EE must drop below the current anchor to count as progress.
    # 1mm filters out tiny oscillations.
    min_decrease_m: float = 0.001
    # Don't fire in the first N steps of the episode.
    min_warmup_steps: int = 50
    # After this many CONSECUTIVE steps ABOVE the current anchor AND the
    # EE has come back down from its away-phase peak by ``reposition_turnaround_m``,
    # declare a repositioning phase and reset the anchor to the current EE.
    # Choose < ``no_progress_window`` so repositioning resets the anchor BEFORE
    # the no-progress counter fires. Choose > 0 (typical 20-40) so tiny
    # one-step blips don't reset the anchor.
    reposition_grace_steps: int = 30
    # Turnaround signal. The EE must drop at least this far below its peak
    # during the away-phase before the anchor-reset will fire. This is what
    # distinguishes real repositioning ("went up, then came back") from
    # monotonic drift ("just kept going up"). Drift never satisfies this and
    # eventually trips the no-progress fire instead.
    reposition_turnaround_m: float = 0.01


@dataclass
class BlendToGoalBiasParams:
    # 0.0 = no override (sanity check), 1.0 = command exactly q_goal_bias on
    # the joints whenever in-range. Intermediate values blend.
    blend_alpha: float = 1.0
    # Refuse the override if it would command a joint-space step larger than
    # this L2 norm — protects pybullet's integrator from kinematic-redundancy
    # teleports that crash the simulator with a C-level SIGABRT.
    max_safe_joint_jump: float = 0.5


@dataclass
class RRTToGoalParams:
    # Whether to disable SA's teleop-recording bookkeeping + teleport-to-
    # q_start optimization when this helper attaches to SA. Defaults True
    # (the eval case — recording is off, the misleading "recorded
    # intervention" warnings should not fire).
    #
    # Set this to False when running ``lerobot-eval --intervention.method=rrt``
    # with last-mile config enabled, so the SA wrapper's recording machinery
    # stays active and the RRT interventions actually get saved to the dataset.
    disable_sa_recording: bool = True


@dataclass
class SwapToAltPolicyParams:
    # Path to a pretrained checkpoint dir for the goal-reaching alt policy.
    # Required when help_backend == "swap_to_alt_policy".
    alt_policy_path: str | None = None
    # Allowed values:
    #   "cpu_cache":   snapshot inner's state_dict to host RAM before unload;
    #                  reload on episode reset from the snapshot.
    #   "disk_reload": re-run from_pretrained on the original inner checkpoint;
    #                  slower but doesn't hold ~7GB host RAM for PI05.
    inner_unload_strategy: str = "cpu_cache"


@dataclass
class LastMileConfig:
    """Configuration for the last-mile help wrapper."""

    enabled: bool = False
    detect_backend: DetectBackend = "oracle_ee_distance"
    help_backend: HelpBackend = "blend_to_goal_bias"
    oracle_ee_distance_params: OracleEEDistanceParams = field(default_factory=OracleEEDistanceParams)
    stall_params: StallParams = field(default_factory=StallParams)
    no_ee_progress_params: NoEEProgressParams = field(default_factory=NoEEProgressParams)
    blend_to_goal_bias_params: BlendToGoalBiasParams = field(default_factory=BlendToGoalBiasParams)
    rrt_to_goal_params: RRTToGoalParams = field(default_factory=RRTToGoalParams)
    swap_to_alt_policy_params: SwapToAltPolicyParams = field(default_factory=SwapToAltPolicyParams)
