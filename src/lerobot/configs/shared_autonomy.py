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

from dataclasses import dataclass


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
    # Number of past steps to keep in a ring buffer of actual joint observations.
    # When RRT is triggered, the wrapper plans from the oldest entry in this
    # buffer (== ~N steps before trigger), then teleports the sim to that pose
    # so the recorded intervention trajectory begins at a clean, pre-jump state
    # with no "sim catching up" frames. Larger N rewinds further; too large and
    # the rewound pose may pre-date a relevant scene change (e.g. an object
    # the policy already pushed). 3-5 is a good default.
    rrt_pre_jump_lookback_steps: int = 5
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
