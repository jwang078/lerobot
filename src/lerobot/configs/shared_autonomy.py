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
