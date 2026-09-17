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

import logging
from typing import Any

import numpy as np
import torch

from .base_robot import BaseSplatSimRobot, resize_image
from .config_splatsim_lerobot import SplatSimLerobotConfig

logger = logging.getLogger(__name__)

# Re-exported for backward compatibility (callers used to import these here).
__all__ = ["SplatSimLerobot", "resize_image"]


class SplatSimLerobot(BaseSplatSimRobot):
    """SplatSim *simulation* robot.

    Joint state/action go over gello ZMQ (handled by the base). Camera images are
    rendered by the SplatSim server and arrive embedded in the RobotEnv
    observation under the configured ``camera_names`` (e.g. ``base_rgb``).
    """

    config_class = SplatSimLerobotConfig
    name = "splatsim_lerobot"

    def __init__(self, config: SplatSimLerobotConfig):
        super().__init__(config)

    def _add_camera_observations(self, obs: dict[str, Any], lerobot_obs: dict[str, Any]) -> None:
        """Pull the server-rendered images out of the RobotEnv observation."""
        for camera_name in self.config.camera_names:
            for image_resize_mode in self.config.image_resize_modes:
                key = f"{camera_name}_{image_resize_mode}"
                if key in obs:
                    # Already resized server-side.
                    img = obs[key]
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                    # (C, H, W) -> (H, W, C)
                    if img.ndim == 3 and img.shape[0] <= 4:
                        img = img.transpose(1, 2, 0)
                    lerobot_obs[key] = img
                else:
                    img = obs.get(camera_name)
                    if img is not None:
                        # Convert to numpy if needed
                        if isinstance(img, torch.Tensor):
                            img = img.detach().cpu().numpy()

                        # Resize to configured size using configured mode
                        img_resized = resize_image(
                            img,
                            output_size=(self.config.image_height, self.config.image_width),
                            mode=image_resize_mode,
                        )
                        # Change from (C, H, W) to (H, W, C)
                        lerobot_obs[key] = img_resized.transpose(1, 2, 0)
            if not any(k for k in lerobot_obs if k.startswith(camera_name)):
                logger.warning(f"No {camera_name} in observation!")
