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

from lerobot.cameras import make_cameras_from_configs

from ..splatsim_lerobot.base_robot import BaseSplatSimRobot, resize_image
from .config_ur5_small_engine import UR5SmallEngineConfig

logger = logging.getLogger(__name__)


class UR5SmallEngine(BaseSplatSimRobot):
    """Real UR5 (small-engine table mount) with physical lerobot cameras.

    Shares the gello ZMQ joint transport, observation/action formatting, and
    action sending with the simulation robot (see ``BaseSplatSimRobot``). The
    only difference is the camera source: images come from physical lerobot
    cameras (e.g. a RealSense D455 over USB) instead of being rendered.
    """

    config_class = UR5SmallEngineConfig
    name = "ur5_small_engine"

    def __init__(self, config: UR5SmallEngineConfig):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  Camera hooks                                                      #
    # ------------------------------------------------------------------ #
    def _make_cameras(self) -> dict[str, Any]:
        # Build real camera drivers from config.cameras (opencv / realsense / ...).
        return make_cameras_from_configs(self.config.cameras)

    def _connect_cameras(self) -> None:
        for name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"{self}: connected camera '{name}' ({cam})")

    def _disconnect_cameras(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

    def _add_camera_observations(self, obs: dict[str, Any], lerobot_obs: dict[str, Any]) -> None:
        """Read each physical camera and format it to match the sim robot's output.

        IMPORTANT: the produced arrays must match the preprocessing used when the
        policy's dataset was recorded (the simulation robot's resize path), so the
        real images land in the same distribution the policy expects:
          - color order RGB (lerobot RealSense defaults to ColorMode.RGB)
          - float32 in [0, 1]
          - resized to (image_height, image_width) with the configured mode
          - HWC layout

        Each camera key K crossed with each resize mode M -> ``"{K}_{M}"``.
        """
        for cam_key, cam in self.cameras.items():
            # HWC uint8 RGB at the camera's native resolution.
            frame = cam.read_latest()

            # -> CHW float32 in [0, 1] so resize_image (letterbox/stretch) matches sim.
            img_chw = np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))

            for image_resize_mode in self.config.image_resize_modes:
                resized = resize_image(
                    img_chw,
                    output_size=(self.config.image_height, self.config.image_width),
                    mode=image_resize_mode,
                )
                # CHW -> HWC to match the sim robot's emitted images.
                lerobot_obs[f"{cam_key}_{image_resize_mode}"] = np.transpose(resized, (1, 2, 0))

        # Sanity check: warn if an expected observation camera has no source.
        for camera_name in self.config.camera_names:
            if not any(k.startswith(camera_name) for k in lerobot_obs):
                logger.warning(
                    f"No camera producing '{camera_name}' (configured cameras: "
                    f"{list(self.cameras.keys())}). Policy inputs will be incomplete."
                )
