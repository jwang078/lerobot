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

"""Shared base for the SplatSim-style lerobot robots.

Both the simulation robot (``SplatSimLerobot``) and the real UR5 small-engine
robot (``UR5SmallEngine``) talk to the same gello ZMQ robot server for joint
state/action and use the exact same observation/action key formatting. The
*only* thing that differs is where camera images come from:

  * simulation -> images are rendered by the SplatSim server and arrive embedded
    in the gello ``RobotEnv`` observation (``base_rgb`` / ``wrist_rgb``).
  * real robot -> images come from physical lerobot cameras (e.g. a RealSense
    D455 over USB) configured via ``config.cameras``.

This base owns everything shared. Subclasses implement three small hooks:

  * ``_make_cameras()``            -> the camera-driver dict (default: none)
  * ``_connect_cameras()``         -> open the physical cameras (default: no-op)
  * ``_add_camera_observations()`` -> populate image keys in the observation
  * ``_disconnect_cameras()``      -> close the physical cameras (default: no-op)
"""

import logging
from functools import cached_property
from typing import Any

import cv2
import numpy as np
import torch
from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot
from splatsim.utils.image_utils import letterbox

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_splatsim_lerobot import BaseSplatSimRobotConfig

logger = logging.getLogger(__name__)


def resize_image(img: np.ndarray, output_size: tuple[int, int], mode: str = "letterbox") -> np.ndarray:
    """Resize image to output_size using the specified mode.

    Args:
        img: Input image in CHW format (channels, height, width), float32 in [0, 1]
        output_size: Target (height, width)
        mode: Resize mode - "letterbox" or "stretch"

    Returns:
        Resized image in CHW format, float32 in [0, 1]
    """
    if mode == "letterbox":
        return letterbox(img, output_size=output_size)
    elif mode == "stretch":
        # Convert from CHW to HWC for cv2
        img_hwc = np.transpose(img, (1, 2, 0))
        # Resize using cv2 (stretches to fill, ignoring aspect ratio)
        img_resized = cv2.resize(img_hwc, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
        # Convert back to CHW
        return np.transpose(img_resized, (2, 0, 1))
    else:
        raise ValueError(f"Unknown image resize mode: {mode}. Use 'letterbox' or 'stretch'.")


class BaseSplatSimRobot(Robot):
    """Shared gello-ZMQ-backed lerobot robot. Subclass and supply camera hooks."""

    config_class = BaseSplatSimRobotConfig

    def __init__(self, config: BaseSplatSimRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_client = None
        self.env = None  # RobotEnv instance
        self._is_connected = False
        # Subclasses provide the camera-driver dict. lerobot-record also reads
        # ``self.cameras`` to count capture threads, so it must always exist.
        self.cameras = self._make_cameras()

    # ------------------------------------------------------------------ #
    #  Camera hooks (overridden by subclasses)                           #
    # ------------------------------------------------------------------ #
    def _make_cameras(self) -> dict[str, Any]:
        """Return the camera-driver dict. Default: no cameras (sim renders them)."""
        return {}

    def _connect_cameras(self) -> None:
        """Open physical cameras. Default: nothing to do."""
        pass

    def _add_camera_observations(self, obs: dict[str, Any], lerobot_obs: dict[str, Any]) -> None:
        """Populate image keys in ``lerobot_obs`` from the raw ``obs`` / cameras."""
        raise NotImplementedError

    def _disconnect_cameras(self) -> None:
        """Close physical cameras. Default: nothing to do."""
        pass

    # ------------------------------------------------------------------ #
    #  Shared feature definitions                                        #
    # ------------------------------------------------------------------ #
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation space"""
        features = {}

        # Add each configured camera (with resize mode suffix to match get_observation keys)
        for camera_name in self.config.camera_names:
            for image_resize_mode in self.config.image_resize_modes:
                features[f"{camera_name}_{image_resize_mode}"] = (
                    self.config.image_height,
                    self.config.image_width,
                    3,
                )

        # Add individual state features for each joint (LeRobot expects named joints)
        for joint_name in self.config.joint_names:
            features[f"state.{joint_name}"] = float

        return features

    @cached_property
    def action_features(self) -> dict:
        """Define action space"""
        features = {}

        # Add individual action features for each joint (LeRobot expects named joints)
        for joint_name in self.config.joint_names:
            features[f"action.{joint_name}"] = float

        return features

    # ------------------------------------------------------------------ #
    #  Connection lifecycle (shared)                                     #
    # ------------------------------------------------------------------ #
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the gello ZMQ robot server and open cameras."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Joint state/action transport is identical for sim and real: the gello
        # ZMQ robot server (launch_nodes.py) serves either a simulated or a real
        # UR5 behind the same interface.
        self.robot_client = ZMQClientRobot(port=self.config.port, host=self.config.hostname)

        # RobotEnv merges robot joint obs with any gello camera_dict. For the
        # real robot we use lerobot's own camera drivers instead, so this stays
        # empty and images are added in _add_camera_observations().
        self.env = RobotEnv(
            robot=self.robot_client,
            control_rate_hz=self.config.control_rate_hz,
            camera_dict={},
        )

        self._connect_cameras()

        self._is_connected = True
        logger.info(f"{self} connected to {self.config.hostname}:{self.config.port}")

    @property
    def is_calibrated(self) -> bool:
        """These robots don't require calibration"""
        return True

    def calibrate(self) -> None:
        """These robots don't require calibration"""
        pass

    def configure(self) -> None:
        """Apply any runtime configuration"""
        pass

    # ------------------------------------------------------------------ #
    #  Observation / action (shared, with camera hook)                   #
    # ------------------------------------------------------------------ #
    def get_observation(self) -> dict[str, Any]:
        """Get current observation: camera images (via hook) + joint state."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs = self.env.get_obs()

        lerobot_obs: dict[str, Any] = {}

        # Image keys are subclass-specific (rendered vs real cameras).
        self._add_camera_observations(obs, lerobot_obs)

        # Joint state formatting is identical for sim and real.
        self._add_joint_observations(obs, lerobot_obs)

        return lerobot_obs

    def _add_joint_observations(self, obs: dict[str, Any], lerobot_obs: dict[str, Any]) -> None:
        """Append per-joint ``state.<joint>`` floats. Shared by sim and real."""
        joint_positions = obs.get("joint_positions")
        if joint_positions is None:
            return

        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.detach().cpu().numpy()
        if not isinstance(joint_positions, np.ndarray):
            joint_positions = np.array(joint_positions)

        # Append gripper state if needed (7 DOF vs 6 DOF)
        if joint_positions.shape[0] == 6:
            gripper_pos = obs.get("gripper_position", 0.0)
            if isinstance(gripper_pos, np.ndarray):
                gripper_pos = gripper_pos[0] if len(gripper_pos) > 0 else 0.0
            elif isinstance(gripper_pos, torch.Tensor):
                gripper_pos = gripper_pos.item()
            joint_positions = np.append(joint_positions, gripper_pos)

        # Add individual joint states (LeRobot expects named joints, not a single array)
        for i, joint_name in enumerate(self.config.joint_names):
            if i < len(joint_positions):
                lerobot_obs[f"state.{joint_name}"] = float(joint_positions[i])

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to the robot via RobotEnv.step(). Shared by sim and real."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract action values
        if isinstance(action, dict):
            # Try different possible keys
            action_array = action.get("action", action.get("joint_positions"))
            if action_array is None:
                # Try to reconstruct from individual joint actions
                # First try with 'action.' prefix
                action_list = [action.get(f"action.{name}") for name in self.config.joint_names]
                if all(a is not None for a in action_list):
                    action_array = np.array(action_list)
                else:
                    # Try without prefix (policy output format)
                    action_list = [action.get(name) for name in self.config.joint_names]
                    if all(a is not None for a in action_list):
                        action_array = np.array(action_list)
                    else:
                        raise ValueError(f"Could not extract action from dict: {action.keys()}")
        else:
            action_array = action

        if isinstance(action_array, torch.Tensor):
            action_array = action_array.detach().cpu().numpy()

        # Ensure it's a 1D array
        if action_array.ndim > 1:
            action_array = action_array.squeeze()

        action_to_send = action_array

        # Send to robot via RobotEnv.step() - this returns the next observation
        # but we don't use it here since get_observation() will be called separately
        self.env.step(action_to_send)

        # Return the action that was sent (in the original format)
        if isinstance(action, dict):
            return action
        else:
            return {"action": action_array}

    def disconnect(self):
        """Disconnect from robot and close cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._disconnect_cameras()

        self._is_connected = False
        self.robot_client = None
        self.env = None
        logger.info(f"{self} disconnected.")
