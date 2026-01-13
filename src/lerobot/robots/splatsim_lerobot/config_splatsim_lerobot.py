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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("splatsim_lerobot")
@dataclass
class SplatSimLerobotConfig(RobotConfig):
    """Configuration for SplatSim LeRobot simulation robot"""

    # ZMQ connection settings
    robot_port: int = 6001
    hostname: str = "127.0.0.1"

    # Camera ports
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001

    # Control rate for RobotEnv
    control_rate_hz: int = 50

    # Image size for observations
    image_width: int = 224
    image_height: int = 224

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Whether to include wrist camera
    use_wrist_camera: bool = False

    # Joint names (can be configured to match your dataset)
    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ]
    )
