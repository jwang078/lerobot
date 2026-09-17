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

from dataclasses import dataclass

from ..config import RobotConfig
from ..splatsim_lerobot.config_splatsim_lerobot import BaseSplatSimRobotConfig


@RobotConfig.register_subclass("ur5_small_engine")
@dataclass
class UR5SmallEngineConfig(BaseSplatSimRobotConfig):
    """Configuration for the *real* UR5 small-engine robot.

    "Small engine" refers to the table-mount fixture the UR5 is set up with. The
    joint state/action transport is identical to the simulation robot (gello ZMQ
    via ``launch_nodes.py``); the difference is that camera images come from
    physical lerobot cameras configured in ``cameras`` instead of being rendered.

    To use a RealSense D455 over USB, configure ``cameras`` keyed by the
    *observation* camera name so the produced keys match what the policy was
    trained on, e.g.::

        from lerobot.cameras.realsense import RealSenseCameraConfig

        UR5SmallEngineConfig(
            port=6001,
            camera_names=["base_rgb", "wrist_rgb"],
            image_resize_modes=["letterbox"],
            cameras={
                "base_rgb": RealSenseCameraConfig(
                    serial_number_or_name="239222303277", fps=30, width=640, height=480
                ),
                # add a second physical camera here for "wrist_rgb" if present
            },
        )

    Each ``cameras`` key K, crossed with each resize mode M, produces an
    observation image under ``"{K}_{M}"`` (e.g. ``base_rgb_letterbox``) — exactly
    matching the keys the simulation robot emits.
    """

    # Default to a single base camera. Override ``cameras`` to add the D455(s);
    # keep ``camera_names`` in sync with the ``cameras`` keys you provide.
    pass
