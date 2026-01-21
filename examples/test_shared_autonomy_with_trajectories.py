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
Test Shared Autonomy with Trajectory Database

This script uses recorded trajectories from a zarr database as simulated
"human input" to test PI0.5 shared autonomy. It follows the same policy
loading pattern as lerobot_record.py with proper preprocessors/postprocessors.

Features:
- Tests multiple forward_flow_ratios to blend human and policy actions
- Computes deviation metrics between human and policy trajectories
- Optionally saves policy trajectories back to zarr format with auto-incrementing
- Supports both offline testing (dummy observations) and online testing (real robot)
- Real robot mode gets observations from SplatSim and sends actions back

Usage (offline testing with dummy observations):
    python examples/test_shared_autonomy_with_trajectories.py \
        --policy_path /path/to/pi05/model \
        --traj_folder /path/to/trajectories.zarr \
        --dataset_repo_id your/dataset \
        --num_trajectories 5 \
        --forward_flow_ratios 0.0 0.2 0.4 0.6 0.8 1.0

Usage (online testing with real robot):
    python examples/test_shared_autonomy_with_trajectories.py \
        --policy_path /path/to/pi05/model \
        --traj_folder /path/to/trajectories.zarr \
        --dataset_repo_id your/dataset \
        --num_trajectories 5 \
        --forward_flow_ratios 0.0 0.2 0.4 0.6 0.8 1.0 \
        --use_robot \
        --robot_hostname 127.0.0.1 \
        --robot_port 6001 \
        --base_camera_port 5001 \
        --control_rate_hz 50

Usage (with trajectory saving):
    python examples/test_shared_autonomy_with_trajectories.py \
        --policy_path /path/to/pi05/model \
        --traj_folder /path/to/trajectories.zarr \
        --dataset_repo_id your/dataset \
        --num_trajectories 5 \
        --forward_flow_ratios 0.0 0.2 0.4 0.6 0.8 1.0 \
        --use_robot \
        --save_trajectories \
        --save_ratios 0.4 0.6 0.8 \
        --output_suffix _sapi05

The output zarr folder will have the same structure as input with suffix added.
For example: trajectories.zarr -> trajectories_sapi05.zarr
Trajectories are auto-incremented within each scenario/obstacle_config group.

Note: When using --use_robot, make sure the SplatSim server is running and accessible
at the specified hostname/port (matching lerobot-record connection parameters).
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import time

import numpy as np
import torch
import zarr
from torch import Tensor
from scipy.interpolate import interp1d


# Add SplatSim to path if needed
splatsim_path = Path.home() / "code" / "SplatSim"
if str(splatsim_path) not in sys.path:
    sys.path.insert(0, str(splatsim_path))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.processor_pi05 import make_pi05_inverse_post_processor
from lerobot.processor import PolicyAction
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor.core import TransitionKey
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05 import SharedAutonomyConfig


class TrajectoryLoader:
    """Loads trajectories from zarr database."""

    def __init__(self, traj_folder: str):
        """Load trajectory database.

        Args:
            traj_folder: Path to .zarr trajectory folder
        """
        assert traj_folder.endswith('.zarr'), "Currently only .zarr trajectory folder is supported."

        self.scenarios_groups = zarr.open(traj_folder, mode='r')['trajectories']
        self.trajectories = self._load_trajectory_list()

        print(f"Loaded {len(self.trajectories)} trajectories from {traj_folder}")

    def _load_trajectory_list(self):
        """Parse zarr structure and extract trajectory list."""
        scenario_re = re.compile(r"^scenario_(\d+)$")
        existing_ids = []

        for name in self.scenarios_groups.keys():
            m = scenario_re.match(name)
            if m:
                existing_ids.append(int(m.group(1)))

        existing_ids.sort()
        trajectories = []

        for scenario_id in existing_ids:
            scenario_name = f'scenario_{scenario_id:04d}'
            scenarios_group = self.scenarios_groups[scenario_name]

            obstacle_re = re.compile(r"^obstacle_config_(\d+)$")
            for obstacle_name in scenarios_group.keys():
                if obstacle_re.match(obstacle_name):
                    obstacle_config = scenarios_group[obstacle_name]
                    obstacle_config_json = json.loads(obstacle_config.attrs['metadata'])

                    traj_re = re.compile(r"^traj_(\d+)$")
                    for trajs_name in obstacle_config.keys():
                        if traj_re.match(trajs_name):
                            traj_group = obstacle_config[trajs_name]
                            qs = np.array(traj_group['qs'])

                            trajectories.append({
                                "qs": qs,
                                "metadata": obstacle_config_json,
                                "name": f"{scenario_name}_{obstacle_name}_{trajs_name}",
                                "zarr_group": traj_group,
                            })

        return trajectories

    def get_trajectory(self, idx: int):
        """Get trajectory by index."""
        return self.trajectories[idx]

    def __len__(self):
        return len(self.trajectories)


class TrajectorySaver:
    """Saves trajectories to zarr database with the same structure as input."""

    def __init__(self, input_traj_folder: str, output_suffix: str = "_sapi05"):
        """Initialize saver with output folder path.

        Args:
            input_traj_folder: Path to input .zarr trajectory folder
            output_suffix: Suffix to append to create output folder name
        """
        assert input_traj_folder.endswith('.zarr'), "Input folder must be .zarr"

        # Create output folder path by adding suffix before .zarr
        self.input_folder = input_traj_folder
        self.output_folder = input_traj_folder.replace('.zarr', f'{output_suffix}.zarr')

        # Open output zarr in append mode (create if doesn't exist)
        self.root = zarr.open(self.output_folder, mode='a')

        # Ensure trajectories group exists
        if 'trajectories' not in self.root:
            self.root.create_group('trajectories')

        self.trajectories_group = self.root['trajectories']

        print(f"Trajectory saver initialized:")
        print(f"  Input:  {self.input_folder}")
        print(f"  Output: {self.output_folder}")

    def _get_next_traj_number(self, obstacle_config_group) -> int:
        """Find the next available trajectory number in an obstacle config.

        Args:
            obstacle_config_group: The zarr group for the obstacle config

        Returns:
            Next available trajectory number (e.g., 0, 1, 2, ...)
        """
        traj_re = re.compile(r"^traj_(\d+)$")
        existing_nums = []

        for key in obstacle_config_group.keys():
            m = traj_re.match(key)
            if m:
                existing_nums.append(int(m.group(1)))

        if not existing_nums:
            return 0

        return max(existing_nums) + 1

    def save_trajectory(
        self,
        scenario_name: str,
        obstacle_config_name: str,
        qs: np.ndarray,
        metadata: dict,
    ) -> str:
        """Save a trajectory to the zarr database.

        Args:
            scenario_name: Scenario name (e.g., "scenario_0000")
            obstacle_config_name: Obstacle config name (e.g., "obstacle_config_00")
            qs: Joint positions array [T, action_dim]
            metadata: Obstacle configuration metadata dict

        Returns:
            Full trajectory path (e.g., "scenario_0000_obstacle_config_00_traj_02")
        """
        # Ensure scenario group exists
        if scenario_name not in self.trajectories_group:
            self.trajectories_group.create_group(scenario_name)
        scenario_group = self.trajectories_group[scenario_name]

        # Ensure obstacle config group exists
        if obstacle_config_name not in scenario_group:
            obs_group = scenario_group.create_group(obstacle_config_name)
            # Store metadata as JSON in attrs
            obs_group.attrs['metadata'] = json.dumps(metadata)
        else:
            obs_group = scenario_group[obstacle_config_name]

        # Find next available trajectory number
        traj_num = self._get_next_traj_number(obs_group)
        traj_name = f"traj_{traj_num:02d}"

        # Create trajectory group and save qs
        traj_group = obs_group.create_group(traj_name)
        traj_group.create_dataset('qs', data=qs, dtype='f4')

        full_name = f"{scenario_name}_{obstacle_config_name}_{traj_name}"
        print(f"  Saved: {full_name} (shape: {qs.shape})")

        return full_name

    def save_from_test_result(
        self,
        traj_data: dict,
        policy_actions: np.ndarray,
        forward_flow_ratio: float,
    ) -> str:
        """Save policy actions from a test result using original trajectory metadata.

        Args:
            traj_data: Original trajectory data dict (from TrajectoryLoader.get_trajectory)
            policy_actions: Policy action array [T, action_dim]
            forward_flow_ratio: The forward flow ratio used for this test

        Returns:
            Full trajectory path
        """
        # Parse scenario and obstacle config from original name
        # Format: "scenario_0000_obstacle_config_00_traj_00"
        name_parts = traj_data['name'].split('_')
        scenario_name = f"{name_parts[0]}_{name_parts[1]}"  # "scenario_0000"
        obstacle_config_name = f"{name_parts[2]}_{name_parts[3]}_{name_parts[4]}"  # "obstacle_config_00"

        # Use original metadata
        metadata = traj_data['metadata']

        # Save trajectory
        return self.save_trajectory(
            scenario_name=scenario_name,
            obstacle_config_name=obstacle_config_name,
            qs=policy_actions,
            metadata=metadata,
        )


class SharedAutonomyTester:
    """Test shared autonomy with trajectory-based human input.

    This class properly loads the policy with preprocessors/postprocessors
    following the lerobot_record.py pattern.
    """

    def __init__(
        self,
        policy_path: str,
        dataset_repo_id: str,
        traj_folder: str,
        traj_replay_speed: float = 1.0,
        device: str = "cuda",
        task_description: str = "obstacle avoidance in scenario_0000",
        use_robot: bool = False,
        robot_hostname: str = "127.0.0.1",
        robot_port: int = 6001,
        base_camera_port: int = 5001,
        control_rate_hz: int = 50,
        n_action_steps: Optional[int] = 10,
    ):
        """Initialize tester with policy and trajectory loader.

        Args:
            policy_path: Path to pretrained PI0.5 model
            dataset_repo_id: Dataset repo ID for loading metadata/stats
            traj_folder: Path to trajectory zarr folder
            device: Device to run policy on
            task_description: Language task description (placeholder for testing)
            use_robot: Whether to connect to real robot for observations
            robot_hostname: Robot IP address (if use_robot=True)
            robot_port: Robot ZMQ port (if use_robot=True)
            base_camera_port: Base camera port (if use_robot=True)
            control_rate_hz: Robot control rate (if use_robot=True)
            n_action_steps: Number of actions to execute from each chunk (None=use config default)
                Lower values create more overlap and smoother motion (e.g., 10 with chunk_size=50)
        """
        self.device = device
        self.policy_path = policy_path
        self.dataset_repo_id = dataset_repo_id
        self.use_robot = use_robot
        self.robot = None
        self.n_action_steps_override = n_action_steps
        self.traj_replay_speed = traj_replay_speed

        # Load trajectory database
        print(f"Loading trajectories from {traj_folder}...")
        self.traj_loader = TrajectoryLoader(traj_folder)

        # Load dataset metadata (needed for policy initialization)
        print(f"Loading dataset metadata from {dataset_repo_id}...")
        self.dataset_meta = LeRobotDatasetMetadata(dataset_repo_id)

        # Store task description (will be tokenized by PI05 preprocessor)
        print(f"Task description: '{task_description}'")
        self.task_description = task_description

        # Connect to robot if requested
        if use_robot:
            print(f"\nConnecting to SplatSim robot at {robot_hostname}:{robot_port}...")
            self._connect_robot(robot_hostname, robot_port, base_camera_port, control_rate_hz)

        # Load policy with proper preprocessing/postprocessing
        print(f"Loading policy from {policy_path}...")
        self._load_policy()

        print("Initialization complete!")

    def _connect_robot(
        self,
        hostname: str,
        robot_port: int,
        base_camera_port: int,
        control_rate_hz: int,
    ):
        """Connect to SplatSim robot.

        Args:
            hostname: Robot IP address
            robot_port: Robot ZMQ port
            base_camera_port: Base camera port
            control_rate_hz: Control rate
        """
        from lerobot.robots.splatsim_lerobot import SplatSimLerobot, SplatSimLerobotConfig

        # Create robot config
        robot_config = SplatSimLerobotConfig(
            hostname=hostname,
            robot_port=robot_port,
            base_camera_port=base_camera_port,
            control_rate_hz=control_rate_hz,
            use_wrist_camera=False,  # Match your dataset
        )

        # Create and connect robot
        self.robot = SplatSimLerobot(robot_config)
        self.robot.connect()

        print(f"  Robot connected successfully!")

    def _load_policy(self):
        """Load policy following lerobot_record.py pattern.

        This ensures proper preprocessors/postprocessors are created.
        """
        # Load policy config from pretrained path
        policy_config = PreTrainedConfig.from_pretrained(self.policy_path)
        policy_config.pretrained_path = Path(self.policy_path)
        policy_config.device = self.device

        # Reduce n_action_steps to create overlap between chunks for smoother motion
        # Default: chunk_size=50, n_action_steps=50 (no overlap, jerky at chunk boundaries)
        # Better: chunk_size=50, n_action_steps=10 (40-step overlap, smooth transitions)
        if self.n_action_steps_override is not None:
            if hasattr(policy_config, 'n_action_steps') and hasattr(policy_config, 'chunk_size'):
                original_n_action_steps = policy_config.n_action_steps
                policy_config.n_action_steps = min(self.n_action_steps_override, policy_config.chunk_size)
                if original_n_action_steps != policy_config.n_action_steps:
                    print(f"  Reducing n_action_steps: {original_n_action_steps} → {policy_config.n_action_steps}")
                    overlap = policy_config.chunk_size - policy_config.n_action_steps
                    print(f"    (Creates {overlap}-step overlap between chunks for smoother motion)")
                    print(f"    (Predicts new chunk every {policy_config.n_action_steps} steps instead of {original_n_action_steps})")

        # Create policy
        self.policy = make_policy(policy_config, ds_meta=self.dataset_meta)

        # Create preprocessors and postprocessors
        # This is critical - same pattern as lerobot_record.py line 472-480
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=policy_config.pretrained_path,
            dataset_stats=rename_stats(self.dataset_meta.stats, {}),  # No rename map
            preprocessor_overrides={
                "device_processor": {"device": policy_config.device},
            },
        )

        # Create inverse postprocessor to normalize human actions
        # CRITICAL: Must use the SAME stats as the postprocessor for the round-trip to work!
        # The postprocessor is loaded from pretrained, so we extract its stats rather than
        # using trajectory stats which would break the normalization round-trip.
        # Extract stats from the loaded postprocessor
        postprocessor_stats = None
        for step in self.postprocessor.steps:
            if hasattr(step, 'stats') and step.stats is not None:
                postprocessor_stats = step.stats
                break

        self.inverse_postprocessor, = make_pi05_inverse_post_processor(
            config=policy_config,
            dataset_stats=postprocessor_stats,  # Use same stats as postprocessor!
        )

        # Set policy to eval mode
        self.policy.eval()

        print(f"  Policy type: {policy_config.__class__.__name__}")
        print(f"  Device: {self.device}")
        print(f"  Preprocessor steps: {len(self.preprocessor.steps) if self.preprocessor else 0}")
        print(f"  Postprocessor steps: {len(self.postprocessor.steps) if self.postprocessor else 0}")

    def enable_shared_autonomy(self, forward_flow_ratio: float):
        """Enable shared autonomy with specified forward_flow_ratio.

        Args:
            forward_flow_ratio: Blending ratio (0.0 = pure human, 1.0 = pure policy)
        """
        sa_config = SharedAutonomyConfig(
            enabled=True,
            forward_flow_ratio=forward_flow_ratio,
            human_action_buffer_size=1,
        )

        self.policy.config.shared_autonomy_config = sa_config
        self.policy.init_shared_autonomy_processor()

        print(f"Shared autonomy enabled with forward_flow_ratio={forward_flow_ratio}")

    def disable_shared_autonomy(self):
        """Disable shared autonomy."""
        self.policy.config.shared_autonomy_config = None
        self.policy.sa_processor = None
        print("Shared autonomy disabled")

    def get_observation(self) -> dict:
        """Get observation from robot or create dummy observation.

        Returns:
            Dictionary with observation keys matching policy expectations
        """
        if self.use_robot and self.robot is not None:
            # Get real observation from robot
            return self._get_robot_observation()
        else:
            # Create dummy observation for testing without robot
            return self._get_dummy_observation()

    def _get_robot_observation(self) -> dict:
        """Get observation from connected robot.

        Returns:
            Dictionary with observation keys matching policy expectations
        """
        # Get raw observation from robot
        robot_obs = self.robot.get_observation()

        obs_dict = {}

        # Add images
        for key, feature_info in self.dataset_meta.features.items():
            if feature_info.get("dtype") in ["image", "video"]:
                # Extract camera name from key (e.g., "observation.images.base_rgb" -> "base_rgb")
                camera_name = key.split(".")[-1]
                if camera_name in robot_obs:
                    # Robot returns (H, W, C), convert to (C, H, W) for policy
                    image = robot_obs[camera_name]
                    image_chw = np.transpose(image, (2, 0, 1))
                    # Convert to torch tensor and move to device
                    obs_dict[key] = torch.from_numpy(image_chw).float().to(self.device)

        # Add state (joint positions)
        # Robot returns individual joint values, combine into array
        state_values = []
        for joint_name in self.robot.config.joint_names:
            if joint_name in robot_obs:
                state_values.append(robot_obs[joint_name])

        if state_values:
            state_array = np.array(state_values, dtype=np.float32)
            obs_dict["observation.state"] = torch.from_numpy(state_array).to(self.device)

        # Add batch dimension for observations
        for key in obs_dict:
            obs_dict[key] = obs_dict[key].unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

        # Add task description as top-level key (batch_to_transition will move it to complementary_data)
        obs_dict["task"] = [self.task_description]  # List with batch size 1

        return obs_dict

    def _get_dummy_observation(self) -> dict:
        """Create dummy observation matching dataset format.

        For testing without robot connection.

        Returns:
            Dictionary with observation keys matching policy expectations
        """
        # Get observation keys from dataset metadata
        obs_dict = {}

        # Add image observations
        for key, feature_info in self.dataset_meta.features.items():
            if feature_info.get("dtype") in ["image", "video"]:
                # Create dummy image [C, H, W]
                obs_dict[key] = torch.randn(3, 224, 224, device=self.device)

        # Add state observations
        for key, feature_info in self.dataset_meta.features.items():
            if "state" in key.lower() and feature_info.get("dtype") not in ["image", "video"]:
                # Get state dim from metadata
                state_shape = feature_info.get("shape", [32])
                state_dim = state_shape[0] if isinstance(state_shape, (list, tuple)) else state_shape
                obs_dict[key] = torch.randn(state_dim, device=self.device)

        # Add batch dimension for observations
        for key in obs_dict:
            obs_dict[key] = obs_dict[key].unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

        # Add task description as top-level key (batch_to_transition will move it to complementary_data)
        obs_dict["task"] = [self.task_description]  # List with batch size 1

        return obs_dict

    def predict_action(
        self,
        observation: dict,
        human_action: Optional[Tensor] = None,
    ) -> PolicyAction:
        """Predict action with optional human input.

        This follows the same pattern as lerobot_record.py's predict_action function.

        Args:
            observation: Dictionary of observations
            human_action: Optional human action tensor for shared autonomy

        Returns:
            PolicyAction dictionary
        """
        # Preprocess observation
        if self.preprocessor is not None:
            observation = self.preprocessor(observation)

        # Provide human action if shared autonomy is enabled AND human action is provided
        # human_action is only provided when the policy's action queue is empty
        if human_action is not None and self.policy.sa_processor is not None:
            if not isinstance(human_action, Tensor):
                human_action = torch.tensor(human_action, dtype=torch.float32, device=self.device)
            
            # human_action shape: [batch, timesteps, action_dim] or [batch, action_dim]
            batch_size = human_action.shape[0]
            num_timesteps = human_action.shape[1] if human_action.ndim == 3 else 1
            action_dim = human_action.shape[-1]
            
            # Reshape to [batch*timesteps, action_dim] for normalization
            human_action_flat = human_action.reshape(-1, action_dim)
            
            # Normalize each timestep using inverse_postprocessor
            # (inverse_postprocessor: unnormalized -> normalized)
            human_action_normalized = self.inverse_postprocessor(human_action_flat)
            
            # Reshape back to [batch, timesteps, action_dim]
            human_action = human_action_normalized.reshape(batch_size, num_timesteps, -1)
            
            # Pad the human action to chunk_size if needed. Repeat the last action until you reach chunk_size.
            if human_action.shape[1] < self.policy.config.chunk_size:
                padding = torch.tile(
                    human_action[:, -1:, :],
                    (1, self.policy.config.chunk_size - human_action.shape[1], 1)
                )
                human_action = torch.cat([human_action, padding], axis=1)  # [batch, chunk_size, action_dim]

            max_action_dim = self.policy.config.max_action_dim
            if human_action.shape[-1] < max_action_dim:
                chunk_size = self.policy.config.chunk_size
                padding = torch.zeros((batch_size, chunk_size, max_action_dim - human_action.shape[-1])).to(human_action.device)
                human_action = torch.cat([human_action, padding], axis=2)  # [batch, chunk_size, max_action_dim]

            self.policy.set_human_action(human_action)

        # Get policy action
        with torch.no_grad():
            policy_action = self.policy.select_action(observation)

        # import pdb; pdb.set_trace()

        # Postprocess action
        if self.postprocessor is not None:
            policy_action = self.postprocessor(policy_action)
        # import pdb; pdb.set_trace()

        return policy_action

    def test_trajectory(
        self,
        traj_idx: int,
        forward_flow_ratio: float,
        max_steps: Optional[int] = None,
    ) -> dict:
        """Test a single trajectory with shared autonomy.

        Args:
            traj_idx: Trajectory index to test
            forward_flow_ratio: Blending ratio
            max_steps: Maximum number of steps to test (None = full trajectory)

        Returns:
            Dictionary with results
        """
        # Get trajectory
        traj_data = self.traj_loader.get_trajectory(traj_idx)
        traj_name = traj_data["name"]
        trajectory_qs = traj_data["qs"]  # [T, action_dim]

        if self.traj_replay_speed != 1.0:
            print(f"  Adjusting trajectory replay speed: {self.traj_replay_speed}x")
            # Adjust timing to match replay speed
            original_num_timesteps = trajectory_qs.shape[0]
            target_num_timesteps = int(len(trajectory_qs) / self.traj_replay_speed)
            # Interpolate trajectory_qs to match target_num_timesteps
            original_timesteps = np.arange(trajectory_qs.shape[0])
            target_timesteps = np.linspace(0, trajectory_qs.shape[0] - 1, target_num_timesteps)
            interpolator = interp1d(original_timesteps, trajectory_qs, axis=0, kind='linear')
            trajectory_qs = interpolator(target_timesteps)

            if target_num_timesteps < original_num_timesteps:
                # Pad the end with the last position to maintain original length
                padding = np.tile(trajectory_qs[-1:], (original_num_timesteps - target_num_timesteps, 1))
                trajectory_qs = np.vstack([trajectory_qs, padding])

        self.policy.reset()

        print(f"\nTesting trajectory: {traj_name}")
        print(f"  Length: {len(trajectory_qs)} steps")
        print(f"  Forward flow ratio: {forward_flow_ratio}")

        # Limit steps if specified
        T = min(len(trajectory_qs), max_steps) if max_steps else len(trajectory_qs)

        # Storage
        human_actions = []
        policy_actions = []

        chunk_size = self.policy.config.chunk_size

        # Initialize the robot with the starting position if connected
        if self.use_robot and self.robot is not None:
            initial_q = trajectory_qs[0]
            if initial_q.shape[0] == 6:
                initial_q = np.append(initial_q, 0.0)  # Add gripper
            for i in range(50):
                self.robot.send_action(initial_q)
                time.sleep(0.02)  # Small delay to ensure robot receives initial position
            self.robot.send_action(initial_q)
            print("  Sent initial position to robot.")

        # Step through trajectory
        for t in range(T):
            # Get observation (from robot if connected, otherwise dummy)
            obs = self.get_observation()

            # Only provide human action when policy queue is empty
            # This prevents re-predicting every timestep and maintains temporal consistency
            human_action_tensor = None
            if len(self.policy._action_queue) == 0:
                # Get human action chunk from trajectory
                human_action_np = trajectory_qs[t : t + chunk_size]  # [prediction horizon, action_dim]

                # Pad 6-DOF to 7-DOF (add gripper value at index 6)
                if human_action_np.shape[1] == 6:
                    # Add gripper dimension (default to 0 = gripper open)
                    human_action_np = np.insert(human_action_np, 6, 0.0, axis=1)

                # Convert to tensor with batch dimension
                human_action_tensor = torch.tensor(
                    human_action_np,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)  # [1, chunk_size, 7]

            # Predict with shared autonomy (only uses human_action if queue is empty)
            policy_action = self.predict_action(obs, human_action_tensor)

            # Extract action (policy_action is a dict with 'action' key)
            policy_action_np = policy_action.cpu().numpy()[0]

            # Send action to robot if connected
            if self.use_robot and self.robot is not None:
                self.robot.send_action(policy_action_np)

            # Store (use 7-DOF for comparison, policy output is also 7-DOF after postprocessing)
            # For human action, use the current timestep from trajectory (not the chunk)
            current_human_action = trajectory_qs[t]  # [action_dim]
            if current_human_action.shape[0] == 6:
                current_human_action = np.append(current_human_action, 0.0)  # Add gripper
            human_actions.append(current_human_action[:7])
            policy_actions.append(policy_action_np[:7])  # Take only first 7 dims

        # Convert to arrays
        human_actions = np.array(human_actions)
        policy_actions = np.array(policy_actions)

        # Compute metrics
        deviation = np.linalg.norm(policy_actions - human_actions, axis=1)
        metrics = {
            "mean_deviation": float(deviation.mean()),
            "std_deviation": float(deviation.std()),
            "max_deviation": float(deviation.max()),
            "min_deviation": float(deviation.min()),
        }

        print(f"  Mean deviation: {metrics['mean_deviation']:.4f}")

        return {
            "traj_name": traj_name,
            "traj_data": traj_data,  # Include original trajectory data
            "forward_flow_ratio": forward_flow_ratio,
            "num_steps": T,
            "metrics": metrics,
            "human_actions": human_actions,
            "policy_actions": policy_actions,
        }

    def test_multiple_ratios(
        self,
        traj_idx: int,
        forward_flow_ratios: list[float],
        max_steps: Optional[int] = None,
    ) -> dict:
        """Test trajectory with multiple forward_flow_ratio values.

        Args:
            traj_idx: Trajectory index
            forward_flow_ratios: List of ratios to test
            max_steps: Maximum steps per test

        Returns:
            Dictionary with results for all ratios
        """
        results = {}

        for ratio in forward_flow_ratios:
            # Enable shared autonomy with this ratio
            self.enable_shared_autonomy(ratio)

            # Test trajectory
            result = self.test_trajectory(traj_idx, ratio, max_steps)
            results[ratio] = result

        return results

    def disconnect(self):
        """Disconnect from robot if connected."""
        if self.robot is not None:
            print("\nDisconnecting from robot...")
            self.robot.disconnect()
            self.robot = None

    def __del__(self):
        """Cleanup on deletion."""
        if self.robot is not None:
            try:
                self.disconnect()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Test shared autonomy with trajectory database"
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to pretrained PI0.5 model",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repo ID (for loading metadata/stats)",
    )
    parser.add_argument(
        "--traj_folder",
        type=str,
        required=True,
        help="Path to zarr trajectory folder",
    )
    parser.add_argument(
        "--traj_replay_speed",
        type=float,
        default=1.0,
        help="Speed multiplier for trajectory replay",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=3,
        help="Number of trajectories to test",
    )
    parser.add_argument(
        "--forward_flow_ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Forward flow ratios to test",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum steps per trajectory (None = full trajectory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/shared_autonomy_tests",
        help="Output directory for results",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="obstacle avoidance in scenario_0000",
        help="Language task description (placeholder for testing)",
    )
    parser.add_argument(
        "--save_trajectories",
        action="store_true",
        help="Save policy trajectories to zarr format",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_sapi05",
        help="Suffix to add to output zarr folder name (default: _sapi05)",
    )
    parser.add_argument(
        "--save_ratios",
        type=float,
        nargs="+",
        default=None,
        help="Which forward_flow_ratios to save (default: all tested ratios)",
    )
    parser.add_argument(
        "--use_robot",
        action="store_true",
        help="Connect to SplatSim robot for real observations and action execution",
    )
    parser.add_argument(
        "--robot_hostname",
        type=str,
        default="127.0.0.1",
        help="Robot IP address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--robot_port",
        type=int,
        default=6001,
        help="Robot ZMQ port (default: 6001)",
    )
    parser.add_argument(
        "--base_camera_port",
        type=int,
        default=5001,
        help="Base camera port (default: 5001)",
    )
    parser.add_argument(
        "--control_rate_hz",
        type=int,
        default=50,
        help="Robot control rate in Hz (default: 50)",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=10,
        help="Number of actions to execute from each predicted chunk (default: 10). "
             "Lower values create more overlap between chunks for smoother motion. "
             "Set to None to use policy's default (usually 50, which causes jerky motion).",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("SHARED AUTONOMY TESTING WITH TRAJECTORY DATABASE")
    print("="*80)

    # Initialize tester
    tester = SharedAutonomyTester(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        traj_folder=args.traj_folder,
        traj_replay_speed=args.traj_replay_speed,
        device=args.device,
        task_description=args.task_description,
        use_robot=args.use_robot,
        robot_hostname=args.robot_hostname,
        robot_port=args.robot_port,
        base_camera_port=args.base_camera_port,
        control_rate_hz=args.control_rate_hz,
        n_action_steps=args.n_action_steps,
    )

    # Initialize trajectory saver if requested
    traj_saver = None
    if args.save_trajectories:
        traj_saver = TrajectorySaver(
            input_traj_folder=args.traj_folder,
            output_suffix=args.output_suffix,
        )
        print()

    # Determine which ratios to save
    save_ratios_set = set(args.save_ratios) if args.save_ratios else set(args.forward_flow_ratios)

    # Test multiple trajectories
    all_results = []

    for traj_idx in range(min(args.num_trajectories, len(tester.traj_loader))):
        print(f"\n{'='*80}")
        print(f"TRAJECTORY {traj_idx + 1}/{args.num_trajectories}")
        print(f"{'='*80}")

        results = tester.test_multiple_ratios(
            traj_idx=traj_idx,
            forward_flow_ratios=args.forward_flow_ratios,
            max_steps=args.max_steps,
        )

        all_results.append(results)

        # Save trajectories if requested
        if traj_saver:
            print("\nSaving trajectories:")
            for ratio, result in results.items():
                if ratio in save_ratios_set:
                    # Convert policy_actions back to 6-DOF (remove gripper) if needed
                    policy_actions_6dof = result["policy_actions"][:, :6]

                    traj_saver.save_from_test_result(
                        traj_data=result["traj_data"],
                        policy_actions=policy_actions_6dof,
                        forward_flow_ratio=ratio,
                    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"shared_autonomy_results_{timestamp}.json"
    )

    # Prepare for JSON (convert numpy arrays to lists)
    results_json = []
    for traj_results in all_results:
        traj_json = {}
        for ratio, result in traj_results.items():
            traj_json[str(ratio)] = {
                "traj_name": result["traj_name"],
                "num_steps": result["num_steps"],
                "metrics": result["metrics"],
            }
        results_json.append(traj_json)

    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    if traj_saver:
        print(f"Trajectories saved to: {traj_saver.output_folder}")
        total_saved = len(save_ratios_set) * args.num_trajectories
        print(f"  Total trajectories saved: {total_saved}")
        print(f"  Ratios saved: {sorted(save_ratios_set)}")
    print(f"{'='*80}")

    # Cleanup
    tester.disconnect()


if __name__ == "__main__":
    main()
