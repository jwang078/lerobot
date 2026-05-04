# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass, field, fields
from typing import Any

import draccus
import gymnasium as gym
from gymnasium.envs.registration import registry as gym_registry

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.processor import IsaaclabArenaProcessorStep, LiberoProcessorStep, PolicyProcessorPipeline
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import (
    ACTION,
    LIBERO_KEY_EEF_MAT,
    LIBERO_KEY_EEF_POS,
    LIBERO_KEY_EEF_QUAT,
    LIBERO_KEY_GRIPPER_QPOS,
    LIBERO_KEY_GRIPPER_QVEL,
    LIBERO_KEY_JOINTS_POS,
    LIBERO_KEY_JOINTS_VEL,
    LIBERO_KEY_PIXELS_AGENTVIEW,
    LIBERO_KEY_PIXELS_EYE_IN_HAND,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
)


def _make_vec_env_cls(use_async: bool, n_envs: int):
    """Return the right VectorEnv constructor."""
    if use_async and n_envs > 1:
        return gym.vector.AsyncVectorEnv
    return gym.vector.SyncVectorEnv


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    def package_name(self) -> str:
        """Package name to import if environment not found in gym registry"""
        return f"gym_{self.type}"

    @property
    def gym_id(self) -> str:
        """ID string used in gym.make() to instantiate the environment"""
        return f"{self.package_name}/{self.task}"

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()

    def create_envs(
        self,
        n_envs: int,
        use_async_envs: bool = False,
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """Create {suite: {task_id: VectorEnv}}.

        Default: single-task env via gym.make(). Multi-task benchmarks override.
        AsyncVectorEnv is the default for n_envs > 1; auto-downgraded to Sync for n_envs=1.
        """
        env_cls = gym.vector.AsyncVectorEnv if (use_async_envs and n_envs > 1) else gym.vector.SyncVectorEnv

        if self.gym_id not in gym_registry:
            print(f"gym id '{self.gym_id}' not found, attempting to import '{self.package_name}'...")
            try:
                importlib.import_module(self.package_name)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Package '{self.package_name}' required for env '{self.type}' not found. "
                    f"Please install it or check PYTHONPATH."
                ) from e

            if self.gym_id not in gym_registry:
                raise gym.error.NameNotFound(
                    f"Environment '{self.gym_id}' not registered even after importing '{self.package_name}'."
                )

        def _make_one():
            return gym.make(self.gym_id, disable_env_checker=self.disable_env_checker, **self.gym_kwargs)

        extra_kwargs: dict = {}
        if env_cls is gym.vector.AsyncVectorEnv:
            extra_kwargs["context"] = "forkserver"
        try:
            from gymnasium.vector import AutoresetMode

            vec = env_cls(
                [_make_one for _ in range(n_envs)], autoreset_mode=AutoresetMode.SAME_STEP, **extra_kwargs
            )
        except ImportError:
            vec = env_cls([_make_one for _ in range(n_envs)], **extra_kwargs)
        return {self.type: {0: vec}}

    def get_env_processors(self):
        """Return (preprocessor, postprocessor) for this env. Default: identity."""
        return PolicyProcessorPipeline(steps=[]), PolicyProcessorPipeline(steps=[])


@dataclass
class HubEnvConfig(EnvConfig):
    """Base class for environments that delegate creation to a hub-hosted make_env.

    Hub environments download and execute remote code from the HF Hub.
    The hub_path points to a repository containing an env.py with a make_env function.
    """

    hub_path: str | None = None  # required: e.g., "username/repo" or "username/repo@branch:file.py"

    @property
    def gym_kwargs(self) -> dict:
        # Not used for hub environments - the hub's make_env handles everything
        return {}


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str | None = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    observation_height: int = 480
    observation_width: int = 640
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str | None = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    observation_height: int = 384
    observation_width: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@dataclass
class ImagePreprocessingConfig:
    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None


@dataclass
class RewardClassifierConfig:
    """Configuration for reward classification."""

    pretrained_path: str | None = None
    success_threshold: float = 0.5
    success_reward: float = 1.0


@dataclass
class InverseKinematicsConfig:
    """Configuration for inverse kinematics processing."""

    urdf_path: str | None = None
    target_frame_name: str | None = None
    end_effector_bounds: dict[str, list[float]] | None = None
    end_effector_step_sizes: dict[str, float] | None = None


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""

    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    add_ee_pose_to_observation: bool = False
    display_cameras: bool = False


@dataclass
class GripperConfig:
    """Configuration for gripper control and penalties."""

    use_gripper: bool = True
    gripper_penalty: float = 0.0


@dataclass
class ResetConfig:
    """Configuration for environment reset behavior."""

    fixed_reset_joint_positions: Any | None = None
    reset_time_s: float = 5.0
    control_time_s: float = 20.0
    terminate_on_success: bool = True


@dataclass
class HILSerlProcessorConfig:
    """Configuration for environment processing pipeline."""

    control_mode: str = "gamepad"
    observation: ObservationConfig | None = None
    image_preprocessing: ImagePreprocessingConfig | None = None
    gripper: GripperConfig | None = None
    reset: ResetConfig | None = None
    inverse_kinematics: InverseKinematicsConfig | None = None
    reward_classifier: RewardClassifierConfig | None = None
    max_gripper_pos: float | None = 100.0


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    processor: HILSerlProcessorConfig = field(default_factory=HILSerlProcessorConfig)

    name: str = "real_robot"

    @property
    def gym_kwargs(self) -> dict:
        return {}


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    task_ids: list[int] | None = None
    fps: int = 30
    episode_length: int | None = None
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True
    camera_name_mapping: dict[str, str] | None = None
    observation_height: int = 360
    observation_width: int = 360
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            LIBERO_KEY_EEF_POS: f"{OBS_STATE}.eef_pos",
            LIBERO_KEY_EEF_QUAT: f"{OBS_STATE}.eef_quat",
            LIBERO_KEY_EEF_MAT: f"{OBS_STATE}.eef_mat",
            LIBERO_KEY_GRIPPER_QPOS: f"{OBS_STATE}.gripper_qpos",
            LIBERO_KEY_GRIPPER_QVEL: f"{OBS_STATE}.gripper_qvel",
            LIBERO_KEY_JOINTS_POS: f"{OBS_STATE}.joint_pos",
            LIBERO_KEY_JOINTS_VEL: f"{OBS_STATE}.joint_vel",
            LIBERO_KEY_PIXELS_AGENTVIEW: f"{OBS_IMAGES}.image",
            LIBERO_KEY_PIXELS_EYE_IN_HAND: f"{OBS_IMAGES}.image2",
        }
    )
    control_mode: str = "relative"  # or "absolute"

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features[LIBERO_KEY_PIXELS_AGENTVIEW] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features[LIBERO_KEY_PIXELS_EYE_IN_HAND] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features[LIBERO_KEY_PIXELS_AGENTVIEW] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features[LIBERO_KEY_PIXELS_EYE_IN_HAND] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )
            self.features[LIBERO_KEY_EEF_POS] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(3,),
            )
            self.features[LIBERO_KEY_EEF_QUAT] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(4,),
            )
            self.features[LIBERO_KEY_EEF_MAT] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(3, 3),
            )
            self.features[LIBERO_KEY_GRIPPER_QPOS] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(2,),
            )
            self.features[LIBERO_KEY_GRIPPER_QVEL] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(2,),
            )
            self.features[LIBERO_KEY_JOINTS_POS] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(7,),
            )
            self.features[LIBERO_KEY_JOINTS_VEL] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(7,),
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        if self.camera_name_mapping is not None:
            mapped_agentview = self.camera_name_mapping.get("agentview_image", "image")
            mapped_eye_in_hand = self.camera_name_mapping.get("robot0_eye_in_hand_image", "image2")
            self.features_map[LIBERO_KEY_PIXELS_AGENTVIEW] = f"{OBS_IMAGES}.{mapped_agentview}"
            self.features_map[LIBERO_KEY_PIXELS_EYE_IN_HAND] = f"{OBS_IMAGES}.{mapped_eye_in_hand}"

    @property
    def gym_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
        }
        if self.task_ids is not None:
            kwargs["task_ids"] = self.task_ids
        return kwargs

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .libero import create_libero_envs

        if self.task is None:
            raise ValueError("LiberoEnv requires a task to be specified")
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        return create_libero_envs(
            task=self.task,
            n_envs=n_envs,
            camera_name=self.camera_name,
            init_states=self.init_states,
            gym_kwargs=self.gym_kwargs,
            env_cls=env_cls,
            control_mode=self.control_mode,
            episode_length=self.episode_length,
            camera_name_mapping=self.camera_name_mapping,
        )

    def get_env_processors(self):
        return (
            PolicyProcessorPipeline(steps=[LiberoProcessorStep()]),
            PolicyProcessorPipeline(steps=[]),
        )


@EnvConfig.register_subclass("metaworld")
@dataclass
class MetaworldEnv(EnvConfig):
    task: str = "metaworld-push-v2"  # add all tasks
    fps: int = 80
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    multitask_eval: bool = True
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}",
            "pixels/top": f"{OBS_IMAGE}",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 480, 3))

        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
        }

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        from .metaworld import create_metaworld_envs

        if self.task is None:
            raise ValueError("MetaWorld requires a task to be specified")
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        return create_metaworld_envs(
            task=self.task,
            n_envs=n_envs,
            gym_kwargs=self.gym_kwargs,
            env_cls=env_cls,
        )


@EnvConfig.register_subclass("splatsim")
@dataclass
class SplatSimEnv(EnvConfig):
    """Configuration for SplatSim Gym environment.

    SplatSim is a Gaussian splatting-based robot simulation environment.
    The actual Gym environment is defined in the splatsim package and must be
    registered with gymnasium before use.

    Example usage:
        lerobot-eval \\
            --policy.path=your/checkpoint \\
            --env.type=splatsim \\
            --env.task=upright_small_engine_new \\
            --eval.n_episodes=50
    """

    task: str = "upright_small_engine_new"
    fps: int = 30
    episode_length: int = 400  # 8 seconds at 50 fps, matching Aloha
    render_mode: str = "rgb_array"

    # Task description for language-conditioned policies (e.g., PI0, PI05)
    # This should match the task description used during training
    task_description: str | None = None

    # SplatSim-specific config
    robot_name: str = "robot_iphone_w_engine_new"
    cam_i: int = 3
    camera_names: list[str] = field(default_factory=lambda: ["base_rgb", "wrist_rgb"])
    use_gripper: bool = True
    debug_mode: str = "off"

    port: int | None = None

    # Run in eval_benchmark mode: restore pre-recorded episode scenarios on each reset().
    # Set to the LeRobot repo ID (e.g. "user/my-eval-dataset") of the dataset whose
    # episode scenarios should be cycled through. Each env.reset() advances to the next episode.
    eval_benchmark_repo_id: str | None = None

    # Optional subset of episode indices to evaluate in eval_benchmark mode.
    # If None, all episodes in the dataset are used (0..N-1).
    # Example: [3, 8, 23, 38] to evaluate only those episodes from the benchmark dataset.
    eval_benchmark_subset: list[int] | None = None

    # Connect to an already-running SplatSim server instead of launching a new one.
    # When set, lerobot-eval uses ZMQSplatSimGymEnv on this port rather than
    # spawning a PybulletRobotServerBase. Useful for shared-autonomy eval where
    # gello is also connected to the same running simulator.
    external_port: int | None = None
    external_host: str = "127.0.0.1"

    # When True, the gym env exposes get_env_config() so the policy / wrapper can
    # access obstacle geometry and the task goal (q_goal_bias, target_ee_pos/quat).
    # Required for the shared autonomy wrapper's "RRT to Goal" mode.
    include_oracle_info: bool = False

    # Teleop recording: save pure-teleop (ratio=0) segments to a LeRobot dataset.
    # Set to a repo ID (e.g. "user/teleop-data") to enable; None to disable.
    teleop_dataset_repo_id: str | None = None
    teleop_min_episode_length: int = 60  # discard segments shorter than this

    # Image dimensions
    observation_height: int = 224
    observation_width: int = 224

    # Image resize mode:
    # - "letterbox": Resize keeping aspect ratio, pad with black bars (good for pretrained VLAs)
    # - "stretch": Resize to fill entire area without keeping aspect ratio (good for diffusion)
    image_resize_modes: list[str] = field(default_factory=lambda: ["letterbox"])

    # State dimension (6 joints + 1 gripper = 7)
    state_dim: int = 7
    # Action dimension (6 joints + 1 gripper = 7)
    action_dim: int = 7

    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels": OBS_IMAGE,
            "pixels/base_rgb": f"{OBS_IMAGES}.base_rgb",
            "pixels/wrist_rgb": f"{OBS_IMAGES}.wrist_rgb",
        }
    )

    def __post_init__(self):
        # Set state feature
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))

        # Set image features - always use "pixels/<camera_name>" format for consistency
        # This maps to "observation.images.<camera_name>" in LeRobot format
        for cam_name in self.camera_names:
            self.features[f"pixels/{cam_name}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3),
            )

    @property
    def gym_kwargs(self) -> dict:
        # When teleop recording is configured, force the SplatSim server to
        # render every ImageResizeMode regardless of what the policy itself
        # uses. This way the saved dataset always has every variant
        # ({cam}_letterbox, {cam}_stretch, ...) available for downstream
        # training of policies that may want a different resize mode. The
        # policy's preprocessor (rename_map) still picks whichever one it
        # was trained on.
        server_image_resize_modes = self.image_resize_modes
        if self.teleop_dataset_repo_id is not None:
            from splatsim.configs.mode_config import ImageResizeMode

            server_image_resize_modes = [m.value for m in ImageResizeMode]

        cfg = {
            "robot_name": self.robot_name,
            "camera_names": self.camera_names,
            "cam_i": self.cam_i,
            "use_gripper": self.use_gripper,
            "debug_mode": self.debug_mode,
            "image_resize_modes": server_image_resize_modes,
            "port": self.port,
        }
        # Include task_description if provided (for language-conditioned policies)
        if self.task_description is not None:
            cfg["task_description"] = self.task_description
        # Pass eval_benchmark_repo_id so the robot server loads the dataset on startup
        if self.eval_benchmark_repo_id is not None:
            cfg["eval_benchmark_repo_id"] = self.eval_benchmark_repo_id
        if self.eval_benchmark_subset is not None:
            cfg["eval_benchmark_subset"] = self.eval_benchmark_subset
        return {
            "cfg": cfg,
            "render_mode": self.render_mode,
        }

    def create_envs(self, n_envs: int, use_async_envs: bool = False) -> dict:
        env_cls = gym.vector.AsyncVectorEnv if (use_async_envs and n_envs > 1) else gym.vector.SyncVectorEnv
        splatsim_render_mode = self.gym_kwargs.get("render_mode", "rgb_array")

        # ---- Teleop recording (shared by ZMQ + local-spawn branches) ---- #
        teleop_context = None
        teleop_dataset = None
        image_keys = None
        if self.teleop_dataset_repo_id is not None:
            from splatsim.configs.mode_config import ImageResizeMode
            from splatsim.utils.lerobot_utils import create_lerobot_dataset, load_lerobot_dataset

            from lerobot.policies.teleop_recording import TeleopRecordingContext

            teleop_context = TeleopRecordingContext.get_instance()
            image_keys = [f"{cam}_{mode.value}" for cam in self.camera_names for mode in ImageResizeMode]
            teleop_dataset = load_lerobot_dataset(self.teleop_dataset_repo_id)
            if teleop_dataset is None:
                teleop_dataset = create_lerobot_dataset(
                    self.teleop_dataset_repo_id, fps=self.fps, image_keys=image_keys
                )

        # Locals captured into _make_splatsim closures below.
        task = self.task
        episode_length = self.episode_length
        teleop_min_episode_length = self.teleop_min_episode_length

        def _wrap_for_recording(env):
            """Apply TeleopRecordingWrapper when teleop recording is configured."""
            if teleop_context is not None and teleop_dataset is not None:
                from lerobot.policies.teleop_recording import TeleopRecordingWrapper

                # image_keys is populated in the same block as teleop_dataset
                # above; assert for the type checker's benefit.
                assert image_keys is not None
                env = TeleopRecordingWrapper(
                    env,
                    context=teleop_context,
                    dataset=teleop_dataset,
                    image_keys=image_keys,
                    task=task,
                    min_episode_length=teleop_min_episode_length,
                )
            return env

        if self.external_port is not None:
            from splatsim.gym_env import ZMQSplatSimGymEnv

            external_host = self.external_host
            external_port = self.external_port
            include_oracle_info = self.include_oracle_info
            camera_names = self.camera_names
            image_resize_modes = self.image_resize_modes
            observation_height = self.observation_height
            observation_width = self.observation_width

            def _make_splatsim():
                env = ZMQSplatSimGymEnv(
                    host=external_host,
                    port=external_port,
                    camera_names=camera_names,
                    image_resize_modes=image_resize_modes,
                    num_dofs=6,
                    image_height=observation_height,
                    image_width=observation_width,
                    render_mode=splatsim_render_mode,
                    max_episode_steps=episode_length,
                    include_oracle_info=include_oracle_info,
                )
                return _wrap_for_recording(env)
        else:
            from splatsim.gym_env import make_single_env
            from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase

            splatsim_cfg = self.gym_kwargs.get("cfg", {})
            splatsim_serve_mode = (
                PybulletRobotServerBase.SERVE_MODES.EVAL_BENCHMARK
                if self.eval_benchmark_repo_id is not None
                else PybulletRobotServerBase.SERVE_MODES.INTERACTIVE
            )

            def _make_splatsim():
                env = make_single_env(
                    task,
                    cfg=splatsim_cfg,
                    render_mode=splatsim_render_mode,
                    serve_mode=splatsim_serve_mode,
                )
                # Local mode honours the lerobot-side episode_length cap. The
                # underlying robot server's _max_episode_steps drives both the
                # gym env's truncation and the rollout loop's max_steps query.
                if hasattr(env, "robot_server") and env.robot_server is not None:
                    env.robot_server._max_episode_steps = episode_length
                if hasattr(env, "_max_episode_steps"):
                    env._max_episode_steps = episode_length
                return _wrap_for_recording(env)

        try:
            from gymnasium.vector import AutoresetMode

            vec = env_cls(
                [_make_splatsim for _ in range(n_envs)],
                # NEXT_STEP: on the termination step, final_info is populated (needed by lerobot_eval
                # to read is_success). The actual auto-reset fires on the *next* step call, which
                # never happens since the rollout loop exits on done=True.
                autoreset_mode=AutoresetMode.NEXT_STEP,
            )
        except ImportError:
            vec = env_cls([_make_splatsim for _ in range(n_envs)])
        return {"splatsim": {0: vec}}


@EnvConfig.register_subclass("isaaclab_arena")
@dataclass
class IsaaclabArenaEnv(HubEnvConfig):
    hub_path: str = "nvidia/isaaclab-arena-envs"
    episode_length: int = 300
    num_envs: int = 1
    embodiment: str | None = "gr1_pink"
    object: str | None = "power_drill"
    mimic: bool = False
    teleop_device: str | None = None
    seed: int | None = 42
    device: str | None = "cuda:0"
    disable_fabric: bool = False
    enable_cameras: bool = False
    headless: bool = False
    enable_pinocchio: bool = True
    environment: str | None = "gr1_microwave"
    task: str | None = "Reach out to the microwave and open it."
    state_dim: int = 54
    action_dim: int = 36
    camera_height: int = 512
    camera_width: int = 512
    video: bool = False
    video_length: int = 100
    video_interval: int = 200
    # Comma-separated keys, e.g., "robot_joint_pos,left_eef_pos"
    state_keys: str = "robot_joint_pos"
    # Comma-separated keys, e.g., "robot_pov_cam_rgb,front_cam_rgb"
    # Set to None or "" for environments without cameras
    camera_keys: str | None = None
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    kwargs: dict | None = None

    def __post_init__(self):
        if self.kwargs:
            # dynamically convert kwargs to fields in the dataclass
            # NOTE! the new fields will not bee seen by the dataclass repr
            field_names = {f.name for f in fields(self)}
            for key, value in self.kwargs.items():
                if key not in field_names and key != "kwargs":
                    setattr(self, key, value)
            self.kwargs = None

        # Set action feature
        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        self.features_map[ACTION] = ACTION

        # Set state feature
        self.features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))
        self.features_map[OBS_STATE] = OBS_STATE

        # Add camera features for each camera key
        if self.enable_cameras and self.camera_keys:
            for cam_key in self.camera_keys.split(","):
                cam_key = cam_key.strip()
                if cam_key:
                    self.features[cam_key] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(self.camera_height, self.camera_width, 3),
                    )
                    self.features_map[cam_key] = f"{OBS_IMAGES}.{cam_key}"

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def get_env_processors(self):
        state_keys = tuple(k.strip() for k in (self.state_keys or "").split(",") if k.strip())
        camera_keys = tuple(k.strip() for k in (self.camera_keys or "").split(",") if k.strip())
        if not state_keys and not camera_keys:
            raise ValueError("At least one of state_keys or camera_keys must be specified.")
        return (
            PolicyProcessorPipeline(
                steps=[IsaaclabArenaProcessorStep(state_keys=state_keys, camera_keys=camera_keys)]
            ),
            PolicyProcessorPipeline(steps=[]),
        )
