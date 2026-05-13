"""Helpers for seeding a SplatSim Gym env to a specific dataset frame's state.

The lerobot env factory only exposes a Gym vector env API (`reset`, `step`). For the
visualize / data-relabelling use case we need to start a rollout from an arbitrary
``(episode_index, frame_index)`` state.

``seed_splatsim_env_to_state`` bridges that gap: it calls ``reset(seed=...)`` so the
env loads the right benchmark scenario, then — for local (in-process pybullet) envs —
reaches into the underlying ``PybulletRobotServerBase`` to teleport the robot to the
requested joint state. For ZMQ (out-of-process) envs the teleport is skipped; the
robot starts at the episode's initial pose as loaded by the server.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


def _add_batch_dim(value: Any) -> Any:
    """Add a leading batch dim of size 1 to numpy arrays in a (possibly nested) dict."""
    if isinstance(value, dict):
        return {k: _add_batch_dim(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value[np.newaxis, ...]
    return value


def seed_splatsim_env_to_state(
    vec_env: gym.vector.VectorEnv,
    *,
    joint_state: np.ndarray | None = None,
    num_dofs: int = 6,
    seed: list[int] | None = None,
) -> dict[str, Any]:
    """Reset ``vec_env`` and optionally teleport the robot to ``joint_state``.

    For **local** (in-process pybullet) envs the teleport fires after the reset,
    placing the robot at the exact joint configuration requested. For **ZMQ**
    (out-of-process) envs the teleport is silently skipped — the robot stays at
    whatever initial pose the server loaded for this scenario.

    Args:
        vec_env: A SyncVectorEnv with n_envs=1 wrapping a SplatSim env.
        joint_state: Raw joint configuration to teleport to. Shape ``(num_dofs,)``
            or ``(num_dofs + 1,)`` (with gripper). Ignored for ZMQ envs.
        num_dofs: Number of arm degrees of freedom. Defaults to 6.
        seed: Forwarded to ``vec_env.reset(seed=seed)``. For ZMQ envs the seed
            selects the benchmark scenario on the server side (e.g.
            ``seed=[episode_index]``).

    Returns:
        Batched gym observation dict matching the shape ``vec_env.step()`` produces.
    """
    if not hasattr(vec_env, "envs"):
        raise TypeError(f"seed_splatsim_env_to_state requires a SyncVectorEnv; got {type(vec_env).__name__}.")
    if len(vec_env.envs) != 1:
        raise ValueError(f"seed_splatsim_env_to_state only supports n_envs=1 (got {len(vec_env.envs)}).")

    # Reset loads the benchmark scenario (object poses + episode-start robot joints).
    env_obs, _info = vec_env.reset(seed=seed)

    # Attempt local joint teleport. Only works with the in-process pybullet backend.
    if joint_state is not None:
        single_env = vec_env.envs[0]
        robot_server = getattr(single_env, "robot_server", None) or single_env.unwrapped

        if hasattr(robot_server, "teleport_joint_state") and hasattr(robot_server, "splatsim_robot"):
            js = np.asarray(joint_state, dtype=np.float64).reshape(-1)
            n_set = min(js.shape[0], num_dofs + 1)
            robot_server.teleport_joint_state(robot_server.splatsim_robot, js[:n_set].tolist())
            raw_obs = robot_server.get_observations()
            if hasattr(single_env, "_to_gym_obs"):
                env_obs = _add_batch_dim(single_env._to_gym_obs(raw_obs))
        else:
            logger.debug(
                "seed_splatsim_env_to_state: teleport_joint_state not available "
                "(ZMQ backend?). Robot starts at episode-initial pose."
            )

    return env_obs
