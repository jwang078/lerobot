"""Helpers for seeding a SplatSim Gym env to a specific dataset frame's state.

The lerobot env factory only exposes a Gym vector env API (`reset`, `step`). For the
visualize / data-relabelling use case we need to start a rollout from an arbitrary
``(episode_index, frame_index)`` state — Splatsim only natively supports resetting
to *episode start* via EVAL_BENCHMARK mode.

``seed_splatsim_env_to_state`` bridges that gap: it calls ``reset()`` so the env
loads the EVAL_BENCHMARK scenario (object poses + episode-start joints), then
reaches into the underlying ``PybulletRobotServerBase`` to teleport the robot to
the requested joint state. Object poses still match episode-start — fine for
tasks where objects are static until the robot makes contact.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


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
    joint_state: np.ndarray,
    num_dofs: int = 6,
) -> dict[str, Any]:
    """Reset ``vec_env`` (loading the configured EVAL_BENCHMARK scenario) and teleport
    the robot to ``joint_state``. Returns a batched gym observation dict matching the
    shape ``vec_env.step()`` produces.

    Args:
        vec_env: A SyncVectorEnv with a single underlying SplatSim env (n_envs=1).
        joint_state: Raw joint configuration to teleport to. Shape ``(num_dofs,)`` to
            set arm joints only, or ``(num_dofs + 1,)`` to also set the gripper.
        num_dofs: Number of arm degrees of freedom. Defaults to 6.

    Notes:
        * Object poses remain at their episode-start state — they are loaded by
          ``reset()`` and not touched by the teleport. Acceptable for tasks where
          objects don't move until the robot makes contact.
        * Reaches into private attributes of ``PybulletRobotServerBase``
          (``teleport_joint_state``, ``splatsim_robot``). The remote ZMQ backend is
          not supported here.
    """
    if not hasattr(vec_env, "envs"):
        raise TypeError(
            "seed_splatsim_env_to_state requires a SyncVectorEnv (so we can reach "
            f"the underlying single env). Got {type(vec_env).__name__}."
        )
    if len(vec_env.envs) != 1:
        raise ValueError(f"seed_splatsim_env_to_state only supports n_envs=1 (got {len(vec_env.envs)}).")

    # First reset loads the EVAL_BENCHMARK scenario (object poses + episode-start joints).
    print("[seed] vec_env.reset() …", flush=True)
    vec_env.reset()
    print("[seed] vec_env.reset() done", flush=True)

    single_env = vec_env.envs[0]
    # SplatSimGymEnv.unwrapped overrides gym.Env.unwrapped to return the robot_server.
    robot_server = getattr(single_env, "robot_server", None) or single_env.unwrapped

    if not hasattr(robot_server, "teleport_joint_state"):
        raise RuntimeError(
            "Underlying env does not expose `teleport_joint_state` "
            "(remote ZMQ backend or unsupported env type)."
        )
    if robot_server.splatsim_robot is None:
        raise RuntimeError("robot_server.splatsim_robot is None — env not fully initialized.")

    js = np.asarray(joint_state, dtype=np.float64).reshape(-1)
    n_set = min(js.shape[0], num_dofs + 1)
    print(f"[seed] teleport_joint_state(joints={js[:n_set].tolist()}) …", flush=True)
    robot_server.teleport_joint_state(robot_server.splatsim_robot, js[:n_set].tolist())
    print("[seed] teleport done; get_observations() …", flush=True)

    raw_obs = robot_server.get_observations()
    print(f"[seed] get_observations() done; keys={list(raw_obs.keys())[:10]}", flush=True)
    gym_obs = single_env._to_gym_obs(raw_obs) if hasattr(single_env, "_to_gym_obs") else raw_obs
    print("[seed] _to_gym_obs done", flush=True)
    return _add_batch_dim(gym_obs)
