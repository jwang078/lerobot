"""Teleop episode recording during shared autonomy evaluation.

When the shared autonomy slider is at ratio=0 (pure teleop), this module
captures observation/action frames and saves them as episodes in a LeRobot
dataset.  Short segments (below a configurable threshold) are padded with
policy frames.

Teleop frames are buffered in memory and trimmed (leading/trailing no-ops
removed) before being committed to the dataset.  This avoids saving frames
where the user hasn't started pressing keys yet or has released keys before
switching out of teleop mode.

Architecture
------------
``TeleopRecordingContext`` is a singleton that bridges the policy wrapper
(which writes the current ratio) and the ``TeleopRecordingWrapper`` gym
wrapper (which reads the ratio to decide whether to record).

``TeleopRecordingWrapper`` is a standard ``gym.Wrapper`` applied to each
individual env *before* vectorisation.  It intercepts ``step`` / ``reset``
/ ``close`` and manages episode boundaries.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared context singleton
# ---------------------------------------------------------------------------


class TeleopRecordingContext:
    """Shared state between SharedAutonomyPolicyWrapper and TeleopRecordingWrapper.

    The policy wrapper writes ``ratio`` and ``has_guidance`` on every
    ``select_action`` call; the recording wrapper reads them on every ``step``.
    """

    _instance: TeleopRecordingContext | None = None

    def __init__(self) -> None:
        self.ratio: float = 1.0  # current forward_flow_ratio
        self.has_guidance: bool = False  # True when user is actively pressing keys
        self.recording: bool = False
        self.episode_frame_count: int = 0
        self.min_episode_length: int = 60
        self.total_saved_episodes: int = 0
        self.padding: bool = False
        self.discard_requested: bool = False

    @classmethod
    def get_instance(cls) -> TeleopRecordingContext:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None


# ---------------------------------------------------------------------------
# Gymnasium recording wrapper
# ---------------------------------------------------------------------------


class TeleopRecordingWrapper(gym.Wrapper):
    """Records pure-teleop (ratio == 0) segments to a LeRobot dataset.

    Wraps an individual (non-vectorised) SplatSim gym env.  Each contiguous
    run of ratio == 0 steps becomes a separate episode.  Segments shorter
    than ``min_episode_length`` are padded with policy frames.

    Teleop frames are buffered in memory and trimmed (leading/trailing
    no-guidance frames removed) before being committed to the dataset.

    When recording, calls ``robot_server.step()`` directly to obtain the
    raw observation dict (which has ``{cam}_{mode}`` image keys for all
    resize modes).  This avoids a second ZMQ round-trip.
    """

    def __init__(
        self,
        env: gym.Env,
        context: TeleopRecordingContext,
        dataset: Any,  # LeRobotDataset — imported lazily to avoid circular deps
        image_keys: list[str],
        task: str,
        min_episode_length: int = 60,
    ) -> None:
        super().__init__(env)
        self._context = context
        self._dataset = dataset
        self._image_keys = image_keys
        self._task = task
        self._min_episode_length = min_episode_length

        self._recording: bool = False
        self._padding: bool = False
        self._episode_frame_count: int = 0
        self._committed_frame_count: int = 0  # frames actually in the dataset
        self._frame_buffer: list[tuple[dict, bool]] = []  # (frame, has_guidance)
        self._context.min_episode_length = min_episode_length

        # Pre-import hf_xet so it's in sys.modules before any KeyboardInterrupt
        # can corrupt Python's import machinery (causes push_to_hub to fail).
        with contextlib.suppress(ImportError):
            import hf_xet  # noqa: F401

    # -- internal helpers ---------------------------------------------------

    def _build_frame(self, action: np.ndarray, gym_obs: dict, raw_obs: dict) -> dict:
        """Build a LeRobot frame dict from gym obs (state) and raw obs (images).

        Uses ``gym_obs["agent_pos"]`` for observation.state and ``raw_obs``
        for images (which are in CHW float32 [0,1] format with
        ``{cam}_{mode}`` keys).
        """
        frame: dict[str, Any] = {
            "observation.state": gym_obs["agent_pos"].astype(np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": self._task,
        }
        for key in self._image_keys:
            img = raw_obs.get(key)
            if img is None:
                raise RuntimeError(
                    f"Image key '{key}' missing from server observations. "
                    f"Launch with --env.image_resize_modes listing all modes "
                    f'from ImageResizeMode enum (e.g. \'["stretch", "letterbox"]\').'
                )
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            frame[f"observation.images.{key}"] = img.astype(np.float32)
        return frame

    def _step_raw(self, action: np.ndarray):
        """Step via robot_server, build frame, return (frame, gym_obs, rew, term, trunc, info)."""
        raw_obs, reward, terminated, truncated, info = self.env.robot_server.step(action)
        gym_obs = self.env._to_gym_obs(raw_obs)
        frame = self._build_frame(action, gym_obs, raw_obs)
        return frame, gym_obs, reward, terminated, truncated, info

    # -- buffer / trim / flush ----------------------------------------------

    def _buffer_frame(self, frame: dict) -> None:
        """Buffer a teleop frame for later trimming and flushing."""
        self._frame_buffer.append((frame, self._context.has_guidance))

    def _trim_frames(self, frames: list[tuple[dict, bool]]) -> list[tuple[dict, bool]]:
        """Trim leading and trailing no-guidance frames from the buffer."""
        if not frames:
            return frames

        # 1. Trim trailing no-guidance frames
        end = len(frames)
        while end > 0 and not frames[end - 1][1]:
            end -= 1
        if end == 0:
            return []

        # 2. Trim leading no-guidance frames
        start = 0
        while start < end and not frames[start][1]:
            start += 1
        if start >= end:
            return []

        return frames[start:end]

    def _flush_buffer(self) -> None:
        """Trim the teleop buffer and commit survivors to the dataset."""
        if not self._frame_buffer:
            return
        trimmed = self._trim_frames(self._frame_buffer)
        for frame, _ in trimmed:
            self._dataset.add_frame(frame)
        n_trimmed = len(self._frame_buffer) - len(trimmed)
        if n_trimmed > 0:
            logger.info(
                f"[TeleopRecording] Trimmed {n_trimmed} frames ({len(self._frame_buffer)} -> {len(trimmed)})"
            )
        self._committed_frame_count += len(trimmed)
        self._frame_buffer.clear()

    # -- episode lifecycle --------------------------------------------------

    def _finish_episode(self) -> None:
        """Flush any buffered frames, then save or discard the episode."""
        if not self._recording:
            return
        if self._frame_buffer:
            self._flush_buffer()
        if self._committed_frame_count >= self._min_episode_length:
            self._dataset.save_episode()
            logger.info(
                f"[TeleopRecording] Saved episode ({self._committed_frame_count} frames, "
                f"total episodes: {self._dataset.meta.total_episodes})"
            )
        else:
            self._dataset.clear_episode_buffer()
            logger.info(
                f"[TeleopRecording] Discarded too-short episode "
                f"({self._committed_frame_count} < {self._min_episode_length} after trimming)"
            )
        self._recording = False
        self._padding = False
        self._episode_frame_count = 0
        self._committed_frame_count = 0
        self._frame_buffer.clear()
        self._sync_context()

    def _discard_episode(self) -> None:
        """Discard the current recording segment (triggered by GUI button)."""
        if not self._recording:
            return
        self._frame_buffer.clear()
        self._dataset.clear_episode_buffer()
        logger.info(f"[TeleopRecording] Manually discarded episode ({self._episode_frame_count} frames)")
        self._recording = False
        self._padding = False
        self._episode_frame_count = 0
        self._committed_frame_count = 0
        self._sync_context()

    def _sync_context(self) -> None:
        """Push local recording state to the shared context for GUI display."""
        self._context.recording = self._recording
        self._context.padding = self._padding
        self._context.episode_frame_count = self._episode_frame_count
        self._context.total_saved_episodes = self._dataset.meta.total_episodes

    # -- gym.Wrapper overrides ----------------------------------------------

    def step(self, action: np.ndarray):
        if self._context.discard_requested:
            self._discard_episode()
            self._context.discard_requested = False

        is_teleop = self._context.ratio == 0.0

        if is_teleop:
            if self._padding:
                # User went back to teleop while padding — continue the same episode.
                self._padding = False
            self._recording = True
            frame, gym_obs, reward, terminated, truncated, info = self._step_raw(action)
            self._buffer_frame(frame)
            self._episode_frame_count = self._committed_frame_count + len(self._frame_buffer)
            self._sync_context()
            return gym_obs, reward, terminated, truncated, info

        # --- Teleop just ended: flush buffer ---
        if self._recording and self._frame_buffer:
            self._flush_buffer()
            self._episode_frame_count = self._committed_frame_count

        if self._recording and self._committed_frame_count < self._min_episode_length:
            self._padding = True

        if self._padding:
            frame, gym_obs, reward, terminated, truncated, info = self._step_raw(action)
            self._dataset.add_frame(frame)
            self._committed_frame_count += 1
            self._episode_frame_count = self._committed_frame_count
            self._sync_context()
            if self._committed_frame_count >= self._min_episode_length:
                self._finish_episode()
            return gym_obs, reward, terminated, truncated, info

        self._finish_episode()
        return self.env.step(action)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._finish_episode()
        return self.env.reset(seed=seed, options=options)

    def close(self) -> None:
        from splatsim.utils.lerobot_utils import finalize_lerobot_dataset

        try:
            self._finish_episode()
        except Exception:
            logger.exception("[TeleopRecording] Error finishing episode during close")
        try:
            finalize_lerobot_dataset(self._dataset)
            logger.info("[TeleopRecording] Dataset finalised.")
            logger.info(f"[TeleopRecording] Pushing dataset to hub as '{self._dataset.repo_id}'...")
            self._dataset.push_to_hub()
            logger.info("[TeleopRecording] Successfully pushed to hub.")
        except Exception:
            # After KeyboardInterrupt, the import system is often corrupted.
            # Fall back to a subprocess with a clean Python interpreter.
            import subprocess
            import sys

            root = str(self._dataset.root)
            repo_id = self._dataset.repo_id
            print("[TeleopRecording] In-process push failed, retrying in subprocess...")
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from huggingface_hub import HfApi; "
                        f"HfApi().upload_folder(folder_path='{root}', "
                        f"repo_id='{repo_id}', repo_type='dataset', "
                        "ignore_patterns=['images/'])",
                    ],
                    timeout=300,
                )
                if result.returncode == 0:
                    print("[TeleopRecording] Successfully pushed to hub (via subprocess).")
                else:
                    print(f"[TeleopRecording] Subprocess push failed (exit code {result.returncode}).")
                    print(f"Dataset saved locally at: {root}")
            except Exception as e2:
                print(f"[TeleopRecording] Subprocess push also failed: {e2}")
                print(f"Dataset saved locally at: {root}")
        super().close()
