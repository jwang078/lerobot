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
from enum import Enum
from typing import Any

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)


class FrameSource(Enum):
    """Origin of an action sample, used by the recorder to decide whether to keep it.

    TELEOP and RRT are "real" frames that count toward ``min_episode_length`` and
    are committed to the dataset. POLICY frames are skipped (not recorded). PADDING
    is reserved for the post-teleop pad-to-min behaviour and is committed only to
    reach the threshold.
    """

    TELEOP = "teleop"
    RRT = "rrt"
    POLICY = "policy"
    PADDING = "padding"


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
        # Set by the policy wrapper each step. The recorder uses this to decide
        # whether to record (TELEOP/RRT) and whether the frame counts toward
        # min_episode_length. POLICY frames are skipped entirely.
        self.frame_source: FrameSource = FrameSource.POLICY
        # Index of the eval-benchmark scenario the current episode came from.
        # Pushed by the intervention controller on each scenario reset; the
        # wrapper reads it in _finish_episode and stores it as a per-episode
        # column in the dataset's episodes parquet (via save_episode's
        # episode_metadata kwarg). None when not running under a controller
        # that knows the scenario index (e.g. interactive teleop).
        self.source_scenario_idx: int | None = None
        # When True, the recording wrapper accumulates finished episodes in
        # an in-memory pending list instead of writing them to the dataset.
        # The caller then invokes ``commit_pending_episodes()`` (save them)
        # or ``discard_pending_episodes()`` (drop them) once it has the
        # information needed to decide. False = legacy immediate-save
        # behavior, used by interactive teleop where there is no upstream
        # decision to defer to.
        self.defer_episode_saves: bool = False

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
        # Most recently committed frame; reused as the pad value for short
        # episodes that ended (e.g. successful goal reached) before reaching
        # min_episode_length, since no further env steps are available to
        # source live padding frames from.
        self._last_committed_frame: dict | None = None
        self._context.min_episode_length = min_episode_length

        # When ``context.defer_episode_saves`` is True, finished episodes go
        # here instead of being written to the dataset. The first list holds
        # frames for the episode currently being built; the second is
        # finalized episodes waiting on a commit/discard call. Both are
        # ignored in immediate-save mode.
        self._in_progress_episode_frames: list[dict] = []
        self._pending_episodes: list[tuple[list[dict], dict | None]] = []

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

    def _emit_frame(self, frame: dict) -> None:
        """Send a frame to the dataset (immediate) or the in-progress
        deferred buffer (deferred mode). All paths that previously called
        ``self._dataset.add_frame`` should go through this so the immediate
        and deferred modes share one code path.
        """
        if self._context.defer_episode_saves:
            # Snapshot — dataset.add_frame mutates the dict by popping "task",
            # so we match the same contract on the eventual replay.
            self._in_progress_episode_frames.append(dict(frame))
        else:
            self._dataset.add_frame(frame)

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
        # Snapshot the last surviving frame BEFORE add_frame mutates it
        # (LeRobotDataset.add_frame pops "task" for separate handling, so
        # post-call the dict is missing fields required by validate_frame).
        # We need a clean copy in case _finish_episode pads short episodes
        # by re-adding this frame.
        last_frame_snapshot = dict(trimmed[-1][0]) if trimmed else None
        for frame, _ in trimmed:
            self._emit_frame(frame)
        n_trimmed = len(self._frame_buffer) - len(trimmed)
        if n_trimmed > 0:
            logger.info(
                f"[TeleopRecording] Trimmed {n_trimmed} frames ({len(self._frame_buffer)} -> {len(trimmed)})"
            )
        self._committed_frame_count += len(trimmed)
        if last_frame_snapshot is not None:
            self._last_committed_frame = last_frame_snapshot
        self._frame_buffer.clear()

    # -- episode lifecycle --------------------------------------------------

    def _finish_episode(self) -> None:
        """Flush any buffered frames, then save or discard the episode."""
        if not self._recording:
            return
        if self._frame_buffer:
            self._flush_buffer()
        # Pad short episodes by repeating the last committed frame. Useful when
        # the env reaches its success condition before min_episode_length —
        # there are no more env steps to source live padding frames from, so we
        # duplicate the final observation/action so the trajectory is still
        # long enough to keep instead of being discarded.
        if (
            self._committed_frame_count > 0
            and self._committed_frame_count < self._min_episode_length
            and self._last_committed_frame is not None
        ):
            n_pad = self._min_episode_length - self._committed_frame_count
            for _ in range(n_pad):
                self._emit_frame(dict(self._last_committed_frame))
            self._committed_frame_count += n_pad
            logger.info(
                f"[TeleopRecording] Padded {n_pad} repeated frame(s) of the final "
                f"observation to reach min_episode_length ({self._min_episode_length})."
            )
        if self._committed_frame_count >= self._min_episode_length:
            scenario_idx = self._context.source_scenario_idx
            # Always include the source_scenario_idx key (None when unset) so
            # the per-episode metadata schema is uniform across runs. Mixing
            # rows that have the key with rows that don't crashes
            # _flush_metadata_buffer (pa.Table.from_pydict requires equal
            # column lengths) and silently strands the buffered rows.
            episode_metadata: dict | None = {
                "source_scenario_idx": int(scenario_idx) if scenario_idx is not None else None
            }
            if self._context.defer_episode_saves:
                # Move the in-progress frames into the pending list and
                # leave the dataset's writer untouched. The caller will
                # decide later whether to commit or drop them.
                self._pending_episodes.append((self._in_progress_episode_frames, episode_metadata))
                self._in_progress_episode_frames = []
                logger.info(
                    f"[TeleopRecording] Buffered episode for deferred save "
                    f"({self._committed_frame_count} frames, "
                    f"{len(self._pending_episodes)} pending"
                    + (f", source_scenario_idx={scenario_idx}" if scenario_idx is not None else "")
                    + ")"
                )
            else:
                self._dataset.save_episode(episode_metadata=episode_metadata)
                logger.info(
                    f"[TeleopRecording] Saved episode ({self._committed_frame_count} frames, "
                    f"total episodes: {self._dataset.meta.total_episodes}"
                    + (f", source_scenario_idx={scenario_idx}" if scenario_idx is not None else "")
                    + ")"
                )
        else:
            if self._context.defer_episode_saves:
                # Drop the in-progress deferred buffer; nothing was sent to
                # the dataset's writer in this mode.
                self._in_progress_episode_frames = []
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
        self._last_committed_frame = None
        self._sync_context()

    def _discard_episode(self) -> None:
        """Discard the current recording segment (triggered by GUI button)."""
        if not self._recording:
            return
        self._frame_buffer.clear()
        self._last_committed_frame = None
        if self._context.defer_episode_saves:
            self._in_progress_episode_frames = []
        else:
            self._dataset.clear_episode_buffer()
        logger.info(f"[TeleopRecording] Manually discarded episode ({self._episode_frame_count} frames)")
        self._recording = False
        self._padding = False
        self._episode_frame_count = 0
        self._committed_frame_count = 0
        self._sync_context()

    # -- deferred-save commit/discard ---------------------------------------

    def flush_in_progress_episode(self) -> bool:
        """Force-finalize any in-progress recording (frames in ``_frame_buffer``
        from a recording stream the env never transitioned out of).

        Normally ``_finish_episode`` is triggered when the next ``step()`` sees
        ``frame_source`` flip back to POLICY. If the caller bails out of the
        env loop while still in TELEOP / RRT (e.g. env declared success
        mid-RRT-execution), no further step happens and the recording sits
        stranded in ``_frame_buffer``. Calling this drains it through the
        normal episode-finish path so the resulting episode either lands in
        ``_pending_episodes`` (deferred mode) or saves immediately
        (immediate mode), and the wrapper's recording state resets.

        Returns True if a finalization occurred, False if there was nothing
        to do.
        """
        if not self._recording:
            return False
        self._finish_episode()
        return True

    def commit_pending_episodes(self) -> int:
        """Replay every pending episode to the dataset (add_frame + save_episode).

        No-op in immediate-save mode (pending list is always empty there).
        Safe to call any time; in-progress recordings are unaffected. Returns
        the number of episodes saved.
        """
        n = 0
        for frames, meta in self._pending_episodes:
            for frame in frames:
                self._dataset.add_frame(frame)
            self._dataset.save_episode(episode_metadata=meta)
            n += 1
        if n > 0:
            logger.info(
                f"[TeleopRecording] Committed {n} pending episode(s); "
                f"total episodes: {self._dataset.meta.total_episodes}"
            )
        self._pending_episodes.clear()
        return n

    def discard_pending_episodes(self) -> int:
        """Drop every pending episode without saving. No-op in immediate-save
        mode. Returns the number of episodes dropped."""
        n = len(self._pending_episodes)
        if n > 0:
            total_frames = sum(len(frames) for frames, _ in self._pending_episodes)
            logger.info(f"[TeleopRecording] Discarded {n} pending episode(s) ({total_frames} frames)")
        self._pending_episodes.clear()
        return n

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

        # A frame is "real" (counts toward min_episode_length, gets saved as a
        # genuine sample) when its source is either pure-teleop or RRT-to-goal.
        # Falling back to ratio==0.0 keeps the legacy contract for callers that
        # haven't been updated to set frame_source.
        source = self._context.frame_source
        is_real_frame = source in (FrameSource.TELEOP, FrameSource.RRT) or (
            source is FrameSource.POLICY and self._context.ratio == 0.0
        )

        if is_real_frame:
            if self._padding:
                # User went back to teleop/RRT while padding — continue same episode.
                self._padding = False
            self._recording = True
            frame, gym_obs, reward, terminated, truncated, info = self._step_raw(action)
            self._buffer_frame(frame)
            self._episode_frame_count = self._committed_frame_count + len(self._frame_buffer)
            self._sync_context()
            return gym_obs, reward, terminated, truncated, info

        # --- Real-frame stream just ended: flush buffer ---
        if self._recording and self._frame_buffer:
            self._flush_buffer()
            self._episode_frame_count = self._committed_frame_count

        if self._recording and self._committed_frame_count < self._min_episode_length:
            self._padding = True

        if self._padding:
            frame, gym_obs, reward, terminated, truncated, info = self._step_raw(action)
            # Snapshot before add_frame, which pops "task" from the dict.
            self._last_committed_frame = dict(frame)
            self._emit_frame(frame)
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
