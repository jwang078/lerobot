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
import builtins
import datetime as dt
import json
import multiprocessing
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs.accelerator import AcceleratorConfig, ActivationCheckpointingMode
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.hub import HubMixin, find_latest_hub_checkpoint
from lerobot.utils.sample_weighting import SampleWeightingConfig

from . import parser
from .default import DatasetConfig, EMAConfig, EvalConfig, JobConfig, PeftConfig, WandBConfig
from .policies import PreTrainedConfig
from .rewards import RewardModelConfig

TRAIN_CONFIG_NAME = "train_config.json"


class CheckpointFormat(str, Enum):
    """Model-artifact format inside training checkpoints.

    Selects only the *model* artifact; the training_state layout is format-independent (the
    optimizer channel is always DCP under sharded runs, safetensors+json otherwise).

    - SAFETENSORS (default): a full `model.safetensors` — maximum compatibility, one gather per
      save under sharding.
    - DCP: sharded `pytorch_model_fsdp_0/*.distcp` only — fastest save/resume; convert with
      `lerobot-convert-dcp` before distributing.
    - SAFETENSORS_AND_DCP: both artifacts, written independently.
    """

    SAFETENSORS = "safetensors"
    DCP = "dcp"
    SAFETENSORS_AND_DCP = "safetensors_dcp"

    @property
    def wants_safetensors(self) -> bool:
        """True when a full `model.safetensors` artifact should be written."""
        return self in (CheckpointFormat.SAFETENSORS, CheckpointFormat.SAFETENSORS_AND_DCP)

    @property
    def wants_dcp(self) -> bool:
        """True when sharded DCP model shards (`pytorch_model_fsdp_0/`) should be written."""
        return self in (CheckpointFormat.DCP, CheckpointFormat.SAFETENSORS_AND_DCP)
def _migrate_legacy_renamed_fields(config: dict[str, Any]) -> dict[str, Any] | None:
    """Rename fields that were renamed in the TrainPipelineConfig schema after
    old checkpoints were written. Applied on load so `--resume` from a
    pre-rename checkpoint still parses under the current class.

    Returns a mutated copy of the config, or None when no legacy field is
    present (fast-path skip).

    Add entries here whenever an upstream rebase renames a
    TrainPipelineConfig field — the alternative is asking every user to
    hand-edit their saved train_config.json before resuming.
    """
    renames = {
        # 2026-07 upstream rename: eval_freq → env_eval_freq (split off from
        # dataset-eval, which now uses --eval_steps).
        "eval_freq": "env_eval_freq",
    }
    if not any(old in config for old in renames):
        return None
    migrated = dict(config)
    for old, new in renames.items():
        if old in migrated:
            # If BOTH old and new are present (weird), prefer new — the config
            # was already partially migrated somehow; don't clobber the newer
            # write with the stale field.
            value = migrated.pop(old)
            migrated.setdefault(new, value)
    return migrated


def _migrate_legacy_rabc_fields(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return migrated payload for legacy RA-BC fields, or None when no migration is needed."""
    legacy_fields = (
        "use_rabc",
        "rabc_progress_path",
        "rabc_kappa",
        "rabc_epsilon",
        "rabc_head_mode",
    )
    if not any(key in config for key in legacy_fields):
        return None

    migrated_config = dict(config)
    use_rabc = bool(migrated_config.pop("use_rabc", False))
    rabc_progress_path = migrated_config.pop("rabc_progress_path", None)
    rabc_kappa = migrated_config.pop("rabc_kappa", None)
    rabc_epsilon = migrated_config.pop("rabc_epsilon", None)
    rabc_head_mode = migrated_config.pop("rabc_head_mode", None)

    # New configs may already define sample_weighting explicitly. In that case,
    # legacy fields are ignored after being stripped from the payload.
    if migrated_config.get("sample_weighting") is None and use_rabc:
        sample_weighting: dict[str, Any] = {"type": "rabc"}
        if rabc_progress_path is not None:
            sample_weighting["progress_path"] = rabc_progress_path
        if rabc_kappa is not None:
            sample_weighting["kappa"] = rabc_kappa
        if rabc_epsilon is not None:
            sample_weighting["epsilon"] = rabc_epsilon
        if rabc_head_mode is not None:
            sample_weighting["head_mode"] = rabc_head_mode
        migrated_config["sample_weighting"] = sample_weighting

    return migrated_config


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    reward_model: RewardModelConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. Pass `--config_path` pointing at either a local
    # checkpoint's train_config.json or a Hub repo id holding `checkpoints/<step>/` subtrees (the
    # latest checkpoint is downloaded and resumed from). Note that when resuming, the default behavior
    # is to use the configuration from the checkpoint, regardless of what's provided with the training
    # command at the time of resumption (CLI `--*` flags still override).
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Set to True to use deterministic cuDNN algorithms for reproducibility.
    # This disables cudnn.benchmark and may reduce training speed by ~10-20 percent.
    cudnn_deterministic: bool = False
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    # DataLoader worker start method. "spawn" is safer than "fork" with
    # non-fork-safe libs (PyAV / torchcodec / ffmpeg), but adds some
    # worker-startup time per run since workers re-import modules instead
    # of inheriting parent state. Override with `--dataloader_multiprocessing_context=fork`
    # when appropriate, or set it to `null` to use Python's platform default.
    dataloader_multiprocessing_context: str | None = "spawn"
    steps: int = 100_000
    # Run policy in the simulation environment every N steps to measure reward/success (0 = disabled).
    env_eval_freq: int = 20_000
    log_freq: int = 200
    # Compute eval loss on held-out episodes every N steps (0 = disabled). Requires eval_split > 0.
    eval_steps: int = 0
    # Cap on total eval samples, split uniformly across tasks (0 = use all held-out data).
    max_eval_samples: int = 0
    # Compute policy loss on the FULL held-out eval-benchmark dataset (the same
    # one env-eval rolls out on, via cfg.env.eval_benchmark_repo_id) every N
    # steps. 0 = disabled. Distinct from `eval_steps` which uses a held-out
    # SLICE of the training dataset via `dataset.eval_split`. This runs on an
    # ENTIRELY UNSEEN dataset, so its loss vs train/loss is a direct
    # overfitting diagnostic (typical ratio 1-3× for well-trained models;
    # 10-30× indicates severe overfitting). Logged to wandb as
    # `eval_benchmark/loss`. NOTE: requires env.eval_benchmark_repo_id to be
    # set — silently no-ops otherwise, so it's safe to enable unconditionally
    # in shared training configs.
    eval_benchmark_loss_freq: int = 0
    # Max number of batches to iterate for eval_benchmark_loss (0 = use the
    # whole benchmark). Set >0 to bound the periodic cost when the benchmark
    # is large — 32 batches × batch_size 32 ≈ 1000 frames is usually enough
    # to get a stable mean loss estimate (±5% relative error).
    eval_benchmark_loss_max_batches: int = 0
    tolerance_s: float = 1e-4
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    # A non-positive value disables periodic saving, keeping only the final checkpoint.
    save_freq: int = 20_000
    # Model-artifact format inside checkpoints; non-default values require a sharded run.
    checkpoint_format: CheckpointFormat = CheckpointFormat.SAFETENSORS
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    # Process topology: dp_replicate / dp_shard (HSDP) and context-parallel degree placeholders.
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    # Execution runtime handed to the Accelerator: mixed precision, gradient accumulation,
    # FSDP/DDP tuning knobs, compile & activation-checkpointing placeholders.
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # Maintain an EMA shadow of the policy weights during training (see EMAConfig).
    ema: EMAConfig = field(default_factory=EMAConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # Where to run training (local default, or an HF Jobs flavor). See JobConfig.
    job: JobConfig = field(default_factory=JobConfig)
    # Push each saved checkpoint to the Hub (policy.repo_id) as it is written, not
    # just the final model (useful to monitor progress mid-run). Optional; the
    # final model is pushed regardless. Works the same locally and remotely.
    save_checkpoint_to_hub: bool = False

    # Sample weighting configuration (e.g., for RA-BC training)
    sample_weighting: SampleWeightingConfig | None = None

    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    # Slice a subset of last-axis dims out of specific observation feature
    # tensors before they reach the policy. Use case: an unused/constant
    # observation dim (e.g., the always-0 gripper joint in the planar_3joint
    # env) has min == max in the dataset stats, which poisons MIN_MAX
    # normalization AND wastes an input slot on a dead feature. Dropping the
    # dim here avoids both without needing to re-record the dataset.
    #
    # Format: {observation_key: [dim_indices_to_keep_on_last_axis]}. Empty
    # dict = disabled. The slice is:
    #   * applied by SelectObservationDimsProcessorStep (inserted as the
    #     first preprocessor step in factory.py's post-hoc block),
    #   * baked into `policy.input_features[key].shape` at validate() time
    #     so the model input dim matches the sliced tensor.
    # Example CLI:
    #   --observation_dim_slice='{"observation.state": [0, 1, 2]}'
    observation_dim_slice: dict[str, list[int]] = field(default_factory=dict)
    checkpoint_path: Path | None = field(init=False, default=None)

    @property
    def is_reward_model_training(self) -> bool:
        """True when the config targets a reward model rather than a policy."""
        return self.reward_model is not None

    @property
    def trainable_config(self) -> PreTrainedConfig | RewardModelConfig:
        """Return whichever config (policy or reward_model) is active."""
        if self.is_reward_model_training:
            return self.reward_model  # type: ignore[return-value]
        return self.policy  # type: ignore[return-value]

    def _resolve_pretrained_from_cli(self) -> None:
        """Resolve the pretrained source passed on the CLI into a loaded config.

        The pretrained paths (`--policy.path`, `--reward_model.path`) and
        `--config_path` are only recoverable by re-reading the CLI args: draccus
        has already consumed them by the time `validate()` runs, so they are not
        reflected on `self`. Exactly one source applies, in priority order:
        reward-model path, policy path, then resume.
        """
        reward_model_path = parser.get_path_arg("reward_model")
        policy_path = parser.get_path_arg("policy")

        if reward_model_path:
            cli_overrides = parser.get_cli_overrides("reward_model")
            self.reward_model = RewardModelConfig.from_pretrained(
                reward_model_path, cli_overrides=cli_overrides
            )
            self.reward_model.pretrained_path = str(Path(reward_model_path))
        elif policy_path:
            overrides = parser.get_yaml_overrides("policy") + (parser.get_cli_overrides("policy") or [])
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=overrides)
            self.policy.pretrained_path = Path(policy_path)
        elif self.resume:
            self._resolve_resume_checkpoint()

    def _resolve_resume_checkpoint(self) -> None:
        """Point the trainable config at the checkpoint named by `--config_path`.

        `config_path` is either a local path (to a checkpoint's train_config.json or its
        pretrained_model/ dir) or a Hub repo id. For a Hub repo, the latest checkpoint is downloaded
        into a fresh local run dir and resumed from there. The download is skipped when dispatching to
        an HF Job (`job.is_remote`): the pod performs it when it runs the resume locally, and
        `submit_to_hf` resolves the source repo for the remote command.
        """
        config_path = parser.parse_arg("config_path")
        if not config_path:
            raise ValueError(
                f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
            )

        if Path(config_path).resolve().exists():
            # `config_path` may point at the checkpoint's train_config.json or at its
            # pretrained_model/ directory (both documented above) — resolve either to
            # the pretrained_model/ directory.
            config_path_obj = Path(config_path)
            policy_dir = config_path_obj.parent if config_path_obj.is_file() else config_path_obj
            self.checkpoint_path = policy_dir.parent
        elif self.job.is_remote:
            return
        else:
            from lerobot.common.train_utils import resolve_resume_checkpoint

            # `self.output_dir` was loaded from the checkpoint's config and points at the original
            # run's (now-absent) local dir. Resume into a fresh local dir instead, unless the user
            # passed --output_dir explicitly.
            cli_output_dir = parser.parse_arg("output_dir")
            if cli_output_dir:
                self.output_dir = Path(cli_output_dir)
            else:
                now = dt.datetime.now()
                self.output_dir = Path("outputs/train") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_resume"
            self.checkpoint_path = resolve_resume_checkpoint(config_path, self.output_dir)
            policy_dir = self.checkpoint_path / PRETRAINED_MODEL_DIR

        if self.policy is not None:
            self.policy.pretrained_path = policy_dir
        if self.reward_model is not None:
            self.reward_model.pretrained_path = str(policy_dir)

    def validate(self) -> None:
        available_contexts = multiprocessing.get_all_start_methods()
        if (
            self.dataloader_multiprocessing_context is not None
            and self.dataloader_multiprocessing_context not in available_contexts
        ):
            raise ValueError(
                "`dataloader_multiprocessing_context` must be None or one of "
                f"{available_contexts} on this platform, got "
                f"{self.dataloader_multiprocessing_context!r}."
            )

        self._resolve_pretrained_from_cli()

        if self.policy is None and self.reward_model is None:
            raise ValueError(
                "Neither policy nor reward_model is configured. "
                "Please specify one with `--policy.path` or `--reward_model.path`."
            )

        active_cfg = self.trainable_config
        # Note: upstream also errors here if `rename_map` is set without a
        # pretrained_path, on the assumption that fresh-init policies derive
        # feature names from the dataset. That assumption breaks when the
        # caller manually overrides `policy.input_features` on fresh init —
        # rename_map then bridges the dataset's raw keys to those overridden
        # names, and is genuinely load-bearing. train_sweep.sh does this. The
        # validation is dropped here rather than gated on
        # `input_features`-being-set because we don't have that field's
        # provenance handy at this point (draccus doesn't distinguish
        # "defaulted" from "explicitly set") and rename_map is inert when the
        # dataset already matches the policy schema.

        # ─── observation_dim_slice → policy.input_features shape ──────────────
        # Auto-shrink the declared shape of each sliced observation feature so
        # the model's input dim matches what SelectObservationDimsProcessorStep
        # will actually emit at runtime (see the field docstring above). Runs
        # BEFORE the multi-dataset validation block so any downstream shape
        # checks see the post-slice shape. Only touches keys that ARE in
        # `policy.input_features` — extra keys in the slice dict are silently
        # ignored (they may target features the policy doesn't consume).
        if self.observation_dim_slice and self.policy is not None:
            for key, idx_list in self.observation_dim_slice.items():
                feat = (self.policy.input_features or {}).get(key)
                if feat is None:
                    continue
                old_shape = tuple(feat.shape)
                if not old_shape:
                    continue
                new_shape = (*old_shape[:-1], len(idx_list))
                self.policy.input_features[key] = type(feat)(type=feat.type, shape=new_shape)

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{active_cfg.type}"
            else:
                self.job_name = f"{self.env.type}_{active_cfg.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        # ─── Multi-dataset weighted-sampling validation ───────────────────────
        # Two modes:
        #   (1) Single-dataset (default): `dataset.repo_id` is set, all
        #       multi-dataset fields are None. Byte-identical to legacy behavior.
        #   (2) Multi-dataset: `dataset.repo_ids` is set, `dataset.sample_weights`
        #       and `dataset.stats_paths` are parallel-length lists; `repo_id`
        #       must be empty. Sample weights must sum to ~1.0 (±epsilon).
        # Anything in between is a config bug — fail fast with a clear message.
        if self.dataset.repo_ids is not None:
            if self.dataset.repo_id:
                raise ValueError(
                    "Set EITHER `dataset.repo_id` (single-dataset mode) OR "
                    "`dataset.repo_ids` (multi-dataset weighted-sampling mode), "
                    f"not both. Got repo_id={self.dataset.repo_id!r} AND "
                    f"repo_ids={self.dataset.repo_ids!r}."
                )
            if not isinstance(self.dataset.repo_ids, list) or len(self.dataset.repo_ids) == 0:
                raise ValueError(
                    "`dataset.repo_ids` must be a non-empty list of dataset repo ids "
                    f"in multi-dataset mode. Got {self.dataset.repo_ids!r}."
                )
            # `stats_path` (singular, single-dataset override) is mutually
            # exclusive with multi-dataset mode. lerobot_train.py applies it
            # AFTER make_dataset, which would silently CLOBBER the per-source
            # `norm_mode` aggregation — turning every run into effective
            # "base_only" regardless of --norm_mode. Per-source stats go through
            # `stats_paths` (plural) instead. Fail loudly rather than silently
            # normalize incorrectly. (resume_training.sh clears an inherited stats_path
            # by forwarding `--dataset.stats_path=`, which is falsy here.)
            if self.dataset.stats_path:
                raise ValueError(
                    "`dataset.stats_path` (single-dataset stats override) cannot be "
                    "combined with `dataset.repo_ids` (multi-dataset weighted-sampling "
                    "mode): it is applied after dataset construction and would silently "
                    "override the per-source `norm_mode` aggregation. Use per-source "
                    "`dataset.stats_paths` (plural) + `--norm_mode` instead, and clear any "
                    f"inherited path with `--dataset.stats_path=`. Got stats_path={self.dataset.stats_path!r}."
                )
            n = len(self.dataset.repo_ids)
            if self.dataset.sample_weights is None or len(self.dataset.sample_weights) != n:
                raise ValueError(
                    f"`dataset.sample_weights` must be a list of {n} floats parallel to "
                    f"`dataset.repo_ids`. Got {self.dataset.sample_weights!r}."
                )
            weight_sum = sum(self.dataset.sample_weights)
            if not (0.999 <= weight_sum <= 1.001):
                raise ValueError(
                    f"`dataset.sample_weights` must sum to 1.0 (±0.001). "
                    f"Got sum={weight_sum:.6f} from {self.dataset.sample_weights!r}."
                )
            if any(w < 0 for w in self.dataset.sample_weights):
                raise ValueError(
                    f"All `dataset.sample_weights` must be non-negative. Got {self.dataset.sample_weights!r}."
                )
            if self.dataset.stats_paths is None or len(self.dataset.stats_paths) != n:
                raise ValueError(
                    f"`dataset.stats_paths` must be a list of {n} stats sidecar paths "
                    f"parallel to `dataset.repo_ids`. Per-source normalization needs "
                    f"every sub-dataset's stats. Got {self.dataset.stats_paths!r}."
                )
        elif not self.dataset.repo_id:
            raise ValueError(
                "Either `dataset.repo_id` (single-dataset) or `dataset.repo_ids` "
                "(multi-dataset weighted sampling) must be set."
            )
        # If `repo_ids` is None and `repo_id` is set, validation passes and the
        # single-dataset code path runs unchanged below.

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = active_cfg.get_optimizer_preset()
            self.scheduler = active_cfg.get_scheduler_preset()

        if self.eval_steps > 0 and self.dataset.eval_split == 0.0:
            raise ValueError("eval_steps > 0 requires dataset.eval_split > 0.0 to hold out eval data.")

        # Remote runs auto-generate the repo_id in submit_to_hf (the policy may only be
        # resolved here, from --policy.path), so don't demand it up front for them.
        if (
            hasattr(active_cfg, "push_to_hub")
            and active_cfg.push_to_hub
            and not active_cfg.repo_id
            and not self.job.is_remote
        ):
            raise ValueError("'repo_id' argument missing. Please specify it to push the model to the hub.")

        if self.save_checkpoint_to_hub and not (self.policy is not None and self.policy.repo_id):
            raise ValueError("save_checkpoint_to_hub requires --policy.repo_id.")

        self._validate_distributed()

    def _validate_distributed(self) -> None:
        """Fail-fasts for the distributed-training scope.

        Raises:
            ValueError: If the config requests anything outside the verified scope: context
                parallelism or CFG parallelism (reserved placeholders), the compile or
                activation-checkpointing placeholders, a DCP checkpoint format on a
                non-sharded run, or — under sharded training — fp16 mixed precision, PEFT,
                reward-model training, in-training environment evaluation, or multi-optimizer
                configs.
        """
        if self.parallelism.cp_size > 1:
            raise ValueError(
                "Context parallelism is not implemented yet: --parallelism.context_parallel.* "
                "degrees must be 1 (reserved for the CP engine round)."
            )
        if self.parallelism.cfg_parallel != 1:
            raise ValueError(
                "CFG parallelism is inference-only and must be 1 for training "
                "(cfg_parallel is reserved for the serving round)."
            )
        if self.accelerator.compile.enabled:
            raise ValueError("--accelerator.compile is a placeholder and not wired yet.")
        if self.accelerator.activation_checkpointing.mode is not ActivationCheckpointingMode.NONE:
            raise ValueError("--accelerator.activation_checkpointing is a placeholder and not wired yet.")
        if self.checkpoint_format is not CheckpointFormat.SAFETENSORS and not self.parallelism.is_sharded:
            raise ValueError(
                f"checkpoint_format={self.checkpoint_format.value} requires a sharded run "
                "(--parallelism.dp_shard != 1); non-sharded checkpoints are always safetensors."
            )
        if self.parallelism.is_sharded:
            if self.accelerator.mixed_precision == "fp16":
                raise ValueError(
                    "fp16 is not supported under sharded training (GradScaler over DTensor "
                    "gradients is unverified); use bf16 or full precision."
                )
            if self.peft is not None:
                raise ValueError("PEFT is not supported under sharded training yet.")
            if self.is_reward_model_training:
                raise ValueError(
                    "Reward-model training is not supported under sharded training yet "
                    "(reward models declare no FSDP wrap units and have no sharded save path)."
                )
            if self.env is not None and self.env_eval_freq > 0:
                raise ValueError(
                    "In-training environment evaluation is not supported under sharded training "
                    "(a rank-0-only rollout of a sharded model deadlocks on collectives); set "
                    "--env_eval_freq=0 and evaluate with lerobot-eval on saved checkpoints."
                )
            if self.optimizer is not None and self.optimizer.builds_multiple_optimizers:
                raise ValueError("Multi-optimizer configs are not supported under sharded training.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Keys for draccus pretrained-path loading."""
        return ["policy", "reward_model"]

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]  # because of the third-party library draccus uses Any as the return type

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            dl_kwargs = {
                "repo_id": model_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "resume_download": resume_download,
                "token": token,
                "local_files_only": local_files_only,
            }
            try:
                config_file = hf_hub_download(filename=TRAIN_CONFIG_NAME, **dl_kwargs)
            except HfHubHTTPError as e:
                # No root train_config.json: this is a repo of periodic checkpoints from an
                # interrupted run. Fall back to the latest checkpoint's config so the run can be
                # resumed straight from the repo with `--config_path=<repo>`.
                latest = find_latest_hub_checkpoint(model_id, token=token, revision=revision)
                if latest is None:
                    raise FileNotFoundError(
                        f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                    ) from e
                config_file = hf_hub_download(
                    filename=f"{latest}/{PRETRAINED_MODEL_DIR}/{TRAIN_CONFIG_NAME}", **dl_kwargs
                )

        cli_args = kwargs.pop("cli_args", [])
        # Legacy RA-BC migration only applies to framework-saved checkpoints (always JSON).
        # Hand-written YAML/TOML configs are expected to use the current sample_weighting schema.
        if config_file is not None and config_file.endswith(".json"):
            with open(config_file) as f:
                config = json.load(f)
            # Chain migrations. Each helper returns None to fast-skip, else a
            # mutated dict; we thread the mutated dict through the chain so
            # multiple stale fields in the same checkpoint all get fixed.
            migrated_config: dict[str, Any] | None = None
            for migrate in (_migrate_legacy_renamed_fields, _migrate_legacy_rabc_fields):
                out = migrate(migrated_config if migrated_config is not None else config)
                if out is not None:
                    migrated_config = out
            if migrated_config is not None:
                with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
                    json.dump(migrated_config, f)
                    config_file = f.name

        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)
