This file provides guidance to AI agents when working with code in this repository.

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **SA wrapper guidance sources** (`src/lerobot/policies/guidance/`): `SharedAutonomyPolicyWrapper` hosts three pluggable `GuidanceSource` implementations — `RRTGuidanceSource` (planner-based), `OracleGoalGuidanceSource` (joint-space interpolation toward `q_goal_bias`), `ObservationTeleopGuidanceSource` (consumes `observation.policy_guidance_chunk`). `select_action` dispatches in that priority order. The wrapper's `_rrt` attribute is a `@property` that returns a `_RRTBackCompatView` proxy to the RRT source's state — **load-bearing** for external callers (`InterventionController` in `lerobot.scripts.intervention_controller`, `shared_autonomy_gui.py`, `last_mile/helpers.py`) that read/write `wrapper._rrt.mode` / `wrapper._rrt.target_steps`. Don't remove the property without first auditing those callsites.
- **Unified eval entry point**: `lerobot-eval` is the sole script for running policies through scenarios. Four modes: (1) passive eval (no SA), (2) keyboard teleop recording (`--policy.shared_autonomy_config.enabled=true --policy.shared_autonomy_config.start_paused=true --env.teleop_dataset_repo_id=...`), (3) RRT intervention (`--intervention.method=rrt`), (4) oracle-goal intervention (`--intervention.method=oracle_goal`). The legacy `my_scripts/intervention_record.py` was merged into `lerobot-eval`; its `InterventionController` now lives at `lerobot.scripts.intervention_controller`, its `InterventionConfig` at `lerobot.configs.intervention`, and intervention-mode is selected by setting `EvalPipelineConfig.intervention` to a non-None instance. Intervention mode forces `batch_size=1` + sync envs (validated at startup), forces `_run_event.set()` and `auto_pause_on_rrt_finish=False` on the SA wrapper, and writes a per-scenario CSV to `<output_dir>/intervention_per_scenario.csv` alongside the standard `eval_info.json`.
