#!/usr/bin/env python3
"""
Summarize evaluation results across multiple runs.

Usage:
    python summarize_evals.py <eval_output_folder>

Example:
    python summarize_evals.py /home/jennyw2/code/lerobot/outputs/eval_output/2026-01-28-102155_5episodes_successhightolerance_approachlever
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class EvalResult:
    """Parsed evaluation result with metadata."""

    folder_name: str
    method: str  # "diffusion" or "pi0.5"
    task: str  # e.g., "approach_lever"
    trajectory_gen: str  # "1st RRT", "5th RRT", or "5path"
    cameras: str  # "external", "wrist", or "external+wrist"
    checkpoint: str  # e.g., "25000", "50000", "last"
    checkpoint_numeric: int  # numeric value for sorting
    has_2enc: bool  # whether it uses 2 encoders
    has_90crop: bool  # whether it uses 90 crop

    # Metrics
    avg_sum_reward: float
    avg_max_reward: float
    pc_success: float
    n_episodes: int
    eval_ep_s: float
    avg_distance_to_goal: float | None


def parse_folder_name(folder_name: str) -> dict[str, Any]:
    """Parse folder name to extract metadata."""
    result: dict[str, Any] = {
        "method": None,
        "task": None,
        "trajectory_gen": None,
        "cameras": None,
        "checkpoint": None,
        "has_2enc": False,
        "has_90crop": False,
    }

    # Determine method
    if folder_name.startswith("diffusion_"):
        result["method"] = "diffusion"
        remaining = folder_name[len("diffusion_") :]
    elif folder_name.startswith("pi05_training_"):
        result["method"] = "pi0.5"
        remaining = folder_name[len("pi05_training_") :]
    else:
        return result

    # Extract task (approach_lever)
    task_match = re.match(r"(approach_lever)_(.+)", remaining)
    if task_match:
        result["task"] = task_match.group(1).replace("_", " ")
        remaining = task_match.group(2)

    # Check for special flags
    result["has_2enc"] = "_2enc_" in remaining or remaining.endswith("_2enc")
    result["has_90crop"] = "_90crop_" in remaining or "_90crop" in remaining

    # Extract trajectory generation method
    # Note: 1path/1strrtpath = "1st RRT", 5path/5thrrtpath = "5th RRT"
    if "1strrtpath" in remaining or "1path" in remaining:
        result["trajectory_gen"] = "1st RRT"
    elif "5thrrtpath" in remaining or "5path" in remaining:
        result["trajectory_gen"] = "5th RRT"

    # Extract cameras
    # Check for combined cameras first
    if "basewristrgb" in remaining or "basewrist" in remaining:
        result["cameras"] = "external+wrist"
    elif "basergb" in remaining or "_base_" in remaining or remaining.endswith("_base"):
        result["cameras"] = "external"
    elif "wristrgb" in remaining or "_wrist_" in remaining or remaining.endswith("_wrist"):
        result["cameras"] = "wrist"

    # More specific camera check using regex for pi0.5 style naming
    if result["cameras"] is None:
        if re.search(r"_base_\d+$", remaining) or re.search(r"_base_last$", remaining):
            result["cameras"] = "external"
        elif re.search(r"_wrist_\d+$", remaining) or re.search(r"_wrist_last$", remaining):
            result["cameras"] = "wrist"
        elif re.search(r"_basewrist_\d+$", remaining) or re.search(r"_basewrist_last$", remaining):
            result["cameras"] = "external+wrist"

    # Extract checkpoint
    checkpoint_match = re.search(r"_(\d{6}|last)$", remaining)
    if checkpoint_match:
        result["checkpoint"] = checkpoint_match.group(1)

    return result


def get_checkpoint_numeric(checkpoint: str, method: str) -> int:
    """Convert checkpoint string to numeric value for sorting."""
    if checkpoint == "last":
        # Last checkpoint values based on method
        if method == "pi0.5":
            return 3000
        else:  # diffusion
            return 100000  # Could be 75000 or 100000, using 100000 as default
    else:
        return int(checkpoint)


def load_eval_results(eval_folder: Path) -> list[EvalResult]:
    """Load all evaluation results from the folder."""
    results = []

    for subdir in sorted(eval_folder.iterdir()):
        if not subdir.is_dir():
            continue

        eval_info_path = subdir / "eval_info.json"
        if not eval_info_path.exists():
            print(f"Warning: No eval_info.json in {subdir.name}")
            continue

        with open(eval_info_path) as f:
            data = json.load(f)

        # Parse folder name
        metadata = parse_folder_name(subdir.name)

        if metadata["method"] is None:
            print(f"Warning: Could not parse folder name: {subdir.name}")
            continue

        # Extract metrics from overall
        overall = data.get("overall", {})

        result = EvalResult(
            folder_name=subdir.name,
            method=metadata["method"],
            task=metadata["task"],
            trajectory_gen=metadata["trajectory_gen"],
            cameras=metadata["cameras"],
            checkpoint=metadata["checkpoint"],
            checkpoint_numeric=get_checkpoint_numeric(metadata["checkpoint"], metadata["method"]),
            has_2enc=metadata["has_2enc"],
            has_90crop=metadata["has_90crop"],
            avg_sum_reward=overall.get("avg_sum_reward", 0.0),
            avg_max_reward=overall.get("avg_max_reward", 0.0),
            pc_success=overall.get("pc_success", 0.0),
            n_episodes=overall.get("n_episodes", 0),
            eval_ep_s=overall.get("eval_ep_s", 0.0),
            avg_distance_to_goal=overall.get("avg_distance_to_goal"),
        )
        results.append(result)

    return results


def create_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
    data = []
    for r in results:
        data.append(
            {
                "folder_name": r.folder_name,
                "method": r.method,
                "task": r.task,
                "trajectory_gen": r.trajectory_gen,
                "cameras": r.cameras,
                "checkpoint": r.checkpoint,
                "checkpoint_numeric": r.checkpoint_numeric,
                "has_2enc": r.has_2enc,
                "has_90crop": r.has_90crop,
                "avg_sum_reward": r.avg_sum_reward,
                "avg_max_reward": r.avg_max_reward,
                "pc_success": r.pc_success,
                "n_episodes": r.n_episodes,
                "eval_ep_s": r.eval_ep_s,
                "avg_distance_to_goal": r.avg_distance_to_goal,
            }
        )
    return pd.DataFrame(data)


CAMERA_ORDER = ["external", "wrist", "external+wrist"]


def plot_metric_by_category(df: pd.DataFrame, metric: str, output_dir: Path, title_suffix: str = ""):
    """Create plots for a metric grouped by different categories."""

    # Filter out rows with NaN for this metric if needed
    df_valid = df[df[metric].notna()].copy()

    if df_valid.empty:
        print(f"No valid data for metric: {metric}")
        return

    metric_label = metric.replace("_", " ").title()

    # 1. Bar plot by method
    fig, ax = plt.subplots(figsize=(10, 6))
    method_means = df_valid.groupby("method")[metric].mean()
    method_stds = df_valid.groupby("method")[metric].std()
    bars = ax.bar(method_means.index, method_means.values, yerr=method_stds.values, capsize=5)
    ax.set_xlabel("Method")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Method{title_suffix}")
    for bar, val in zip(bars, method_means.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + method_stds.max() * 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_method.png", dpi=150)
    plt.close()

    # 2. Bar plot by trajectory generation
    fig, ax = plt.subplots(figsize=(10, 6))
    traj_means = df_valid.groupby("trajectory_gen")[metric].mean()
    traj_stds = df_valid.groupby("trajectory_gen")[metric].std()
    bars = ax.bar(traj_means.index, traj_means.values, yerr=traj_stds.values, capsize=5)
    ax.set_xlabel("Trajectory Generation")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Trajectory Generation{title_suffix}")
    for bar, val in zip(bars, traj_means.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + traj_stds.max() * 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_trajectory_gen.png", dpi=150)
    plt.close()

    # 3. Bar plot by cameras
    fig, ax = plt.subplots(figsize=(10, 6))
    cam_means = df_valid.groupby("cameras")[metric].mean().reindex(CAMERA_ORDER)
    cam_stds = df_valid.groupby("cameras")[metric].std().reindex(CAMERA_ORDER)
    bars = ax.bar(cam_means.index, cam_means.values, yerr=cam_stds.values, capsize=5)
    ax.set_xlabel("Cameras")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Camera Configuration{title_suffix}")
    for bar, val in zip(bars, cam_means.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + cam_stds.max() * 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_cameras.png", dpi=150)
    plt.close()

    # 4. Grouped bar plot: method x trajectory_gen
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df_valid.pivot_table(values=metric, index="trajectory_gen", columns="method", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Trajectory Generation")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Trajectory Generation and Method{title_suffix}")
    ax.legend(title="Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_traj_and_method.png", dpi=150)
    plt.close()

    # 5. Grouped bar plot: method x cameras
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df_valid.pivot_table(values=metric, index="cameras", columns="method", aggfunc="mean")
    pivot = pivot.reindex(CAMERA_ORDER)
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Cameras")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Camera Configuration and Method{title_suffix}")
    ax.legend(title="Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_cameras_and_method.png", dpi=150)
    plt.close()

    # 6. Line plot: metric over checkpoints by method
    fig, ax = plt.subplots(figsize=(12, 6))
    for method in df_valid["method"].unique():
        method_df = df_valid[df_valid["method"] == method]
        checkpoint_means = method_df.groupby("checkpoint_numeric")[metric].mean().sort_index()
        ax.plot(checkpoint_means.index, checkpoint_means.values, marker="o", label=method)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} over Checkpoints by Method{title_suffix}")
    ax.legend()
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_over_checkpoints.png", dpi=150)
    plt.close()


def plot_heatmap(df: pd.DataFrame, metric: str, output_dir: Path):
    """Create a heatmap showing metric across method/trajectory/camera combinations."""
    df_valid = df[df[metric].notna()].copy()

    if df_valid.empty:
        return

    metric_label = metric.replace("_", " ").title()

    # Create a combined category with consistent ordering
    df_valid["config"] = df_valid["trajectory_gen"] + " / " + df_valid["cameras"]

    # Define the order for configs (trajectory x camera)
    traj_order = ["1st RRT", "5th RRT"]
    config_order = [f"{t} / {c}" for t in traj_order for c in CAMERA_ORDER]

    # Pivot for heatmap
    pivot = df_valid.pivot_table(values=metric, index="config", columns="method", aggfunc="mean")
    # Reindex to get consistent ordering (only keep configs that exist)
    config_order_existing = [c for c in config_order if c in pivot.index]
    pivot = pivot.reindex(config_order_existing)

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_label)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > pivot.values.max() * 0.5 else "black",
                )

    ax.set_title(f"{metric_label} Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_heatmap.png", dpi=150)
    plt.close()


def plot_success_rate_comparison(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive success rate comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Success rate by method and checkpoint (line plot)
    ax = axes[0, 0]
    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        checkpoint_means = method_df.groupby("checkpoint_numeric")["pc_success"].mean().sort_index()
        ax.plot(checkpoint_means.index, checkpoint_means.values, marker="o", label=method, linewidth=2)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate over Checkpoints")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # 2. Success rate by trajectory and method (grouped bar)
    ax = axes[0, 1]
    pivot = df.pivot_table(values="pc_success", index="trajectory_gen", columns="method", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Trajectory Generation")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Trajectory and Method")
    ax.legend(title="Method")
    plt.sca(ax)
    plt.xticks(rotation=45)

    # 3. Success rate by cameras and method (grouped bar)
    ax = axes[1, 0]
    pivot = df.pivot_table(values="pc_success", index="cameras", columns="method", aggfunc="mean")
    pivot = pivot.reindex(CAMERA_ORDER)
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Cameras")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Camera and Method")
    ax.legend(title="Method")
    plt.sca(ax)
    plt.xticks(rotation=45)

    # 4. Best configurations (top 10)
    ax = axes[1, 1]
    df_sorted = df.sort_values("pc_success", ascending=False).head(15)
    labels = [
        f"{r.method[:4]}/{r.trajectory_gen}/{r.cameras[:4]}/{r.checkpoint}" for _, r in df_sorted.iterrows()
    ]
    colors = ["#2ecc71" if r.method == "pi0.5" else "#3498db" for _, r in df_sorted.iterrows()]
    ax.barh(range(len(labels)), df_sorted["pc_success"].values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title("Top 15 Configurations by Success Rate")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_comparison.png", dpi=150)
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate a summary CSV and print summary statistics."""
    # Save full results
    df.to_csv(output_dir / "all_results.csv", index=False)

    # Summary by method
    method_summary = (
        df.groupby("method")
        .agg(
            {
                "avg_sum_reward": ["mean", "std"],
                "pc_success": ["mean", "std"],
                "eval_ep_s": ["mean", "std"],
                "avg_distance_to_goal": ["mean", "std"],
            }
        )
        .round(3)
    )
    method_summary.to_csv(output_dir / "summary_by_method.csv")

    # Summary by trajectory
    traj_summary = (
        df.groupby("trajectory_gen")
        .agg(
            {
                "avg_sum_reward": ["mean", "std"],
                "pc_success": ["mean", "std"],
                "eval_ep_s": ["mean", "std"],
                "avg_distance_to_goal": ["mean", "std"],
            }
        )
        .round(3)
    )
    traj_summary.to_csv(output_dir / "summary_by_trajectory.csv")

    # Summary by cameras
    cam_summary = (
        df.groupby("cameras")
        .agg(
            {
                "avg_sum_reward": ["mean", "std"],
                "pc_success": ["mean", "std"],
                "eval_ep_s": ["mean", "std"],
                "avg_distance_to_goal": ["mean", "std"],
            }
        )
        .round(3)
    )
    cam_summary.to_csv(output_dir / "summary_by_cameras.csv")

    # Best configurations
    best_configs = df.nlargest(10, "pc_success")[
        ["folder_name", "method", "trajectory_gen", "cameras", "checkpoint", "pc_success", "avg_sum_reward"]
    ]
    best_configs.to_csv(output_dir / "best_configurations.csv", index=False)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nTotal evaluations: {len(df)}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Trajectory generation: {df['trajectory_gen'].unique().tolist()}")
    print(f"Camera configurations: {df['cameras'].unique().tolist()}")

    print("\n--- Summary by Method ---")
    print(method_summary.to_string())

    print("\n--- Summary by Trajectory Generation ---")
    print(traj_summary.to_string())

    print("\n--- Summary by Cameras ---")
    print(cam_summary.to_string())

    print("\n--- Top 10 Configurations ---")
    print(best_configs.to_string(index=False))

    return method_summary, traj_summary, cam_summary


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("eval_folder", type=str, help="Path to the evaluation output folder")
    args = parser.parse_args()

    eval_folder = Path(args.eval_folder)
    if not eval_folder.exists():
        print(f"Error: Folder does not exist: {eval_folder}")
        return 1

    # Create output directory for plots
    output_dir = eval_folder / "summary"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {eval_folder}")
    results = load_eval_results(eval_folder)
    print(f"Loaded {len(results)} evaluation results")

    if not results:
        print("No results found!")
        return 1

    # Create DataFrame
    df = create_dataframe(results)

    # Generate summary tables
    generate_summary_table(df, output_dir)

    # Generate plots for each metric
    print("\nGenerating plots...")

    # avg_sum_reward plots
    plot_metric_by_category(df, "avg_sum_reward", output_dir)
    plot_heatmap(df, "avg_sum_reward", output_dir)

    # eval_ep_s plots
    plot_metric_by_category(df, "eval_ep_s", output_dir)
    plot_heatmap(df, "eval_ep_s", output_dir)

    # avg_distance_to_goal plots (only for entries that have it)
    df_with_distance = df[df["avg_distance_to_goal"].notna()]
    if not df_with_distance.empty:
        plot_metric_by_category(
            df_with_distance,
            "avg_distance_to_goal",
            output_dir,
            title_suffix=" (subset with distance metric)",
        )
        plot_heatmap(df_with_distance, "avg_distance_to_goal", output_dir)
    else:
        print("No entries have avg_distance_to_goal metric")

    # pc_success plots
    plot_metric_by_category(df, "pc_success", output_dir)
    plot_heatmap(df, "pc_success", output_dir)

    # Comprehensive success rate comparison
    plot_success_rate_comparison(df, output_dir)

    print(f"\nPlots saved to: {output_dir}")
    print("Generated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    exit(main())
