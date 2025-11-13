#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualization utilities for Phase P1 experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path("results/final_experiments")
CHART_DIR = Path("charts")
TRAINING_LOG_DIR = Path("results/training_runs")


SCHEDULER_ORDER: list[str] = [
    "WASS-RAG (Full)",
    "WASS-RAG (HEFT-only)",
    "WASS-DRL (Vanilla)",
    "HEFT",
    "MIN-MIN",
]


plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_inputs(detailed_csv: Path) -> pd.DataFrame:
    if not detailed_csv.exists():
        raise FileNotFoundError(f"Missing experimental results: {detailed_csv}")
    df = pd.read_csv(detailed_csv)
    expected_cols = {"scheduler", "workflow", "makespan", "status", "task_count", "seed"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"detailed_results.csv missing columns: {sorted(missing)}")
    df = df[df["status"] == "success"].copy()
    return df


def _order_schedulers(labels: Iterable[str]) -> list[str]:
    present = [name for name in SCHEDULER_ORDER if name in labels]
    absent = sorted(set(labels) - set(SCHEDULER_ORDER))
    return present + absent


def plot_overall_bar(df: pd.DataFrame) -> None:
    grouped = df.groupby("scheduler")["makespan"].agg(["mean", "std", "count"])
    grouped = grouped.reindex(_order_schedulers(grouped.index))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"], capsize=6, color="#6096ba")
    ax.set_ylabel("Average Makespan")
    ax.set_title("Overall Makespan Across Strategies")
    ax.grid(axis="y", alpha=0.3)
    for idx, (mean_val, std_val) in enumerate(zip(grouped["mean"], grouped["std"].fillna(0.0))):
        ax.text(idx, mean_val + (std_val or 0) + 0.05, f"{mean_val:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(CHART_DIR / "overall_makespan_bar.png", dpi=300)
    plt.close(fig)


def plot_per_workflow(df: pd.DataFrame) -> None:
    workflows = sorted(df["workflow"].unique())
    fig, axes = plt.subplots(1, len(workflows), figsize=(5 * len(workflows), 5), sharey=True)
    if len(workflows) == 1:
        axes = [axes]
    for ax, workflow in zip(axes, workflows):
        subset = df[df["workflow"] == workflow]
        stats = subset.groupby("scheduler")["makespan"].agg(["mean", "std"])
        stats = stats.reindex(_order_schedulers(stats.index))
        ax.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=5, color="#c6def1")
        ax.set_title(Path(workflow).stem)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
    axes[0].set_ylabel("Average Makespan")
    plt.tight_layout()
    fig.savefig(CHART_DIR / "makespan_by_workflow.png", dpi=300)
    plt.close(fig)


def plot_distribution_box(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = _order_schedulers(df["scheduler"].unique())
    data = [df.loc[df["scheduler"] == label, "makespan"].values for label in ordered]
    ax.boxplot(data, labels=ordered, patch_artist=True, boxprops=dict(facecolor="#9ad1d4"))
    ax.set_ylabel("Makespan")
    ax.set_title("Makespan Distribution per Strategy")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(CHART_DIR / "makespan_boxplot.png", dpi=300)
    plt.close(fig)


def export_ablation_summary(df: pd.DataFrame, output_csv: Path) -> None:
    baseline = "WASS-DRL (Vanilla)"
    if baseline not in df["scheduler"].unique():
        print("Baseline WASS-DRL (Vanilla) missing; skip ablation summary.")
        return
    summary = df.groupby(["workflow", "scheduler"])["makespan"].agg(["mean", "std"]).reset_index()
    baseline_stats = summary[summary["scheduler"] == baseline][["workflow", "mean"]].rename(columns={"mean": "baseline_mean"})
    merged = summary.merge(baseline_stats, on="workflow", how="left")
    merged["delta_vs_baseline"] = merged["mean"] - merged["baseline_mean"]
    merged["relative_improvement_pct"] = -(merged["delta_vs_baseline"] / merged["baseline_mean"]) * 100
    merged.to_csv(output_csv, index=False)


def load_training_logs() -> pd.DataFrame:
    if not TRAINING_LOG_DIR.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for csv_path in TRAINING_LOG_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover
            print(f"⚠️ Unable to read training log {csv_path.name}: {exc}")
            continue
        df["log_file"] = csv_path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["episode"] = pd.to_numeric(combined.get("episode"), errors="coerce")
    combined["episode_reward"] = pd.to_numeric(combined.get("episode_reward"), errors="coerce")
    combined["status"] = combined.get("status", "success").fillna("success")
    combined = combined[(combined["status"] == "success") & combined["episode"].notna() & combined["episode_reward"].notna()]
    if combined.empty:
        return pd.DataFrame()
    if "strategy_label" not in combined.columns:
        combined["strategy_label"] = combined.get("strategy", "unknown")
    combined["seed"] = combined.get("seed").fillna(-1)
    combined["seed"] = pd.to_numeric(combined["seed"], errors="coerce").fillna(-1).astype(int)
    combined["episode"] = combined["episode"].astype(int)
    # Normalize optional numeric fields introduced by reward normalization logging.
    numeric_cols = [
        "rag_mean",
        "rag_std",
        "rag_positive_frac",
        "rag_negative_frac",
        "rag_min",
        "rag_max",
        "reward_mean",
        "reward_std",
        "reward_positive_frac",
        "reward_negative_frac",
        "final_reward_component",
        "clamped_pct",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    if "workflow_band" not in combined.columns:
        combined["workflow_band"] = "unknown"
    else:
        combined["workflow_band"] = combined["workflow_band"].fillna("unknown")
    if "workflow" not in combined.columns:
        combined["workflow"] = ""
    return combined


def plot_training_curves(log_df: pd.DataFrame, smoothing_window: int = 5) -> None:
    if log_df.empty:
        print("⚠️ No training logs found; skipping training curves.")
        return

    log_df = log_df.copy().sort_values(["strategy_label", "log_file", "episode"])
    group_cols = ["strategy_label", "log_file"]
    log_df["smoothed_reward"] = (
        log_df.groupby(group_cols)["episode_reward"].transform(
            lambda series: series.rolling(window=smoothing_window, min_periods=1).mean()
        )
    )
    agg = log_df.groupby(["strategy_label", "episode"]) ["smoothed_reward"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "reward_mean", "std": "reward_std", "count": "sample_count"})
    agg["reward_std"] = agg["reward_std"].fillna(0.0)

    ordered_labels = _order_schedulers(agg["strategy_label"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in ordered_labels:
        subset = agg[agg["strategy_label"] == label]
        if subset.empty:
            continue
        ax.plot(subset["episode"], subset["reward_mean"], label=label)
        if (subset["reward_std"] > 0).any():
            upper = subset["reward_mean"] + subset["reward_std"]
            lower = subset["reward_mean"] - subset["reward_std"]
            ax.fill_between(subset["episode"], lower, upper, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Episode Reward")
    ax.set_title("Training Curves Across Strategies")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(CHART_DIR / "training_curves.png", dpi=300)
    plt.close(fig)


def export_reward_norm_summary(log_df: pd.DataFrame, output_csv: Path) -> None:
    if log_df.empty:
        return

    required_cols = {
        "workflow_band",
        "rag_mean",
        "rag_std",
        "rag_positive_frac",
        "rag_negative_frac",
        "reward_mean",
        "reward_std",
        "reward_positive_frac",
        "reward_negative_frac",
        "final_reward_component",
    }
    missing = required_cols - set(log_df.columns)
    if missing:
        print(f"⚠️ Reward normalization columns missing ({sorted(missing)}); skip summary export.")
        return

    df = log_df.copy()
    df["family"] = df.get("workflow", "").fillna("").astype(str).str.split("-", n=1).str[0]

    metric_cols = [
        "rag_mean",
        "rag_std",
        "rag_positive_frac",
        "rag_negative_frac",
        "reward_mean",
        "reward_std",
        "reward_positive_frac",
        "reward_negative_frac",
        "final_reward_component",
    ]

    records: list[pd.DataFrame] = []

    def append_group(scope: str, group_cols: list[str] | None) -> None:
        ordered_cols = ["scope", "workflow_band", "family", *metric_cols]
        if not group_cols:
            metrics = df[metric_cols].mean().to_frame().T
            metrics.insert(0, "scope", scope)
            metrics["workflow_band"] = None
            metrics["family"] = None
            metrics = metrics[[col for col in ordered_cols if col in metrics.columns]]
            records.append(metrics)
            return
        grouped = df.groupby(group_cols)[metric_cols].mean().reset_index()
        grouped.insert(0, "scope", scope)
        if "workflow_band" not in group_cols:
            grouped["workflow_band"] = None
        if "family" not in group_cols:
            grouped["family"] = None
        # Ensure column ordering is consistent before appending.
        grouped = grouped[[col for col in ordered_cols if col in grouped.columns]]
        records.append(grouped)

    append_group("overall", None)
    append_group("by_band", ["workflow_band"])
    append_group("by_family", ["family"])
    append_group("by_family_band", ["family", "workflow_band"])

    summary_df = pd.concat(records, ignore_index=True)
    # Sort for readability: scope order plus alphabetical keys.
    scope_order = {"overall": 0, "by_band": 1, "by_family": 2, "by_family_band": 3}
    summary_df["scope_rank"] = summary_df["scope"].map(scope_order).fillna(99)
    summary_df = summary_df.sort_values(["scope_rank", "family", "workflow_band"], na_position="last")
    summary_df = summary_df.drop(columns=["scope_rank"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"✅ Reward normalization summary: {output_csv}")


def main(results_dir: Path) -> None:
    detailed_csv = results_dir / "detailed_results.csv"
    summary_csv = results_dir / "summary_results.csv"
    ablation_csv = results_dir / "ablation_summary.csv"
    print("📊 Generating Phase P1 charts...")
    try:
        df = _ensure_inputs(detailed_csv)
    except FileNotFoundError as exc:
        print(f"⚠️ {exc}")
        df = pd.DataFrame()

    if not df.empty:
        CHART_DIR.mkdir(exist_ok=True)
        plot_overall_bar(df)
        plot_per_workflow(df)
        plot_distribution_box(df)
        export_ablation_summary(df, ablation_csv)
    training_logs = load_training_logs()
    plot_training_curves(training_logs)
    export_reward_norm_summary(training_logs, TRAINING_LOG_DIR / "reward_norm_summary.csv")
    generated = [p.name for p in CHART_DIR.glob("*.png")]
    if generated:
        print("✅ Charts saved:")
        for name in sorted(generated):
            print(f"  - {name}")
    if ablation_csv.exists():
        print(f"✅ Ablation summary: {ablation_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment comparisons from detailed_results.csv")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing detailed_results.csv (default: results/final_experiments)",
    )
    args = parser.parse_args()
    main(args.results_dir)
