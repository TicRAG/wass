#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualization utilities for Phase P1 experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results/final_experiments")
DETAILED_CSV = RESULTS_DIR / "detailed_results.csv"
SUMMARY_CSV = RESULTS_DIR / "summary_results.csv"
ABLATION_CSV = RESULTS_DIR / "ablation_summary.csv"
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


def _ensure_inputs() -> pd.DataFrame:
    if not DETAILED_CSV.exists():
        raise FileNotFoundError(f"Missing experimental results: {DETAILED_CSV}")
    df = pd.read_csv(DETAILED_CSV)
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


def export_ablation_summary(df: pd.DataFrame) -> None:
    baseline = "WASS-DRL (Vanilla)"
    if baseline not in df["scheduler"].unique():
        print("Baseline WASS-DRL (Vanilla) missing; skip ablation summary.")
        return
    summary = df.groupby(["workflow", "scheduler"])["makespan"].agg(["mean", "std"]).reset_index()
    baseline_stats = summary[summary["scheduler"] == baseline][["workflow", "mean"]].rename(columns={"mean": "baseline_mean"})
    merged = summary.merge(baseline_stats, on="workflow", how="left")
    merged["delta_vs_baseline"] = merged["mean"] - merged["baseline_mean"]
    merged["relative_improvement_pct"] = -(merged["delta_vs_baseline"] / merged["baseline_mean"]) * 100
    merged.to_csv(ABLATION_CSV, index=False)


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


def main() -> None:
    print("📊 Generating Phase P1 charts...")
    df = _ensure_inputs()
    CHART_DIR.mkdir(exist_ok=True)
    plot_overall_bar(df)
    plot_per_workflow(df)
    plot_distribution_box(df)
    export_ablation_summary(df)
    training_logs = load_training_logs()
    plot_training_curves(training_logs)
    generated = [p.name for p in CHART_DIR.glob("*.png")]
    if generated:
        print("✅ Charts saved:")
        for name in sorted(generated):
            print(f"  - {name}")
    if ABLATION_CSV.exists():
        print(f"✅ Ablation summary: {ABLATION_CSV}")


if __name__ == "__main__":
    main()
