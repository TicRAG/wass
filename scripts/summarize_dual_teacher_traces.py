#!/usr/bin/env python3
"""Summarise teacher trace logs produced during the dual-teacher sweep.

The script iterates over every JSONL trace under ``results/wass_rag_dual_teacher/traces``
and exports a CSV containing per-run statistics such as host assignment counts
and shaped-reward aggregates. This enables quick inspection of host bias after
the hyperparameter grid completes.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

TRACE_ROOT = Path("results/wass_rag_dual_teacher/traces")
OUTPUT_CSV = Path("results/wass_rag_dual_teacher/trace_summary_20251110.csv")


@dataclass
class TraceSummary:
    path: Path
    run_label: str | None
    workflow: str | None
    seed: int | None
    repeat: int | None
    total_entries: int
    host_counts: dict[str, int]
    shaped_reward_mean: float | None
    shaped_reward_min: float | None
    shaped_reward_max: float | None


def summarise_trace(path: Path) -> TraceSummary | None:
    host_counter: Counter[str] = Counter()
    rewards: list[float] = []
    run_label: str | None = None
    workflow: str | None = None
    seed: int | None = None
    repeat: int | None = None
    total_entries = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_entries += 1
            context = payload.get("context") or {}
            if run_label is None:
                run_label = context.get("run_label")
            if workflow is None:
                workflow = context.get("workflow_file")
            if seed is None:
                seed = context.get("seed")
            if repeat is None:
                repeat = context.get("repeat")
            host = payload.get("decision_host")
            if host:
                host_counter[str(host)] += 1
            shaped = payload.get("shaped_reward")
            if isinstance(shaped, (int, float)):
                rewards.append(float(shaped))

    if total_entries == 0:
        return None

    rewards_sorted = sorted(rewards)
    shaped_mean = sum(rewards_sorted) / len(rewards_sorted) if rewards_sorted else None
    shaped_min = rewards_sorted[0] if rewards_sorted else None
    shaped_max = rewards_sorted[-1] if rewards_sorted else None

    return TraceSummary(
        path=path,
        run_label=run_label,
        workflow=workflow,
        seed=int(seed) if seed is not None else None,
        repeat=int(repeat) if repeat is not None else None,
        total_entries=total_entries,
        host_counts=dict(host_counter),
        shaped_reward_mean=shaped_mean,
        shaped_reward_min=shaped_min,
        shaped_reward_max=shaped_max,
    )


def main() -> None:
    if not TRACE_ROOT.exists():
        raise FileNotFoundError(f"Trace directory not found: {TRACE_ROOT}")

    summaries: list[TraceSummary] = []
    for path in sorted(TRACE_ROOT.rglob("*.jsonl")):
        summary = summarise_trace(path)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("❌ No trace files were summarised.")
        return

    rows: list[dict[str, Any]] = []
    host_keys: set[str] = set()
    for summary in summaries:
        host_keys.update(summary.host_counts.keys())

    for summary in summaries:
        row = {
            "trace_path": summary.path.as_posix(),
            "run_label": summary.run_label,
            "workflow": summary.workflow,
            "seed": summary.seed,
            "repeat": summary.repeat,
            "total_entries": summary.total_entries,
            "shaped_mean": summary.shaped_reward_mean,
            "shaped_min": summary.shaped_reward_min,
            "shaped_max": summary.shaped_reward_max,
        }
        for host in host_keys:
            row[f"host_{host}"] = summary.host_counts.get(host, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(by=["workflow", "seed", "run_label"], inplace=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Trace summary written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
