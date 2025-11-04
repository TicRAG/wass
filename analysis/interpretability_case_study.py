#!/usr/bin/env python3
"""Utilities for inspecting teacher trace logs produced during PBRS training.

This script loads JSONL trace files emitted by ``scripts/2_train_rag_agent.py``
when ``--trace_log_dir`` is enabled. It aggregates task-level reward shaping
signals, flattens neighbour metadata, and optionally persists CSV/JSON
artifacts that power downstream interpretability notebooks.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass
class TraceData:
    entries: pd.DataFrame
    neighbors: pd.DataFrame


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Failed to parse JSON on line {line_no}: {exc}") from exc


def load_trace(path: Path) -> TraceData:
    record_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    for idx, payload in enumerate(_read_jsonl(path)):
        context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
        entry = {
            "entry_id": idx,
            "run_label": context.get("run_label"),
            "episode": context.get("episode"),
            "workflow_file": context.get("workflow_file"),
            "task_name": payload.get("task_name"),
            "decision_host": payload.get("decision_host"),
            "timestamp": payload.get("timestamp"),
            "agent_eft": payload.get("agent_eft"),
            "potential": payload.get("potential"),
            "potential_before": payload.get("potential_before"),
            "potential_after": payload.get("potential_after"),
            "potential_delta": payload.get("potential_delta"),
            "shaped_reward": payload.get("shaped_reward"),
            "top_k": payload.get("top_k"),
            "lambda": payload.get("lambda"),
            "gamma": payload.get("gamma"),
            "temperature": payload.get("temperature"),
        }
        record_rows.append(entry)
        neighbors = payload.get("neighbors") or []
        for rank, neighbor in enumerate(neighbors, start=1):
            if not isinstance(neighbor, dict):
                continue
            neighbor_rows.append({
                "entry_id": idx,
                "rank": rank,
                "neighbor_workflow": neighbor.get("workflow_file"),
                "neighbor_scheduler": neighbor.get("scheduler_used"),
                "neighbor_q_value": neighbor.get("q_value"),
                "neighbor_similarity": neighbor.get("similarity"),
                "neighbor_weight": neighbor.get("weight"),
            })
    entries_df = pd.DataFrame(record_rows)
    neighbors_df = pd.DataFrame(neighbor_rows)
    return TraceData(entries=entries_df, neighbors=neighbors_df)


def _build_top_events(entries: pd.DataFrame, k: int) -> dict[str, list[dict[str, Any]]]:
    shaped = entries.dropna(subset=["shaped_reward"]).copy()
    if shaped.empty or k <= 0:
        return {"top_positive": [], "top_negative": []}
    shaped["shaped_reward"] = shaped["shaped_reward"].astype(float)
    shaped["potential_delta"] = shaped["potential_delta"].astype(float, errors="ignore")

    def _select(df: pd.DataFrame, ascending: bool) -> list[dict[str, Any]]:
        subset = df.sort_values("shaped_reward", ascending=ascending).head(k)
        cols = [
            "shaped_reward",
            "potential_delta",
            "task_name",
            "workflow_file",
            "decision_host",
            "episode",
            "agent_eft",
        ]
        return [
            {col: (row[col] if pd.notna(row[col]) else None) for col in cols}
            for _, row in subset.iterrows()
        ]

    top_positive = _select(shaped, ascending=False)
    top_negative = _select(shaped, ascending=True)
    return {"top_positive": top_positive, "top_negative": top_negative}


def _print_summary(trace: TraceData, *, top_k: int) -> dict[str, list[dict[str, Any]]]:
    entries = trace.entries
    if entries.empty:
        print("No trace entries found.")
        return {"top_positive": [], "top_negative": []}
    run_labels = entries["run_label"].dropna().unique().tolist()
    workflows = entries["workflow_file"].dropna().unique().tolist()
    shaped = entries["shaped_reward"].dropna().astype(float)
    potential_delta = entries["potential_delta"].dropna().astype(float)
    print("Trace Summary")
    print("--------------")
    print(f"Total entries      : {len(entries)}")
    print(f"Distinct workflows : {len(workflows)} -> {', '.join(workflows)}")
    print(f"Run labels         : {', '.join(run_labels) if run_labels else 'n/a'}")
    print(f"Hosts used         : {entries['decision_host'].dropna().nunique()}")
    if not shaped.empty:
        print(
            "Shaped reward      : mean={:.4f} min={:.4f} max={:.4f} std={:.4f}".format(
                shaped.mean(), shaped.min(), shaped.max(), shaped.std(ddof=0)
            )
        )
    if not potential_delta.empty:
        print(
            "Δpotential         : mean={:.4f} min={:.4f} max={:.4f}".format(
                potential_delta.mean(), potential_delta.min(), potential_delta.max()
            )
        )
    per_host = entries.groupby("decision_host")["shaped_reward"].mean().dropna()
    if not per_host.empty:
        print("Average reward per host:")
        for host, avg_reward in per_host.sort_values(ascending=False).items():
            print(f"  - {host}: {avg_reward:.4f}")
    top_events = _build_top_events(entries, top_k)
    if top_events["top_positive"]:
        print(f"Top +{top_k} shaped rewards:")
        for item in top_events["top_positive"]:
            print(
                "  + {task} ({workflow}) episode={episode} host={host} reward={reward:.4f}".format(
                    task=item.get("task_name"),
                    workflow=item.get("workflow_file"),
                    episode=int(item["episode"]) if item.get("episode") is not None else "?",
                    host=item.get("decision_host"),
                    reward=float(item["shaped_reward"]),
                )
            )
    if top_events["top_negative"]:
        print(f"Top -{top_k} shaped rewards:")
        for item in top_events["top_negative"]:
            print(
                "  - {task} ({workflow}) episode={episode} host={host} reward={reward:.4f}".format(
                    task=item.get("task_name"),
                    workflow=item.get("workflow_file"),
                    episode=int(item["episode"]) if item.get("episode") is not None else "?",
                    host=item.get("decision_host"),
                    reward=float(item["shaped_reward"]),
                )
            )
    return top_events


def _apply_filters(trace: TraceData, args: argparse.Namespace) -> TraceData:
    entries = trace.entries
    neighbors = trace.neighbors
    mask = pd.Series([True] * len(entries))
    if args.episodes:
        mask &= entries["episode"].isin(args.episodes)
    if args.workflow:
        mask &= entries["workflow_file"] == args.workflow
    if args.task:
        mask &= entries["task_name"] == args.task
    filtered_entries = entries[mask]
    filtered_neighbors = neighbors[neighbors["entry_id"].isin(filtered_entries["entry_id"])].copy()
    return TraceData(entries=filtered_entries.reset_index(drop=True), neighbors=filtered_neighbors.reset_index(drop=True))


def _write_outputs(trace: TraceData, output_dir: Path, *, top_events: dict[str, list[dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries_path = output_dir / "trace_entries.csv"
    neighbors_path = output_dir / "trace_neighbors.csv"
    summary_path = output_dir / "trace_summary.json"
    trace.entries.to_csv(entries_path, index=False)
    trace.neighbors.to_csv(neighbors_path, index=False)
    summary = {
        "total_entries": int(trace.entries.shape[0]),
        "unique_workflows": sorted(set(trace.entries["workflow_file"].dropna())),
        "episodes": sorted(e for e in trace.entries["episode"].dropna().unique().tolist() if e is not None),
        "hosts": sorted(set(trace.entries["decision_host"].dropna())),
        "reward_stats": {
            "mean": float(trace.entries["shaped_reward"].dropna().mean()) if not trace.entries["shaped_reward"].dropna().empty else math.nan,
            "min": float(trace.entries["shaped_reward"].dropna().min()) if not trace.entries["shaped_reward"].dropna().empty else math.nan,
            "max": float(trace.entries["shaped_reward"].dropna().max()) if not trace.entries["shaped_reward"].dropna().empty else math.nan,
        },
        "top_events": top_events,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved entries  -> {entries_path}")
    print(f"Saved neighbors-> {neighbors_path}")
    print(f"Saved summary  -> {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse teacher trace JSONL logs for interpretability case studies.")
    parser.add_argument("trace_log", type=Path, help="Path to trace JSONL produced by RAG training.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory to write CSV/JSON artifacts.")
    parser.add_argument("--workflow", help="Filter to a single workflow filename (exact match).")
    parser.add_argument("--task", help="Filter to a single task name (exact match).")
    parser.add_argument("--episodes", type=int, nargs="+", help="Filter to one or more episode numbers.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of extreme reward events to highlight in the summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_log = args.trace_log.expanduser().resolve()
    if not trace_log.exists():
        raise FileNotFoundError(f"Trace log not found: {trace_log}")
    trace = load_trace(trace_log)
    trace = _apply_filters(trace, args)
    top_events = _print_summary(trace, top_k=max(args.top_k, 0))
    if args.output_dir:
        _write_outputs(trace, args.output_dir.expanduser().resolve(), top_events=top_events)


if __name__ == "__main__":  # pragma: no cover
    main()
