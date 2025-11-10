#!/usr/bin/env python3
"""Execute the WASS-RAG dual-teacher hyperparameter grid and collate results.

The grid spans temperature, greedy threshold, top-k sampling, and epsilon values
across the three benchmark workflows defined for Task 5. After all simulations
finish, a consolidated CSV is generated containing per-strategy metrics for each
parameter combination.
"""
from __future__ import annotations

import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class WorkflowGroup:
    name: str
    workflow_dir: Path
    workflows: list[str]


SEEDS: tuple[int, ...] = (0, 1, 2)
TEMPERATURES: tuple[float, ...] = (0.6, 0.7, 0.8)
GREEDY_THRESHOLDS: tuple[float, ...] = (0.85, 0.9)
TOP_K_VALUES: tuple[int, ...] = (2, 3, 4)
EPSILONS: tuple[float, ...] = (0.0, 0.05)
RESULT_ROOT = Path("results/wass_rag_dual_teacher")
SCANS_ROOT = RESULT_ROOT / "scans"
TRACE_ROOT = RESULT_ROOT / "traces"
AGGREGATE_CSV = RESULT_ROOT / "sensitivity_20251110.csv"
MODEL_PATH = Path("models/saved_models/drl_agent.pth")

WORKFLOW_GROUPS: tuple[WorkflowGroup, ...] = (
    WorkflowGroup(
        name="montage",
        workflow_dir=Path("data/workflows/training"),
        workflows=["montage-chameleon-2mass-01d-001_aug1.json"],
    ),
    WorkflowGroup(
        name="synthetic",
        workflow_dir=Path("data/workflows/synthetic_test"),
        workflows=["synthetic_workflow_001.json", "synthetic_workflow_000.json"],
    ),
)


def ensure_paths() -> None:
    SCANS_ROOT.mkdir(parents=True, exist_ok=True)
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"RAG model checkpoint not found: {MODEL_PATH}")


def build_command(
    group: WorkflowGroup,
    temperature: float,
    greedy: float,
    top_k: int,
    epsilon: float,
    output_dir: Path,
    run_label: str,
) -> list[str]:
    base_cmd: list[str] = [
        sys.executable,
        "scripts/4_run_experiments.py",
        "--strategies",
        "HEFT",
        "MINMIN",
        "WASS_RAG_FULL",
        "--workflow-dir",
        str(group.workflow_dir),
        "--workflows",
        *group.workflows,
        "--seeds",
        *(str(seed) for seed in SEEDS),
        "--rag-model",
        str(MODEL_PATH),
        "--rag-temperature",
        f"{temperature}",
        "--rag-greedy-threshold",
        f"{greedy}",
        "--rag-sample-topk",
        str(top_k),
        "--rag-epsilon",
        f"{epsilon}",
        "--trace-log-dir",
        str(TRACE_ROOT),
        "--trace-run-label",
        run_label,
        "--output-dir",
        str(output_dir),
    ]
    return base_cmd


def iter_grid() -> Iterable[tuple[float, float, int, float]]:
    return itertools.product(TEMPERATURES, GREEDY_THRESHOLDS, TOP_K_VALUES, EPSILONS)


def run_grid() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for temperature, greedy, top_k, epsilon in iter_grid():
        for group in WORKFLOW_GROUPS:
            label = f"{group.name}_temp{temperature:.1f}_g{greedy:.2f}_top{top_k}_eps{epsilon:.2f}"
            safe_label = label.replace(".", "p")
            output_dir = SCANS_ROOT / safe_label
            output_dir.mkdir(parents=True, exist_ok=True)
            run_label = label
            summary_path = output_dir / "summary_results.csv"
            if summary_path.exists():
                print(f"\n=== Skipping completed grid point: {label} ===")
            else:
                cmd = build_command(group, temperature, greedy, top_k, epsilon, output_dir, run_label)
                print(f"\n=== Running grid point: {label} ===")
                print("Command:", " ".join(cmd))
                result = subprocess.run(cmd, check=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed for {label} with return code {result.returncode}")
            runs.append({
                "group": group.name,
                "temperature": temperature,
                "greedy_threshold": greedy,
                "top_k": top_k,
                "epsilon": epsilon,
                "output_dir": output_dir,
                "run_label": run_label,
            })
    return runs


def collate_results(runs: list[dict[str, object]]) -> None:
    records: list[dict[str, object]] = []
    for meta in runs:
        output_dir: Path = meta["output_dir"]  # type: ignore[assignment]
        detailed_path = output_dir / "detailed_results.csv"
        if not detailed_path.exists():
            print(f"⚠️  Skipping missing detailed results: {detailed_path}")
            continue
        df = pd.read_csv(detailed_path)
        for _, row in df.iterrows():
            records.append({
                "group": meta["group"],
                "temperature": meta["temperature"],
                "greedy_threshold": meta["greedy_threshold"],
                "top_k": meta["top_k"],
                "epsilon": meta["epsilon"],
                "scheduler": row["scheduler"],
                "workflow": row["workflow"],
                "makespan": row["makespan"],
                "status": row.get("status"),
                "seed": row.get("seed"),
            })
    if not records:
        print("❌ No records collected; skipping CSV generation.")
        return
    df = pd.DataFrame(records)
    df.sort_values(
        by=["group", "workflow", "scheduler", "temperature", "greedy_threshold", "top_k", "epsilon", "seed"],
        inplace=True,
    )
    df.to_csv(AGGREGATE_CSV, index=False)
    print(f"✅ Aggregated sensitivity results -> {AGGREGATE_CSV}")


def main() -> None:
    ensure_paths()
    runs = run_grid()
    collate_results(runs)


if __name__ == "__main__":
    main()
