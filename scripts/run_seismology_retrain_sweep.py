#!/usr/bin/env python3
"""Coordinate a focused hyperparameter sweep for seismology-only RAG retraining."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_RAG_MULTIPLIERS: Sequence[float] = (1.5, 2.25)
DEFAULT_TEACHER_LAMBDAS: Sequence[float] = (0.75, 0.9)
DEFAULT_TEACHER_TEMPS: Sequence[float] = (0.05, 0.08)
DEFAULT_TEACHER_TOPK: Sequence[int] = (4, 6)
DEFAULT_SEEDS: Sequence[int] = (111, 222)


@dataclass(frozen=True)
class SweepPoint:
    rag_multiplier: float
    teacher_lambda: float
    teacher_temperature: float
    teacher_top_k: int
    seed: int


def _float_sequence(value: Sequence[float] | None, default: Sequence[float]) -> List[float]:
    if value is None or len(value) == 0:
        return [float(v) for v in default]
    return [float(v) for v in value]


def _int_sequence(value: Sequence[int] | None, default: Sequence[int]) -> List[int]:
    if value is None or len(value) == 0:
        return [int(v) for v in default]
    return [int(v) for v in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seismology-only retraining sweep to improve rag_mean.")
    parser.add_argument("--rag-multipliers", type=float, nargs="*", help="Dense reward multipliers to test (default: 1.5 2.25).")
    parser.add_argument("--teacher-lambdas", type=float, nargs="*", help="Teacher lambda scaling factors (default: 0.75 0.9).")
    parser.add_argument("--teacher-temperatures", type=float, nargs="*", help="Teacher softmax temperatures (default: 0.05 0.08).")
    parser.add_argument("--teacher-top-k", type=int, nargs="*", help="Teacher neighbor counts to evaluate (default: 4 6).")
    parser.add_argument("--seeds", type=int, nargs="*", help="Random seeds for PPO (default: 111 222).")
    parser.add_argument("--max-episodes", type=int, default=240, help="Episode cap for each training run (default: 240).")
    parser.add_argument("--log-root", default="results/seismology_retrain", help="Root directory for sweep outputs.")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint path for warm-starting each run.")
    parser.add_argument("--include-aug", action="store_true", help="Include augmented workflows alongside seismology family.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def build_grid(args: argparse.Namespace) -> Iterable[SweepPoint]:
    rag_vals = _float_sequence(args.rag_multipliers, DEFAULT_RAG_MULTIPLIERS)
    lambda_vals = _float_sequence(args.teacher_lambdas, DEFAULT_TEACHER_LAMBDAS)
    temp_vals = _float_sequence(args.teacher_temperatures, DEFAULT_TEACHER_TEMPS)
    topk_vals = _int_sequence(args.teacher_top_k, DEFAULT_TEACHER_TOPK)
    seed_vals = _int_sequence(args.seeds, DEFAULT_SEEDS)
    for rag_val, lambda_val, temp_val, topk_val, seed in itertools.product(
        rag_vals, lambda_vals, temp_vals, topk_vals, seed_vals
    ):
        yield SweepPoint(
            rag_multiplier=rag_val,
            teacher_lambda=lambda_val,
            teacher_temperature=temp_val,
            teacher_top_k=topk_val,
            seed=seed,
        )


def safe_slug(value: str) -> str:
    slug = value.replace(" ", "_")
    slug = slug.replace(".", "p")
    slug = slug.replace("-", "m")
    return slug


def write_metadata(run_dir: Path, payload: dict) -> None:
    meta_path = run_dir / "sweep_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def command_for(point: SweepPoint, args: argparse.Namespace, run_label: str, log_dir: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "scripts/2_train_rag_agent.py",
        "--family-filter",
        "seismology",
        "--run_label",
        run_label,
        "--log_dir",
        str(log_dir),
        "--rag-multiplier",
        f"{point.rag_multiplier}",
        "--teacher-lambda",
        f"{point.teacher_lambda}",
        "--teacher-temperature",
        f"{point.teacher_temperature}",
        "--teacher-top-k",
        str(point.teacher_top_k),
        "--seed",
        str(point.seed),
    ]
    if args.max_episodes:
        cmd.extend(["--max_episodes", str(args.max_episodes)])
    if args.include_aug:
        cmd.append("--include_aug")
    if args.resume_from:
        cmd.extend(["--resume-from", args.resume_from])
    return cmd


def main() -> None:
    args = parse_args()
    sweep_started_at = datetime.utcnow()
    sweep_root = Path(args.log_root) / sweep_started_at.strftime("%Y%m%dT%H%M%SZ")
    sweep_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[dict] = []

    for point in build_grid(args):
        combo_label = (
            f"rag{point.rag_multiplier:.2f}_"
            f"lam{point.teacher_lambda:.2f}_"
            f"temp{point.teacher_temperature:.2f}_"
            f"top{point.teacher_top_k}_"
            f"seed{point.seed}"
        )
        safe_label = safe_slug(combo_label)
        run_dir = sweep_root / safe_label
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        run_label = f"seismo_{combo_label}"
        cmd = command_for(point, args, run_label, log_dir)
        print("\n=== Running", combo_label, "===")
        print("Command:", " ".join(cmd))
        status = "pending"
        return_code = None
        started_at = datetime.utcnow().isoformat(timespec="seconds")
        if args.dry_run:
            status = "dry-run"
        else:
            try:
                result = subprocess.run(cmd, check=True)
                status = "success"
                return_code = result.returncode
            except subprocess.CalledProcessError as exc:
                status = "failed"
                return_code = exc.returncode
                print(f"[Sweep] Command failed for {combo_label} with return code {exc.returncode}")
        metadata_payload = {
            "run_label": run_label,
            "combo": combo_label,
            "command": cmd,
            "status": status,
            "started_at": started_at,
            "rag_multiplier": point.rag_multiplier,
            "teacher_lambda": point.teacher_lambda,
            "teacher_temperature": point.teacher_temperature,
            "teacher_top_k": point.teacher_top_k,
            "seed": point.seed,
            "max_episodes": args.max_episodes,
            "include_aug": bool(args.include_aug),
            "resume_from": args.resume_from,
            "log_dir": str(log_dir),
            "return_code": return_code,
        }
        write_metadata(run_dir, metadata_payload)
        manifest_rows.append(metadata_payload)

    manifest_path = sweep_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    print("\nSweep completed. Manifest ->", manifest_path)


if __name__ == "__main__":
    main()
