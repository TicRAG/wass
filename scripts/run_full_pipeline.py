#!/usr/bin/env python3
"""End-to-end driver for the WASS-RAG experimental pipeline.

This helper orchestrates the following (optional) stages:

1. Synthetic workflow generation (scripts/generate_synthetic_workflows.py)
2. Knowledge-base seeding (scripts/1_seed_knowledge_base.py)
3. Scheduler training for DRL and RAG variants
   (scripts/3_train_drl_agent.py and scripts/2_train_rag_agent.py)
4. Multi-strategy experiment execution (scripts/4_run_experiments.py)
5. Result aggregation & plotting (analysis/plot_results.py)

Each stage can be skipped with dedicated CLI flags. By default, only the
experiment + plotting stages run, as those are the quickest to iterate.

Example usage (full run with default parameters)::

    ./scripts/run_full_pipeline.py \
        --output-dir results/extreme_top3_noise01 \
        --min-host-speed 100 \
        --heft-noise-sigma 0.1 \
        --seeds 0 1 2 3 4 \
        --strategies FIFO HEFT MINMIN WASS_RAG_FULL WASS_DRL_VANILLA

To include the long-running training stages, add ``--include-training``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = sys.executable or "/usr/bin/env python3"
RESULTS_ROOT = PROJECT_ROOT / "results"
FINAL_RESULTS_DIR = RESULTS_ROOT / "final_experiments"
CHART_DIR = PROJECT_ROOT / "charts"


class PipelineError(RuntimeError):
    """Raised when any pipeline stage fails."""


def run_command(command: List[str], *, dry_run: bool, cwd: Path, description: str) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n▶ {description}\n   $ {printable}")
    if dry_run:
        print("   (dry-run: skipped execution)")
        return
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise PipelineError(f"Command failed ({result.returncode}): {printable}")


def copy_results(output_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        print("Dry-run: skipping result copy to results/final_experiments")
        return
    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ["detailed_results.csv", "summary_results.csv"]:
        src = output_dir / filename
        if not src.exists():
            print(f"⚠️ Not found: {src} (skipping copy)")
            continue
        dst = FINAL_RESULTS_DIR / filename
        dst.write_bytes(src.read_bytes())
        print(f"📁 Copied {src} -> {dst}")


def write_metadata(output_dir: Path, metadata: dict, *, dry_run: bool) -> None:
    if dry_run:
        print("Dry-run: skipping metadata write")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "pipeline_config.json"
    config_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"📝 Saved pipeline metadata -> {config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full WASS-RAG workflow pipeline.")
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Python interpreter to use for subprocess calls.")
    parser.add_argument("--output-dir", default="results/extreme_top3_noise01", help="Directory for experiment outputs.")
    parser.add_argument("--strategies", nargs="+", default=[
        "FIFO",
        "HEFT",
        "MINMIN",
        "WASS_RAG_FULL",
        "WASS_DRL_VANILLA",
    ], help="Strategies to evaluate (passed through to scripts/4_run_experiments.py)")
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"], help="Random seeds for experiments.")
    parser.add_argument("--workflow-dir", default="data/workflows/experiment", help="Workflow directory for experiments.")
    parser.add_argument("--min-host-speed", type=float, default=0.0, help="Minimum host speed (Gf/s) filter passed to experiment runner.")
    parser.add_argument("--heft-noise-sigma", type=float, default=0.1, help="Noise sigma forwarded to HEFT scheduler.")
    parser.add_argument("--minmin-comm-scale", type=float, default=15.0, help="Communication scale for Min-Min scheduler.")
    parser.add_argument("--minmin-remote-penalty", type=float, default=20.0, help="Remote penalty for Min-Min scheduler.")
    parser.add_argument("--minmin-balance-weight", type=float, default=200.0, help="Balance weight for Min-Min scheduler.")
    parser.add_argument("--minmin-availability-weight", type=float, default=50.0, help="Availability weight for Min-Min scheduler.")
    parser.add_argument("--rag-model", default="models/saved_models/drl_agent.pth", help="Checkpoint for WASS-RAG (Full).")
    parser.add_argument("--drl-model", default="models/saved_models/drl_agent_no_rag.pth", help="Checkpoint for WASS-DRL (Vanilla).")
    parser.add_argument("--platform-key", default="extreme_hetero", help="Platform key (configs/workflow_config.yaml) to use.")
    parser.add_argument("--include-training", action="store_true", help="If set, run DRL/RAG training stages before experiments.")
    parser.add_argument("--rag-trace-log-dir", default=None, help="If set, pass directory to 2_train_rag_agent.py for interpretability traces.")
    parser.add_argument("--skip-workflow-generation", action="store_true", help="Skip synthetic workflow generation stage.")
    parser.add_argument("--skip-knowledge-seeding", action="store_true", help="Skip knowledge-base seeding stage.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--tag", default=None, help="Optional tag stored in pipeline metadata for bookkeeping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = args.python
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    run_label = args.tag or output_dir.name
    rag_trace_dir = None
    if args.rag_trace_log_dir:
        candidate = Path(args.rag_trace_log_dir)
        rag_trace_dir = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    timestamp = datetime.now(timezone.utc).isoformat()

    command_log: list[dict] = []

    try:
        if not args.skip_workflow_generation:
            cmd = [python, "scripts/generate_synthetic_workflows.py"]
            run_command(cmd, dry_run=args.dry_run, cwd=PROJECT_ROOT, description="Generate synthetic workflows")
            command_log.append({"stage": "workflow_generation", "command": cmd})

        if not args.skip_knowledge_seeding:
            cmd = [python, "scripts/1_seed_knowledge_base.py"]
            run_command(cmd, dry_run=args.dry_run, cwd=PROJECT_ROOT, description="Seed scheduling knowledge base")
            command_log.append({"stage": "knowledge_seeding", "command": cmd})

        if args.rag_trace_log_dir and not args.include_training:
            print("⚠️ Provided --rag-trace-log-dir but training is skipped; no trace logs will be produced.")

        if args.include_training:
            train_drl_cmd = [python, "scripts/3_train_drl_agent.py"]
            if run_label:
                train_drl_cmd.extend(["--run_label", run_label])
            run_command(train_drl_cmd, dry_run=args.dry_run, cwd=PROJECT_ROOT, description="Train WASS-DRL (Vanilla) agent")
            command_log.append({"stage": "train_drl", "command": train_drl_cmd})

            train_rag_cmd = [python, "scripts/2_train_rag_agent.py"]
            if run_label:
                train_rag_cmd.extend(["--run_label", run_label])
            if rag_trace_dir is not None:
                train_rag_cmd.extend(["--trace_log_dir", str(rag_trace_dir)])
            run_command(train_rag_cmd, dry_run=args.dry_run, cwd=PROJECT_ROOT, description="Train WASS-RAG (Full) agent")
            command_log.append({"stage": "train_rag", "command": train_rag_cmd})
        else:
            print("ℹ️ Training stages skipped (use --include-training to enable).")

        experiment_cmd = [
            python,
            "scripts/4_run_experiments.py",
            "--strategies",
            *args.strategies,
            "--workflow-dir",
            args.workflow_dir,
            "--output-dir",
            str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
            "--heft-noise-sigma",
            str(args.heft_noise_sigma),
            "--minmin-comm-scale",
            str(args.minmin_comm_scale),
            "--minmin-remote-penalty",
            str(args.minmin_remote_penalty),
            "--minmin-balance-weight",
            str(args.minmin_balance_weight),
            "--minmin-availability-weight",
            str(args.minmin_availability_weight),
            "--platform-key",
            args.platform_key,
            "--rag-model",
            args.rag_model,
            "--drl-model",
            args.drl_model,
            "--seeds",
            *args.seeds,
        ]
        if args.min_host_speed > 0:
            experiment_cmd.extend(["--min-host-speed", str(args.min_host_speed)])
        run_command(experiment_cmd, dry_run=args.dry_run, cwd=PROJECT_ROOT, description="Run comparison experiments")
        command_log.append({"stage": "experiments", "command": experiment_cmd})

        copy_results(output_dir, dry_run=args.dry_run)

        plot_cmd = [python, "analysis/plot_results.py"]
        run_command(plot_cmd, dry_run=args.dry_run, cwd=PROJECT_ROOT, description="Generate analysis charts & summaries")
        command_log.append({"stage": "plotting", "command": plot_cmd})

    except PipelineError as exc:  # pragma: no cover
        print(f"❌ Pipeline failed: {exc}")
        raise SystemExit(1) from exc

    metadata = {
        "timestamp": timestamp,
        "tag": args.tag,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
        "strategies": args.strategies,
        "seeds": args.seeds,
        "workflow_dir": args.workflow_dir,
    "min_host_speed": args.min_host_speed,
        "heft_noise_sigma": args.heft_noise_sigma,
        "minmin_comm_scale": args.minmin_comm_scale,
        "minmin_remote_penalty": args.minmin_remote_penalty,
        "minmin_balance_weight": args.minmin_balance_weight,
        "minmin_availability_weight": args.minmin_availability_weight,
        "rag_model": args.rag_model,
        "drl_model": args.drl_model,
        "platform_key": args.platform_key,
        "run_label": run_label,
        "include_training": args.include_training,
        "skip_workflow_generation": args.skip_workflow_generation,
        "skip_knowledge_seeding": args.skip_knowledge_seeding,
        "dry_run": args.dry_run,
        "python": args.python,
        "rag_trace_log_dir": str(rag_trace_dir) if rag_trace_dir is not None else None,
        "commands": command_log,
    }
    write_metadata(output_dir, metadata, dry_run=args.dry_run)

    print("\n✅ Pipeline completed successfully")
    print(f"   Results directory : {output_dir}")
    print(f"   Charts directory  : {CHART_DIR.resolve()}")
    if not args.dry_run:
        print(f"   Final summaries   : {FINAL_RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
