# run_experiments.py
import argparse
import os
import sys
from datetime import datetime, UTC
from functools import partial
from pathlib import Path

# --- 路径修正 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import torch

from src.simulation.experiment_runner import WrenchExperimentRunner
from src.simulation.schedulers import FIFOScheduler, HEFTScheduler, MinMinScheduler, WASS_DRL_Scheduler_Inference
from src.drl.gnn_encoder import DecoupledGNNEncoder
from src.workflows.manager import WorkflowManager
from src.rag.teacher import KnowledgeBase, KnowledgeableTeacher
from src.utils.config import load_training_config


STRATEGY_DEFINITIONS = {
    "FIFO": {
        "label": "FIFO",
        "factory": lambda args: FIFOScheduler,
    },
    "WASS_RAG_FULL": {
        "label": "WASS-RAG (Full)",
        "factory": lambda args: partial(
            WASS_DRL_Scheduler_Inference,
            variant="rag",
            model_path=args.rag_model,
            stochastic_tie_break=args.stochastic_tie_break,
            temperature=args.rag_temperature,
            greedy_threshold=args.rag_greedy_threshold,
            epsilon=args.rag_epsilon,
            sample_top_k=args.rag_sample_topk,
        ),
    },
    "WASS_DRL_VANILLA": {
        "label": "WASS-DRL (Vanilla)",
        "factory": lambda args: partial(
            WASS_DRL_Scheduler_Inference,
            variant="drl",
            model_path=args.drl_model,
            stochastic_tie_break=args.stochastic_tie_break,
            temperature=args.rag_temperature,
            greedy_threshold=args.rag_greedy_threshold,
            epsilon=args.rag_epsilon,
            sample_top_k=args.rag_sample_topk,
        ),
    },
    "WASS_RAG_HEFT": {
        "label": "WASS-RAG (HEFT-only)",
        "factory": lambda args: partial(
            WASS_DRL_Scheduler_Inference,
            variant="rag",
            model_path=args.rag_heft_model or args.rag_model,
            stochastic_tie_break=args.stochastic_tie_break,
            temperature=args.rag_temperature,
            greedy_threshold=args.rag_greedy_threshold,
            epsilon=args.rag_epsilon,
            sample_top_k=args.rag_sample_topk,
        ),
    },
    "HEFT": {
        "label": "HEFT",
        "factory": lambda args: partial(HEFTScheduler, noise_sigma=args.heft_noise_sigma),
    },
    "MINMIN": {
        "label": "MIN-MIN",
        "factory": lambda args: MinMinScheduler,
    },
}

DEFAULT_STRATEGIES = ["FIFO", "HEFT", "MINMIN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WASS-RAG comparison experiments across multiple strategies.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGY_DEFINITIONS.keys()),
    default=DEFAULT_STRATEGIES,
        help="Subset of strategies to execute. Defaults to all.",
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        help="Optional list of workflow basenames (with or without .json) to filter.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds for stochastic schedulers/agents (default: 5 seeds).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Additional repetitions per (strategy, workflow, seed).",
    )
    parser.add_argument(
        "--include-aug",
        action="store_true",
        help="Include augmented training workflows when available.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_experiments",
        help="Directory to write detailed and summary CSV outputs.",
    )
    parser.add_argument("--rag-model", dest="rag_model", help="Override checkpoint for WASS-RAG (Full) strategy.")
    parser.add_argument("--drl-model", dest="drl_model", help="Override checkpoint for WASS-DRL (Vanilla) strategy.")
    parser.add_argument(
        "--rag-heft-model",
        dest="rag_heft_model",
        help="Checkpoint for WASS-RAG (HEFT-only); defaults to --rag-model when omitted.",
    )
    parser.add_argument(
        "--workflow-dir",
        default="data/workflows/experiment",
        help="Directory containing experiment workflows (default: data/workflows/experiment).",
    )
    parser.add_argument(
        "--minmin-comm-scale",
        type=float,
        default=1.0,
        help="Scaling factor for communication penalties inside Min-Min scheduler (default: 1.0).",
    )
    parser.add_argument(
        "--minmin-remote-penalty",
        type=float,
        default=0.0,
        help="Fixed penalty (seconds) added for each remote parent edge in Min-Min when transfer size is zero or bandwidth unknown (default: 0.0).",
    )
    parser.add_argument(
        "--platform-key",
        default=None,
        help="Override platform configuration key defined in configs/workflow_config.yaml (e.g., medium, tight).",
    )
    parser.add_argument(
        "--heft-noise-sigma",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian multiplicative noise applied to HEFT's predicted task durations (default: 0.0, i.e., perfect estimates).",
    )
    parser.add_argument(
        "--minmin-balance-weight",
        type=float,
        default=0.0,
        help="Penalty weight applied per in-flight assignment on a host in Min-Min (encourages load balancing).",
    )
    parser.add_argument(
        "--minmin-availability-weight",
        type=float,
        default=0.0,
        help="Penalty weight applied to a host's queued time beyond the current simulation time in Min-Min.",
    )
    parser.add_argument(
        "--min-host-speed",
        type=float,
        default=0.0,
        help="Exclude compute hosts whose reported flop-rate (in Gf/s) is below this threshold (default: keep all hosts).",
    )
    parser.add_argument(
        "--trace-log-dir",
        default=None,
        help="If set, emit teacher trace JSONL logs for WASS-RAG (Full) runs into this directory.",
    )
    parser.add_argument(
        "--trace-run-label",
        default=None,
        help="Optional label recorded in trace context; defaults to the output directory name.",
    )
    parser.add_argument(
        "--trace-max-neighbors",
        type=int,
        default=None,
        help="Limit number of neighbor entries stored per trace event (default: use teacher configuration).",
    )
    parser.add_argument(
        "--stochastic-tie-break",
        action="store_true",
        help="Sample actions from the WASS DRL policy instead of argmax to inject stochastic host tie-breaking.",
    )
    parser.add_argument(
        "--rag-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature applied to WASS-RAG action probabilities (values < 1 tighten distribution).",
    )
    parser.add_argument(
        "--rag-greedy-threshold",
        type=float,
        default=1.1,
        help="If the max action probability exceeds this threshold, force greedy selection (default disables).",
    )
    parser.add_argument(
        "--rag-epsilon",
        type=float,
        default=0.0,
        help="Epsilon-greedy mixing weight injected after temperature/top-k adjustments (default 0).",
    )
    parser.add_argument(
        "--rag-sample-topk",
        type=int,
        default=None,
        help="Limit stochastic sampling to the top-k hosts before renormalization (default: use all hosts).",
    )
    return parser.parse_args()


def resolve_workflows(runner: WrenchExperimentRunner, args: argparse.Namespace) -> list[str]:
    workflows_dir = Path(args.workflow_dir)
    all_workflows = runner._load_workflows(workflows_dir)  # pylint: disable=protected-access
    if not all_workflows:
        print(f"❌ No experiment workflows found in {workflows_dir}. Ensure JSON files are present.")
        return []
    if not args.workflows:
        return all_workflows
    normalized = {Path(wf).stem: wf for wf in all_workflows}
    selected = []
    for name in args.workflows:
        stem = Path(name).stem
        if stem in normalized:
            selected.append(normalized[stem])
        else:
            print(f"⚠️ Requested workflow '{name}' not found; skipping.")
    if not selected:
        print("❌ No workflows matched the provided filters.")
    return selected


def main():
    args = parse_args()
    print("🚀 [P1] Starting comparison experiments...")

    workflow_manager = WorkflowManager(config_path="configs/workflow_config.yaml")
    platform_file = workflow_manager.get_platform_file(key=args.platform_key)
    experiment_config = {
        "platform_file": platform_file,
        "workflow_dir": args.workflow_dir,
        "workflow_sizes": [],
        "repetitions": args.repetitions,
        "output_dir": args.output_dir,
        "random_seeds": args.seeds,
        "include_aug": args.include_aug,
        "min_host_speed": args.min_host_speed,
    }

    trace_dir: Path | None = None
    trace_run_label: str | None = None
    feature_scaler = None
    teacher_resources: dict[str, object] | None = None
    if args.trace_log_dir:
        trace_dir = Path(args.trace_log_dir).expanduser()
        trace_dir.mkdir(parents=True, exist_ok=True)
        training_cfg = load_training_config()
        common_cfg = training_cfg.get("common", {})
        rag_cfg = training_cfg.get("rag_training", {})
        gnn_cfg = common_cfg.get("gnn", {})
        gnn_in = int(gnn_cfg.get("in_channels", 4))
        gnn_hidden = int(gnn_cfg.get("hidden_channels", 64))
        gnn_out = int(gnn_cfg.get("out_channels", 32))
        teacher_cfg = rag_cfg.get("teacher", {})
        knowledge_base = KnowledgeBase(dimension=gnn_out)
        dual_gnn = DecoupledGNNEncoder(gnn_in, gnn_hidden, gnn_out)
        gnn_seed_path = Path("models/saved_models/gnn_encoder_kb.pth")
        if gnn_seed_path.exists():
            try:
                state_dict = torch.load(gnn_seed_path, map_location=torch.device("cpu"))
                dual_gnn.policy_encoder.load_state_dict(state_dict)
                dual_gnn.sync_retrieval_encoder()
                print(f"[Trace] Loaded retrieval encoder weights from {gnn_seed_path}")
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"⚠️ Failed to load retrieval encoder weights ({exc}); continuing with random init.")
        else:
            print(f"⚠️ Retrieval encoder checkpoint not found at {gnn_seed_path}; using random initialization.")
        dual_gnn.freeze_retrieval_encoder()
        rag_encoder = dual_gnn.retrieval_encoder
        scaler_path = Path("models/saved_models/feature_scaler.joblib")
        if scaler_path.exists():
            try:
                feature_scaler = joblib.load(scaler_path)
                print(f"[Trace] Loaded feature scaler from {scaler_path}")
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"⚠️ Failed to load feature scaler ({exc}); continuing without scaling.")
                feature_scaler = None
        else:
            print(f"⚠️ Feature scaler not found at {scaler_path}; continuing without scaling.")
        trace_run_label = args.trace_run_label or Path(args.output_dir).name or "wass_rag"
        teacher_resources = {
            "kb": knowledge_base,
            "rag_encoder": rag_encoder,
            "teacher_cfg": teacher_cfg,
            "state_dim": gnn_out,
        }

    strategy_factories = {}
    for key in args.strategies:
        definition = STRATEGY_DEFINITIONS[key]
        # Inject Min-Min specific penalty parameters
        if key == "MINMIN":
            factory = definition["factory"](args)
            def wrapped_factory(*f_args, **f_kwargs):
                return factory(
                    *f_args,
                    communication_scale=args.minmin_comm_scale,
                    default_remote_penalty=args.minmin_remote_penalty,
                    balance_weight=args.minmin_balance_weight,
                    availability_weight=args.minmin_availability_weight,
                    **f_kwargs,
                )
            strategy_factories[definition["label"]] = wrapped_factory
        else:
            strategy_factories[definition["label"]] = definition["factory"](args)
    print(f"📊 Strategies to compare: {list(strategy_factories.keys())}")

    runner = WrenchExperimentRunner(schedulers=strategy_factories, config=experiment_config)
    workflow_files = resolve_workflows(runner, args)
    if not workflow_files:
        return
    print(f"✅ Loaded {len(workflow_files)} workflows for experiments.")

    all_results = []
    for strategy_label, sched_impl in strategy_factories.items():
        for workflow_file in workflow_files:
            for seed in args.seeds:
                for rep in range(args.repetitions):
                    print(
                        f"--- Running Experiment: Strategy={strategy_label}, Workflow={Path(workflow_file).name}, Seed={seed}, Rep={rep + 1} ---"
                    )
                    extra_kwargs: dict[str, object] = {
                        "seed": seed,
                        "strategy_label": strategy_label,
                        "repeat_index": rep,
                    }
                    if feature_scaler is not None:
                        extra_kwargs["feature_scaler"] = feature_scaler
                    if trace_dir is not None and teacher_resources is not None and strategy_label == "WASS-RAG (Full)":
                        workflow_stem = Path(workflow_file).stem
                        run_subdir = trace_dir / workflow_stem
                        run_subdir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
                        trace_name = f"{workflow_stem}_seed{seed}_rep{rep}_{timestamp}.jsonl"
                        trace_path = run_subdir / trace_name
                        teacher = KnowledgeableTeacher(
                            state_dim=int(teacher_resources["state_dim"]),
                            knowledge_base=teacher_resources["kb"],
                            gnn_encoder=teacher_resources["rag_encoder"],
                            reward_config=teacher_resources["teacher_cfg"],
                        )
                        teacher.enable_trace_logging(trace_path, max_neighbors=args.trace_max_neighbors)
                        teacher.set_trace_context(
                            run_label=trace_run_label,
                            workflow_file=Path(workflow_file).name,
                            seed=seed,
                            repeat=rep,
                            strategy=strategy_label,
                            trace_path=str(trace_path),
                        )
                        extra_kwargs.update({
                            "teacher": teacher,
                            "run_label": trace_run_label,
                            "trace_output_path": str(trace_path),
                        })
                    result = runner._run_single_simulation(
                        scheduler_name=strategy_label,
                        scheduler_impl=sched_impl,
                        workflow_file=workflow_file,
                        seed=seed,
                        extra_kwargs=extra_kwargs,
                    )
                    all_results.append(result)

    print("✅ All simulations completed.")
    print("\n[Step] Analyzing and saving results...")
    if all_results:
        runner.analyze_results(all_results)
    else:
        print("❌ No results were generated.")

    print("\n🎉 [P1] Experiment run finished! 🎉")


if __name__ == "__main__":
    main()