# run_experiments.py
import argparse
import os
import sys
from functools import partial
from pathlib import Path

# --- 路径修正 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.experiment_runner import WrenchExperimentRunner
from src.simulation.schedulers import HEFTScheduler, MinMinScheduler, WASS_DRL_Scheduler_Inference
from src.workflows.manager import WorkflowManager


STRATEGY_DEFINITIONS = {
    "WASS_RAG_FULL": {
        "label": "WASS-RAG (Full)",
        "factory": lambda args: partial(WASS_DRL_Scheduler_Inference, variant="rag", model_path=args.rag_model),
    },
    "WASS_DRL_VANILLA": {
        "label": "WASS-DRL (Vanilla)",
        "factory": lambda args: partial(WASS_DRL_Scheduler_Inference, variant="drl", model_path=args.drl_model),
    },
    "WASS_RAG_HEFT": {
        "label": "WASS-RAG (HEFT-only)",
        "factory": lambda args: partial(WASS_DRL_Scheduler_Inference, variant="rag", model_path=args.rag_heft_model or args.rag_model),
    },
    "HEFT": {
        "label": "HEFT",
        "factory": lambda args: HEFTScheduler,
    },
    "MINMIN": {
        "label": "MIN-MIN",
        "factory": lambda args: MinMinScheduler,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WASS-RAG comparison experiments across multiple strategies.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGY_DEFINITIONS.keys()),
        default=list(STRATEGY_DEFINITIONS.keys()),
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
    platform_file = workflow_manager.get_platform_file()
    experiment_config = {
        "platform_file": platform_file,
        "workflow_dir": args.workflow_dir,
        "workflow_sizes": [],
        "repetitions": args.repetitions,
        "output_dir": args.output_dir,
        "random_seeds": args.seeds,
        "include_aug": args.include_aug,
    }

    strategy_factories = {}
    for key in args.strategies:
        definition = STRATEGY_DEFINITIONS[key]
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
                    result = runner._run_single_simulation(
                        scheduler_name=strategy_label,
                        scheduler_impl=sched_impl,
                        workflow_file=workflow_file,
                        seed=seed,
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