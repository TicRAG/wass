# run_experiments.py
import os
import sys
from pathlib import Path

# --- 路径修正 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------

from src.utils import WrenchExperimentRunner
from scripts.workflow_manager import WorkflowManager
# --- 核心修改：导入所有需要对比的调度器 ---
from src.wrench_schedulers import (
    FIFOScheduler, 
    HEFTScheduler, 
    WASS_DRL_Scheduler_Inference, 
    WASS_DRL_NO_RAG_Scheduler_Inference
)

def main():
    """主函数，用于运行所有最终的对比实验。"""
    print("🚀 [Milestone 3] Starting Final Experiments...")

    # --- 核心修改：更新调度器字典 ---
    schedulers_to_compare = {
        "FIFO": FIFOScheduler,
        "HEFT": HEFTScheduler,
        "WASS_DRL": WASS_DRL_NO_RAG_Scheduler_Inference, # 没有RAG的模型
        "WASS_RAG": WASS_DRL_Scheduler_Inference      # 有RAG的模型 (我们之前的WASS_DRL)
    }
    print(f"📊 Schedulers to compare: {list(schedulers_to_compare.keys())}")
    # --- 修改结束 ---

    experiment_config = {
        "platform_file": "configs/test_platform.xml",
        "workflow_dir": "data/workflows",
        "workflow_sizes": [20, 50, 100],
        "repetitions": 5,
        "output_dir": "results/final_experiments"
    }
    print(f"📝 Experiment Config: Testing on sizes {experiment_config['workflow_sizes']} with {experiment_config['repetitions']} repetitions.")

    print("\n[Step 1/3] Generating experiment workflows...")
    workflow_manager = WorkflowManager(config_path="configs/workflow_config.yaml")
    experiment_workflow_files = workflow_manager.generate_experiment_workflows()
    print("✅ Experiment workflows are ready.")

    print("\n[Step 2/3] Initializing and running WrenchExperimentRunner...")
    runner = WrenchExperimentRunner(schedulers=schedulers_to_compare, config=experiment_config)
    
    all_results = []
    # (这部分循环逻辑保持不变)
    for name, sched_impl in schedulers_to_compare.items():
        for wf_file in experiment_workflow_files:
            for rep in range(experiment_config["repetitions"]):
                print(f"--- Running Experiment: Scheduler={name}, Workflow={Path(wf_file).name}, Rep={rep+1} ---")
                result = runner._run_single_simulation(
                    scheduler_name=name,
                    scheduler_impl=sched_impl,
                    workflow_file=str(wf_file)
                )
                all_results.append(result)

    print("✅ All simulations completed.")

    print("\n[Step 3/3] Analyzing and saving results...")
    if all_results:
        runner.analyze_results(all_results)
    else:
        print("❌ No results were generated.")

    print("\n🎉 [Milestone 3] Final Experiments Completed Successfully! 🎉")

if __name__ == "__main__":
    main()