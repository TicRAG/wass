# run_experiments.py
import os
import sys
from pathlib import Path

# --- 添加项目根目录到sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --------------------------------

from src.utils import WrenchExperimentRunner
from scripts.workflow_manager import WorkflowManager
from src.wrench_schedulers import FIFOScheduler, HEFTScheduler, WASS_DRL_Scheduler_Inference

def main():
    """主函数，用于运行所有最终的对比实验。"""
    print("🚀 [Milestone 3] Starting Final Experiments...")

    schedulers_to_compare = {
        "FIFO": FIFOScheduler,
        "HEFT": HEFTScheduler,
        "WASS_DRL": WASS_DRL_Scheduler_Inference
    }
    print(f"📊 Schedulers to compare: {list(schedulers_to_compare.keys())}")

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
    # 生成用于本次实验的、全新的测试工作流
    experiment_workflow_files = workflow_manager.generate_experiment_workflows()
    print("✅ Experiment workflows are ready.")

    print("\n[Step 2/3] Initializing and running WrenchExperimentRunner...")
    runner = WrenchExperimentRunner(schedulers=schedulers_to_compare, config=experiment_config)
    
    # --- 这是核心修改：由主脚本控制循环，确保使用正确的测试文件 ---
    all_results = []
    total_exps = len(schedulers_to_compare) * len(experiment_workflow_files) * experiment_config["repetitions"]
    exp_count = 0

    for name, sched_impl in schedulers_to_compare.items():
        # 遍历所有刚刚生成的实验工作流文件
        for wf_file in experiment_workflow_files:
            for rep in range(experiment_config["repetitions"]):
                exp_count += 1
                print(f"--- Running Experiment [{exp_count}/{total_exps}]: Scheduler={name}, Workflow={Path(wf_file).name}, Repetition={rep+1} ---")
                
                # 直接调用底层的、更可控的 _run_single_simulation 方法
                result = runner._run_single_simulation(
                    scheduler_name=name,
                    scheduler_impl=sched_impl,
                    workflow_file=str(wf_file)
                )
                all_results.append(result)
    # --- 修改结束 ---

    print("✅ All simulations completed.")

    print("\n[Step 3/3] Analyzing and saving results...")
    if all_results:
        runner.analyze_results(all_results)
    else:
        print("❌ No results were generated from the simulations.")

    print("\n🎉 [Milestone 3] Final Experiments Completed Successfully! 🎉")

if __name__ == "__main__":
    main()