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
# 导入所有需要对比的调度器
from src.wrench_schedulers import FIFOScheduler, HEFTScheduler, WASS_DRL_Scheduler_Inference

def main():
    """主函数，用于运行所有最终的对比实验。"""
    print("🚀 [Milestone 3] Starting Final Experiments...")

    # 1. 定义需要对比的调度器
    #    我们在这里直接使用类，而不是字符串，这样更清晰
    schedulers_to_compare = {
        "FIFO": FIFOScheduler,
        "HEFT": HEFTScheduler,
        "WASS_DRL": WASS_DRL_Scheduler_Inference # 这就是我们训练好的智能体！
    }
    print(f"📊 Schedulers to compare: {list(schedulers_to_compare.keys())}")

    # 2. 定义实验配置
    #    我们将使用 'experiment_workflows' 部分的配置
    experiment_config = {
        "platform_file": "configs/test_platform.xml",
        "workflow_dir": "data/workflows",
        "workflow_sizes": [20, 50, 100], # 在这些规模上进行测试
        "repetitions": 5, # 每个实验重复5次以获得更稳定的结果
        "output_dir": "results/final_experiments"
    }
    print(f"📝 Experiment Config: Testing on sizes {experiment_config['workflow_sizes']} with {experiment_config['repetitions']} repetitions.")

    # 3. 生成用于实验的工作流 (确保我们有干净的测试集)
    print("\n[Step 1/3] Generating experiment workflows...")
    workflow_manager = WorkflowManager(config_path="configs/workflow_config.yaml")
    # 清理旧的实验工作流（可选，但推荐）
    # import shutil
    # if Path(experiment_config["workflow_dir"]).exists():
    #     shutil.rmtree(experiment_config["workflow_dir"])
    workflow_manager.generate_experiment_workflows()
    print("✅ Experiment workflows are ready.")

    # 4. 创建并运行实验
    print("\n[Step 2/3] Initializing and running WrenchExperimentRunner...")
    runner = WrenchExperimentRunner(schedulers=schedulers_to_compare, config=experiment_config)
    all_results = runner.run_all()
    print("✅ All simulations completed.")

    # 5. 分析并保存结果
    print("\n[Step 3/3] Analyzing and saving results...")
    if all_results:
        runner.analyze_results(all_results)
    else:
        print("❌ No results were generated from the simulations.")

    print("\n🎉 [Milestone 3] Final Experiments Completed Successfully! 🎉")

if __name__ == "__main__":
    main()