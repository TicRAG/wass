# scripts/seed_knowledge_base.py
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

# --- 修正导入路径问题 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------

# 导入我们的模块
from workflow_manager import WorkflowManager
from src.drl.gnn_encoder import GNNEncoder
from src.drl.knowledge_teacher import KnowledgeBase
from src.drl.utils import workflow_json_to_pyg_data
from src.wrench_schedulers import RecordingHEFTScheduler
from src.utils import WrenchExperimentRunner

# --- 配置区 ---
GNN_IN_CHANNELS = 1
GNN_HIDDEN_CHANNELS = 64
GNN_OUT_CHANNELS = 32
KB_DIMENSION = GNN_OUT_CHANNELS
PLATFORM_FILE = "configs/test_platform.xml"
WORKFLOW_CONFIG_FILE = "configs/workflow_config.yaml"

# --- 主逻辑 ---
def main():
    """主函数，执行知识库生成流程"""
    print("🚀 [Phase 1] Starting Knowledge Base Seeding Process...")
    
    # 1. 初始化所有组件
    print("\n[Step 1/5] Initializing components...")
    workflow_manager = WorkflowManager(WORKFLOW_CONFIG_FILE)
    gnn_encoder = GNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    knowledge_base = KnowledgeBase(dimension=KB_DIMENSION)
    
    # --- 这是修正的部分 ---
    # 根据您提供的 utils.py 的构造函数 WrenchExperimentRunner.__init__(self, schedulers, config) 进行实例化
    # 1. 构造 config 参数
    config_params = {
        "platform_file": PLATFORM_FILE
        # WrenchExperimentRunner需要的其他参数可以暂时为空或默认，因为我们只使用它的单个模拟功能
    }
    # 2. schedulers 参数可以为空，因为我们不是运行对比实验
    schedulers_dict = {}

    # 3. 使用正确的关键字参数进行实例化
    wrench_runner = WrenchExperimentRunner(schedulers=schedulers_dict, config=config_params)
    print("✅ Components initialized.")
    # --- 修正结束 ---

    # 2. 生成用于“播种”的工作流
    print("\n[Step 2/5] Generating workflows for seeding...")
    seeding_workflows = workflow_manager.generate_training_workflows()
    if not seeding_workflows:
        print("❌ No workflows generated. Please check your config.")
        return
    print(f"✅ Generated {len(seeding_workflows)} workflows.")

    # 3. 循环处理每个工作流
    print("\n[Step 3/5] Simulating workflows with HEFT to gather experience...")
    all_embeddings = []
    all_metadata = []

    for i, wf_file in enumerate(seeding_workflows):
        wf_path = Path(wf_file)
        print(f"\n--- Processing workflow {i+1}/{len(seeding_workflows)}: {wf_path.name} ---")

        # a. 调用 WrenchExperimentRunner 的新方法
        #    在这里，我们明确告诉它本次模拟使用 RecordingHEFTScheduler
        print(f"  ▶️ Running WRENCH simulation via WrenchExperimentRunner using HEFT...")
        makespan, decisions = wrench_runner.run_single_seeding_simulation(
            scheduler_class=RecordingHEFTScheduler,
            workflow_file=str(wf_path)
        )

        # b. 检查模拟是否成功
        if makespan < 0 or not decisions:
            print(f"❌ Simulation failed for {wf_path.name}. Skipping.")
            continue
        
        print(f"  ⏹️ Simulation finished. Makespan: {makespan:.2f} seconds.")
        print(f"  📝 Recorded {len(decisions)} scheduling decisions.")

        # c. 将工作流图编码为向量
        try:
            pyg_data = workflow_json_to_pyg_data(str(wf_path))
            graph_embedding = gnn_encoder(pyg_data)
        except Exception as e:
            print(f"❌ Error encoding workflow {wf_path.name} to graph: {e}")
            continue

        # d. 准备存储数据
        all_embeddings.append(graph_embedding.detach().numpy().flatten())
        all_metadata.append({
            "workflow_file": wf_path.name,
            "makespan": makespan,
            "decisions": json.dumps(decisions) 
        })
            
    print("\n--- All workflows processed ---")

    # 4. 将收集到的所有数据添加到知识库
    print("\n[Step 4/5] Adding all collected experience to the Knowledge Base...")
    if not all_embeddings:
        print("❌ No experience was collected. Cannot build knowledge base.")
        return
        
    knowledge_base.add(np.array(all_embeddings), all_metadata)
    print(f"✅ Added {len(all_embeddings)} entries to the knowledge base.")

    # 5. 保存知识库到磁盘
    print("\n[Step 5/5] Saving the Knowledge Base to disk...")
    knowledge_base.save()
    print(f"✅ Knowledge Base saved successfully to '{knowledge_base.storage_path}'")
    print("\n🎉 [Phase 1] Knowledge Base Seeding Process Completed! 🎉")


if __name__ == "__main__":
    main()