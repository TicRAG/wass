# scripts/workflow_manager.py
import os
import sys
import yaml
from pathlib import Path

# --- 修正导入路径问题 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------

# 现在我们导入原始的、功能强大的生成器
from scripts.workflow_generator import WorkflowGenerator

class WorkflowManager:
    """管理工作流的生成，适配原始的WorkflowGenerator。"""
    def __init__(self, config_path="configs/workflow_config.yaml"):
        self.config_path = config_path
        print(f"🔄 [WorkflowManager] Loading config from: {self.config_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Workflow config file not found at: {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print("✅ [WorkflowManager] Config loaded successfully.")

    def _generate_workflows(self, workflow_type, config):
        """内部辅助函数，用于生成特定类型的工作流。"""
        generated_files = []
        output_dir = "data/workflows"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # --- 这是修正的部分: 使用原始的WorkflowGenerator ---
        # 创建一个生成器实例
        # 注意：原版生成器的构造函数可能需要 output_dir 和 ccr
        generator = WorkflowGenerator(output_dir=output_dir, ccr=1.0)
        # --- 修正结束 ---

        print(f"  -> Generating '{workflow_type}' workflows into '{output_dir}'...")
        
        for name, params in config.items():
            sizes = params.get("sizes", [])
            count = params.get("count", 1)
            seed = params.get("seed_start", 1)
            for size in sizes:
                for i in range(count):
                    current_seed = seed + i
                    
                    # --- 这是修正的部分: 调用正确的 generate_single_workflow 方法 ---
                    filename = f"{name}_{size}_seed{current_seed}_{workflow_type}.json"
                    output_file = generator.generate_single_workflow(
                        pattern=name,
                        task_count=size,
                        random_seed=current_seed,
                        filename=filename
                    )
                    # --- 修正结束 ---
                    generated_files.append(output_file)
        return generated_files

    def generate_experiment_workflows(self):
        """生成用于最终对比实验的工作流。"""
        if "experiment_workflows" not in self.config:
            return []
        print("\n🔬 [WorkflowManager] Generating EXPERIMENT workflows...")
        return self._generate_workflows("experiment", self.config["experiment_workflows"])

    def generate_training_workflows(self):
        """生成用于知识库和训练的工作流。"""
        if "training_workflows" not in self.config:
            return []
        print("\n📚 [WorkflowManager] Generating TRAINING workflows...")
        num_types = len(self.config["training_workflows"])
        total_to_generate = sum(
            len(p.get("sizes", [])) * p.get("count", 0)
            for p in self.config["training_workflows"].values()
        )
        print(f"  [Config] Found {num_types} workflow types to generate.")
        print(f"  [Config] Total workflows to be generated: {total_to_generate}")
        return self._generate_workflows("training", self.config["training_workflows"])