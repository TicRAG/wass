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
from src.workflows.generator import WorkflowGenerator  # Deprecated synthetic generator (see module docstring)

class WorkflowManager:
    """WorkflowManager (LEGACY Synthetic Generation Layer)

    DEPRECATED: Only `get_platform_file()` is used in the WFCommons-based pipeline.
    Synthetic generation methods remain for historical benchmarking and will be
    removed once confirmed unnecessary.
    """
    # NOTE: Not used by run_pipeline.sh except for platform resolution.

    def __init__(self, config_path="configs/workflow_config.yaml"):
        self.config_path = config_path
        print(f"🔄 [WorkflowManager] Loading config from: {self.config_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Workflow config file not found at: {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print("✅ [WorkflowManager] Config loaded successfully.")

    def get_platform_file(self, key: str = None) -> str:
        """Resolve platform XML path using config/env override."""
        px_cfg = self.config.get('platform_xml')
        if not px_cfg:
            raise KeyError("platform_xml section missing in workflow config; please add it.")
        base_dir = px_cfg.get('base_dir', 'configs')
        mapping = px_cfg.get('mapping', {})
        env_key = os.environ.get('WASS_PLATFORM')
        chosen = key or env_key or px_cfg.get('default', 'small')
        if chosen not in mapping:
            raise ValueError(f"Platform key '{chosen}' not found. Available: {list(mapping.keys())}")
        platform_path = os.path.join(base_dir, mapping[chosen])
        if not os.path.exists(platform_path):
            print(f"⚠️  Platform XML '{platform_path}' does not exist yet.")
        return platform_path

    # Deprecated synthetic generation (unused in WFCommons flow)
    def _generate_workflows(self, workflow_type, config):
        """(Deprecated) Generate synthetic workflows (not used)."""
        generated_files = []
        output_dir = "data/workflows"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        generator = WorkflowGenerator(output_dir=output_dir, ccr=1.0)
        for name, params in config.items():
            sizes = params.get("sizes", [])
            count = params.get("count", 1)
            seed = params.get("seed_start", 1)
            for size in sizes:
                for i in range(count):
                    current_seed = seed + i
                    filename = f"{name}_{size}_seed{current_seed}_{workflow_type}.json"
                    output_file = generator.generate_single_workflow(
                        pattern=name,
                        task_count=size,
                        random_seed=current_seed,
                        filename=filename
                    )
                    generated_files.append(output_file)
        return generated_files

    def generate_experiment_workflows(self):
        """(Deprecated) Synthetic experiment workflows (unused)."""
        if "experiment_workflows" not in self.config:
            return []
        print("\n🔬 [WorkflowManager] Generating EXPERIMENT workflows...")
        return self._generate_workflows("experiment", self.config["experiment_workflows"])

    def generate_training_workflows(self):
        """(Deprecated) Synthetic training workflows (unused)."""
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