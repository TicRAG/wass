import sys
from pathlib import Path
import torch
import yaml

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.ai_schedulers import WASSDRLScheduler, WASSRAGScheduler
from src.utils import WrenchExperimentRunner
from src.wrench_schedulers import FIFOScheduler, HEFTScheduler, WASSHeuristicScheduler
from src.drl_agent import DQNAgent

class RealWrenchExperiment:
    """
    Runs a real WASS-RAG experiment using WRENCH, comparing various schedulers.
    """
    def __init__(self, config):
        self.config = config
        # [FIX] Use .get() for safer access to the config dictionary
        self.drl_model_path = self.config.get('drl_model_path')
        if not self.drl_model_path:
            raise ValueError("配置文件中缺少必须的 'drl_model_path' 配置项")
        self.drl_agent = None

    def _find_model_file(self):
        """Finds and validates the DRL model file path."""
        model_path = Path(self.drl_model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"DRL model file not found at {model_path}")
        print(f"✅ 找到模型文件: {model_path}")
        return str(model_path)

    def _create_and_load_model(self):
        """Creates the DRL agent and loads the trained weights."""
        model_file = self._find_model_file()
        print(f"📁 使用模型文件: {model_file}")

        try:
            print("🔍 正在加载模型文件...")
            checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['q_network_state_dict']
            first_layer_key = next(iter(state_dict))
            state_dim = state_dict[first_layer_key].shape[1]
            last_layer_key = next(reversed(state_dict))
            action_dim = state_dict[last_layer_key].shape[0]
            print(f"✅ 模型结构动态识别: state_dim={state_dim}, action_dim={action_dim}")

            self.drl_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
            self.drl_agent.load_state_dict(state_dict)
            self.drl_agent.eval()
            print("✅ DRL Agent模型加载并验证成功！")
        except Exception as e:
            print(f"❌ 加载DRL模型失败: {e}"); sys.exit(1)

    def run(self):
        """Executes the entire experiment."""
        print("🚀 开始基于WRENCH的真实WASS-RAG实验...")
        self._create_and_load_model()
        
        schedulers_map = {
            "FIFO": FIFOScheduler,
            "HEFT": HEFTScheduler,
            "WASS-Heuristic": WASSHeuristicScheduler,
            "WASS-DRL": lambda sim, cs, hosts: WASSDRLScheduler(self.drl_agent, sim, cs, hosts),
            "WASS-RAG": lambda sim, cs, hosts: WASSRAGScheduler(self.drl_agent, self.config.get('rag_config_path'), sim, cs, hosts)
        }
        
        enabled_schedulers = self.config.get('enabled_schedulers', list(schedulers_map.keys()))
        schedulers_to_run = {name: s_class for name, s_class in schedulers_map.items() if name in enabled_schedulers}
        print(f"🔧 已启用调度器: {list(schedulers_to_run.keys())}")

        runner = WrenchExperimentRunner(schedulers=schedulers_to_run, config=self.config)
        
        print("🔬 开始完整WRENCH实验...")
        results = runner.run_all()

        print("\n📈 实验结果分析:")
        runner.analyze_results(results)

if __name__ == "__main__":
    config_path = project_root / "configs/real_heuristic_experiment.yaml"
    with open(config_path, 'r') as f: exp_config = yaml.safe_load(f)
    experiment = RealWrenchExperiment(exp_config)
    experiment.run()