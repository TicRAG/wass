import sys
from pathlib import Path
import torch
import yaml
import sys
import os

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.ai_schedulers import WASSDRLScheduler, WASSRAGScheduler
from src.utils import WrenchExperimentRunner
from src.wrench_schedulers import FIFOScheduler, HEFTScheduler, WASSHeuristicScheduler
from src.drl_agent import DQNAgent
from src.shared_models import SimplePerformancePredictor

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
        self.predictor_model_path = self.config.get('predictor_model_path', "models/performance_predictor.pth")  # 使用配置文件中的路径，默认与阶段二的输出路径一致
        self.drl_agent = None
        self.predictor = None

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
    
    def _create_and_load_predictor(self):
        """创建并加载性能预测器（导师模型）"""
        try:
            predictor_path = Path(self.predictor_model_path)
            if not predictor_path.is_absolute():
                predictor_path = project_root / predictor_path
            
            if not predictor_path.exists():
                raise FileNotFoundError(f"性能预测器模型文件未找到: {predictor_path}")
            
            print(f"📊 加载性能预测器模型: {predictor_path}")
            
            # 动态确定 input_dim
            # 阶段二的产物是 state_dict，我们需要先加载它来确定 input_dim
            state_dict = torch.load(predictor_path, map_location="cpu", weights_only=False)
            
            # 检查是否有嵌套的 state_dict
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # 获取第一层的输入维度
            first_layer_key = next(key for key in state_dict if 'weight' in key)
            input_dim = state_dict[first_layer_key].shape[1]
            print(f"✅ 动态识别模型输入维度: {input_dim}")
            
            # 使用与阶段二相同的模型类和参数创建模型
            self.predictor = SimplePerformancePredictor(input_dim=input_dim)
            
            # 加载模型权重
            self.predictor.load_state_dict(state_dict)
            self.predictor.eval()
            print("✅ 性能预测器（导师）加载成功！")
        except Exception as e:
            print(f"❌ 加载性能预测器失败: {e}"); sys.exit(1)

    def run(self):
        """Executes the entire experiment."""
        print("🚀 开始基于WRENCH的真实WASS-RAG实验...")
        self._create_and_load_model()
        self._create_and_load_predictor()
        import traceback
        print("✅ 所有组件加载完成，准备运行实验...")
        
        # 定义调度器工厂函数，解决构造函数参数不匹配问题
        def create_wass_drl(sim, cs, hosts):
            node_names = list(hosts.keys())
            scheduler = WASSDRLScheduler(self.drl_agent, node_names, self.predictor)
            # 设置仿真上下文
            scheduler.set_simulation_context(sim, cs, list(hosts.keys()))
            return scheduler
        
        def create_wass_rag(sim, cs, hosts):
            node_names = list(hosts.keys())
            # 从rag配置文件中获取知识库路径
            rag_config_path = self.config.get('rag_config_path')
            if rag_config_path:
                import yaml
                with open(rag_config_path, 'r') as f:
                    rag_config = yaml.safe_load(f)
                knowledge_base_path = rag_config.get('rag', {}).get('knowledge_base_path', 'data/real_heuristic_kb.json')
            else:
                knowledge_base_path = 'data/real_heuristic_kb.json'
            scheduler = WASSRAGScheduler(self.drl_agent, node_names, self.predictor, knowledge_base_path)
            # 设置仿真上下文
            scheduler.set_simulation_context(sim, cs, list(hosts.keys()))
            return scheduler
        
        schedulers_map = {
            "FIFO": FIFOScheduler,
            "HEFT": HEFTScheduler,
            "WASS-Heuristic": WASSHeuristicScheduler,
            "WASS-DRL": create_wass_drl,
            "WASS-RAG": create_wass_rag
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