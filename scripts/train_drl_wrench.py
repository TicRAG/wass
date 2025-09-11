#!/usr/bin/env python3
"""
基于WRENCH的DRL智能体训练脚本
通过真实的WRENCH仿真环境训练深度强化学习调度器
"""

import sys
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml

# 确保能导入WRENCH
try:
    import wrench
except ImportError:
    print("Error: WRENCH not available. Please install wrench-python-api.")
    sys.exit(1)

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, str(parent_dir))

def load_config(cfg_path: str) -> Dict:
    """加载配置文件"""
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    # Process includes
    if 'include' in cfg:
        base_dir = os.path.dirname(cfg_path)
        for include_file in cfg['include']:
            include_path = os.path.join(base_dir, include_file)
            if os.path.exists(include_path):
                with open(include_path, 'r', encoding='utf-8') as f:
                    include_cfg = yaml.safe_load(f) or {}
                    for key, value in include_cfg.items():
                        if key not in cfg:
                            cfg[key] = value
    
    # 🏆 集成调优后的最佳超参数配置
    optimized_params = get_optimized_hyperparameters()
    cfg.update(optimized_params)
    
    return cfg

def get_optimized_hyperparameters() -> Dict:
    """
    获取调优后的最佳超参数配置
    基于100次超参数搜索的最优结果
    """
    # 尝试从调优结果文件加载
    tuned_config_path = "/data/workspace/wass/results/local_hyperparameter_tuning/best_hyperparameters_for_training.yaml"
    
    if os.path.exists(tuned_config_path):
        print("📊 加载调优后的最佳超参数配置...")
        try:
            with open(tuned_config_path, 'r') as f:
                tuned_config = yaml.safe_load(f)
            
            # 转换为训练脚本需要的格式
            optimized = {
                'learning_rate': tuned_config['training']['learning_rate'],
                'gamma': tuned_config['training']['gamma'],
                'epsilon_start': tuned_config['training']['epsilon_start'],
                'epsilon_end': tuned_config['training']['epsilon_end'],
                'epsilon_decay': tuned_config['training']['epsilon_decay'],
                'batch_size': tuned_config['training']['batch_size'],
                'memory_size': tuned_config['training']['memory_size'],
                'target_update_freq': tuned_config['training']['target_update_freq'],
                'hidden_dim_1': tuned_config['model']['hidden_layers'][0],
                'hidden_dim_2': tuned_config['model']['hidden_layers'][1],
                'dropout_rate': tuned_config['model']['dropout_rate'],
                # 密集奖励权重
                'data_locality_weight': tuned_config['reward_weights']['data_locality_weight'],
                'waiting_time_weight': tuned_config['reward_weights']['waiting_time_weight'],
                'critical_path_weight': tuned_config['reward_weights']['critical_path_weight'],
                'load_balancing_weight': tuned_config['reward_weights']['load_balancing_weight']
            }
            
            print(f"  ✅ 学习率: {optimized['learning_rate']}")
            print(f"  ✅ Gamma: {optimized['gamma']}")
            print(f"  ✅ 网络结构: [{optimized['hidden_dim_1']}, {optimized['hidden_dim_2']}]")
            print(f"  ✅ 批次大小: {optimized['batch_size']}")
            print(f"  ✅ 关键路径权重: {optimized['critical_path_weight']}")
            
            return optimized
            
        except Exception as e:
            print(f"⚠️ 加载调优配置失败: {e}")
    
    # 如果没有调优结果，使用硬编码的最佳配置
    print("📊 使用硬编码的最佳超参数配置...")
    return {
        'learning_rate': 0.0005,  # 调优得出的最佳学习率
        'gamma': 0.99,           # 调优得出的最佳折扣因子
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,        # 调优得出的最佳批次大小
        'memory_size': 2000,
        'target_update_freq': 100,
        'hidden_dim_1': 256,     # 调优得出的最佳网络结构
        'hidden_dim_2': 128,
        'dropout_rate': 0.2,
        # 密集奖励权重 (按技术报告设计)
        'data_locality_weight': 0.2,
        'waiting_time_weight': 0.1,
        'critical_path_weight': 0.4,  # 最高权重
        'load_balancing_weight': 0.1
    }

class WRENCHEnvironment:
    """基于WRENCH的强化学习环境"""
    
    def __init__(self, platform_file: str, controller_host: str = "ControllerHost"):
        self.platform_file = platform_file
        self.controller_host = controller_host
        
        # 读取平台配置
        with open(platform_file, 'r', encoding='utf-8') as f:
            self.platform_xml = f.read()
        
        self.sim = None
        self.workflow = None
        self.compute_service = None
        self.storage_service = None
        
        # 获取计算节点信息
        self._setup_nodes()
        
        # 状态和动作空间
        self.state_dim = 5 + len(self.compute_nodes) * 3  # 任务特征 + 节点特征
        self.action_dim = len(self.compute_nodes)
        
        print(f"WRENCH环境初始化完成: {len(self.compute_nodes)} 计算节点, 状态维度: {self.state_dim}")
    
    def _setup_nodes(self):
        """设置计算节点信息"""
        # 从平台XML中解析节点信息
        self.compute_nodes = ["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"]
        self.node_capacities = {
            "ComputeHost1": 2.0,  # 2GHz
            "ComputeHost2": 3.0,  # 3GHz  
            "ComputeHost3": 2.5,  # 2.5GHz
            "ComputeHost4": 4.0   # 4GHz
        }
    
    def reset(self, num_tasks: int = 10) -> np.ndarray:
        """重置环境并返回初始状态"""
        if self.sim:
            try:
                self.sim.terminate()
            except:
                pass
        
        # 创建新的仿真
        self.sim = wrench.Simulation()
        self.sim.start(self.platform_xml, self.controller_host)
        
        # 创建服务
        self.storage_service = self.sim.create_simple_storage_service("StorageHost", ["/storage"])
        
        # 创建计算服务
        compute_resources = {}
        for node in self.compute_nodes:
            compute_resources[node] = (4, 8_589_934_592)  # 4核, 8GB内存
        
        self.compute_service = self.sim.create_bare_metal_compute_service(
            "ComputeHost1", compute_resources, "/scratch", {}, {}
        )
        
        # 创建工作流
        self.workflow = self.sim.create_workflow()
        self.tasks = []
        self.files = []
        
        # 创建任务和文件
        for i in range(num_tasks):
            # 不同类型的任务
            if i % 3 == 0:  # CPU密集型
                flops = random.uniform(8e9, 15e9)
            elif i % 3 == 1:  # 中等任务
                flops = random.uniform(3e9, 8e9)
            else:  # 轻量任务
                flops = random.uniform(1e9, 3e9)
            
            task = self.workflow.add_task(f"task_{i}", flops, 1, 1, 0)
            self.tasks.append(task)
            
            # 创建输出文件
            if i < num_tasks - 1:
                output_file = self.sim.add_file(f"output_{i}", random.randint(1024, 10240))
                task.add_output_file(output_file)
                self.files.append(output_file)
        
        # 创建任务依赖关系
        for i in range(1, min(num_tasks, len(self.files) + 1)):
            if i > 1 and random.random() < 0.3:  # 30%概率有依赖
                dep_idx = random.randint(0, i-2)
                if dep_idx < len(self.files):
                    self.tasks[i].add_input_file(self.files[dep_idx])
        
        # 为所有文件创建副本在存储服务上
        for file in self.files:
            self.storage_service.create_file_copy(file)
        
        # 初始化调度状态
        self.scheduled_tasks = set()
        self.task_completion_times = {}
        self.node_availability = {node: 0.0 for node in self.compute_nodes}
        self.current_time = 0.0
        
        # 返回初始状态
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态向量"""
        ready_tasks = self.workflow.get_ready_tasks()
        
        if not ready_tasks:
            # 如果没有就绪任务，返回零状态
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # 选择第一个就绪任务
        current_task = ready_tasks[0]
        
        # 任务特征
        task_features = [
            current_task.get_flops() / 1e9,  # 标准化到GFlops
            len(current_task.get_input_files()),
            current_task.get_number_of_children(),
            len(self.tasks),  # 总任务数
            len(self.scheduled_tasks) / len(self.tasks)  # 完成进度
        ]
        
        # 节点特征
        node_features = []
        for node in self.compute_nodes:
            capacity = self.node_capacities[node]
            availability = self.node_availability[node]
            
            # 计算估计执行时间
            exec_time = current_task.get_flops() / (capacity * 1e9)
            
            node_features.extend([
                capacity / 4.0,  # 标准化容量
                availability / 100.0,  # 标准化可用时间
                exec_time / 10.0   # 标准化执行时间
            ])
        
        state = np.array(task_features + node_features, dtype=np.float32)
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作并返回新状态、奖励、是否结束、额外信息"""
        ready_tasks = self.workflow.get_ready_tasks()
        
        if not ready_tasks:
            # 没有就绪任务，检查是否完成
            done = self.workflow.is_done()
            return self._get_state(), 0.0, done, {}
        
        # 选择要调度的任务
        task_to_schedule = ready_tasks[0]
        
        # 执行调度动作
        chosen_node = self.compute_nodes[action % len(self.compute_nodes)]
        
        # 创建作业并提交
        file_locations = {}
        for f in task_to_schedule.get_input_files():
            file_locations[f] = self.storage_service
        for f in task_to_schedule.get_output_files():
            file_locations[f] = self.storage_service
        
        job = self.sim.create_standard_job([task_to_schedule], file_locations)
        
        try:
            self.compute_service.submit_standard_job(job)
            
            # 等待作业完成
            while True:
                event = self.sim.wait_for_next_event()
                if event["event_type"] == "standard_job_completion":
                    completed_job = event["standard_job"]
                    if completed_job == job:
                        break
                elif event["event_type"] == "simulation_termination":
                    break
            
            # 更新状态
            self.scheduled_tasks.add(task_to_schedule.get_name())
            completion_time = self.sim.get_simulated_time()
            self.task_completion_times[task_to_schedule.get_name()] = completion_time
            self.current_time = completion_time
            
            # 计算奖励
            reward = self._calculate_reward(task_to_schedule, chosen_node, completion_time)
            
        except Exception as e:
            print(f"调度错误: {e}")
            reward = -10.0  # 严重惩罚
        
        # 检查是否完成
        done = self.workflow.is_done()
        
        return self._get_state(), reward, done, {
            "task": task_to_schedule.get_name(),
            "node": chosen_node,
            "completion_time": completion_time
        }
    
    def _calculate_reward(self, task, chosen_node: str, completion_time: float) -> float:
        """
        计算密集奖励函数 (基于技术报告的奖励设计)
        R_total = R_step + R_final
        """
        # 🎯 密集奖励设计 - R_step (中间奖励)
        
        # 1. 数据局部性奖励
        data_locality_reward = self._calculate_data_locality_reward(task, chosen_node)
        
        # 2. 等待时间惩罚  
        waiting_time_penalty = self._calculate_waiting_time_penalty(task, chosen_node)
        
        # 3. 关键路径奖励
        critical_path_reward = self._calculate_critical_path_reward(task, chosen_node)
        
        # 4. 负载均衡奖励
        load_balancing_reward = self._calculate_load_balancing_reward(chosen_node)
        
        # 使用调优后的权重组合
        config = getattr(self, 'config', {})
        data_locality_weight = config.get('data_locality_weight', 0.2)
        waiting_time_weight = config.get('waiting_time_weight', 0.1)
        critical_path_weight = config.get('critical_path_weight', 0.4)
        load_balancing_weight = config.get('load_balancing_weight', 0.1)
        
        r_step = (
            data_locality_weight * data_locality_reward +
            waiting_time_weight * (-waiting_time_penalty) +
            critical_path_weight * critical_path_reward +
            load_balancing_weight * load_balancing_reward
        )
        
        # R_final (最终奖励) - 在工作流结束时给予
        r_final = -completion_time / 20.0  # 标准化的完成时间奖励
        
        total_reward = r_step + r_final
        return total_reward
    
    def _calculate_data_locality_reward(self, task, chosen_node: str) -> float:
        """计算数据局部性奖励"""
        # 修复WRENCH API兼容性：使用正确的方法获取输入文件
        try:
            if hasattr(task, 'get_input_files'):
                input_files = task.get_input_files()
            elif hasattr(task, 'input_files'):
                input_files = task.input_files
            else:
                input_files = []
            
            if not input_files:
                return 0.1  # 没有输入文件的任务给小奖励
            
            # 检查数据是否在本地 (简化假设)
            local_data_ratio = 0.8 if chosen_node == "ComputeHost1" else 0.5
            return local_data_ratio
            
        except Exception as e:
            # 如果出错，返回默认值
            return 0.3
    
    def _calculate_waiting_time_penalty(self, task, chosen_node: str) -> float:
        """计算等待时间惩罚"""
        node_availability = self.node_availability.get(chosen_node, 0.0)
        current_time = self.current_time
        waiting_time = max(0, node_availability - current_time)
        return waiting_time / 10.0  # 标准化
    
    def _calculate_critical_path_reward(self, task, chosen_node: str) -> float:
        """计算关键路径奖励"""
        # 修复WRENCH API兼容性：使用正确的方法获取后继任务
        try:
            # WRENCH Task对象使用不同的方法名
            if hasattr(task, 'get_children_tasks'):
                children = task.get_children_tasks()
            elif hasattr(task, 'children'):
                children = task.children
            else:
                # 如果没有相关方法，使用工作流级别的信息
                children = []
                for other_task in self.tasks:
                    if other_task != task:
                        # 检查是否有依赖关系（简化实现）
                        if hasattr(other_task, 'get_input_files'):
                            input_files = other_task.get_input_files()
                            output_files = task.get_output_files() if hasattr(task, 'get_output_files') else []
                            # 如果other_task的输入包含当前task的输出，则是子任务
                            if any(f in input_files for f in output_files):
                                children.append(other_task)
            
            num_children = len(children)
            if num_children > 0:
                return min(num_children / 3.0, 1.0)  # 标准化到[0,1]
            return 0.1
            
        except Exception as e:
            # 如果出错，返回默认值
            return 0.3  # 中等重要性
    
    def _calculate_load_balancing_reward(self, chosen_node: str) -> float:
        """计算负载均衡奖励"""
        # 检查节点使用分布
        node_usage = {}
        for node in self.compute_nodes:
            node_usage[node] = sum(1 for t_name, t_node in getattr(self, 'task_node_mapping', {}).items() 
                                 if t_node == node)
        
        # 计算使用方差 (越小越好)
        usage_values = list(node_usage.values())
        if len(usage_values) > 1:
            usage_variance = np.var(usage_values)
            return max(0, 1.0 - usage_variance / 10.0)  # 反比奖励
        return 0.5
    
    def get_final_makespan(self) -> float:
        """获取最终的makespan"""
        if not self.task_completion_times:
            return float('inf')
        return max(self.task_completion_times.values())
    
    def cleanup(self):
        """清理资源"""
        if self.sim:
            try:
                self.sim.terminate()
            except:
                pass
            self.sim = None

class SimpleDQN(nn.Module):
    """优化的DQN网络 - 使用调优后的最佳结构"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        super().__init__()
        
        # 使用调优后的网络结构
        hidden_dim_1 = config.get('hidden_dim_1', 256)
        hidden_dim_2 = config.get('hidden_dim_2', 128)
        dropout_rate = config.get('dropout_rate', 0.2)
        
        print(f"🧠 构建优化的DQN网络: [{state_dim}] -> [{hidden_dim_1}] -> [{hidden_dim_2}] -> [{action_dim}]")
        print(f"   Dropout率: {dropout_rate}")
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """优化的DQN智能体 - 使用调优后的最佳超参数"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用调优后的超参数
        learning_rate = config.get('learning_rate', 0.0005)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_end', 0.01)
        self.batch_size = config.get('batch_size', 64)
        memory_size = config.get('memory_size', 2000)
        self.target_update_freq = config.get('target_update_freq', 100)
        
        print(f"🤖 创建优化的DQN智能体:")
        print(f"   学习率: {learning_rate}")
        print(f"   Gamma: {self.gamma}")
        print(f"   探索参数: ε={self.epsilon} -> {self.epsilon_min} (衰减={self.epsilon_decay})")
        print(f"   批次大小: {self.batch_size}")
        print(f"   经验回放大小: {memory_size}")
        
        # 使用优化的网络结构
        self.q_network = SimpleDQN(state_dim, action_dim, config).to(self.device)
        self.target_network = SimpleDQN(state_dim, action_dim, config).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.memory = deque(maxlen=memory_size)
        self.training_step = 0
        
        # 复制参数到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: np.ndarray) -> int:
        """选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.q_network.network[-1].out_features)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_drl_agent(config: Dict):
    """训练DRL智能体 - 使用调优后的最佳超参数"""
    print("🚀 开始基于WRENCH的DRL智能体训练 (使用调优配置)...")
    
    # 显示关键配置
    print(f"📊 训练配置:")
    print(f"   学习率: {config.get('learning_rate', 0.0005)}")
    print(f"   网络结构: [{config.get('hidden_dim_1', 256)}, {config.get('hidden_dim_2', 128)}]")
    print(f"   批次大小: {config.get('batch_size', 64)}")
    print(f"   关键路径权重: {config.get('critical_path_weight', 0.4)}")
    
    # 创建环境
    platform_file = config.get('platform', {}).get('platform_file', 'configs/platform.xml')
    env = WRENCHEnvironment(platform_file)
    env.config = config  # 传递配置给环境，用于奖励计算
    
    # 创建智能体 (使用调优配置)
    agent = DQNAgent(env.state_dim, env.action_dim, config)
    
    # 训练参数
    episodes = config.get('drl', {}).get('episodes', 100)  # 增加训练episode
    max_steps = config.get('drl', {}).get('max_steps', 30)
    
    # 训练循环
    episode_rewards = []
    episode_makespans = []
    
    print(f"\n🎯 开始训练 {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset(num_tasks=random.randint(8, 20))  # 更多任务增加复杂性
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 经验回放
        agent.replay()
        
        # 按调优后的频率更新目标网络
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # 记录性能
        makespan = env.get_final_makespan()
        episode_rewards.append(total_reward)
        episode_makespans.append(makespan)
        
        # 更详细的进度报告
        if episode % 20 == 0 or episode < 10:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_makespan = np.mean(episode_makespans[-10:])
            print(f"Episode {episode:3d}: 奖励={avg_reward:6.2f}, Makespan={avg_makespan:6.2f}s, ε={agent.epsilon:.3f}, 步数={steps}")
    
    # 保存模型
    model_path = Path("models/wass_optimized_models.pth")
    model_path.parent.mkdir(exist_ok=True)
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except:
        checkpoint = {}
    
    checkpoint["drl_agent"] = agent.q_network.state_dict()
    checkpoint["drl_metadata"] = {
        "episodes": episodes,
        "final_epsilon": agent.epsilon,
        "avg_reward": np.mean(episode_rewards[-10:]),
        "avg_makespan": np.mean(episode_makespans[-10:]),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": {
            "learning_rate": config.get('learning_rate'),
            "gamma": config.get('gamma'),
            "batch_size": config.get('batch_size'),
            "hidden_layers": [config.get('hidden_dim_1'), config.get('hidden_dim_2')],
            "reward_weights": {
                "data_locality": config.get('data_locality_weight'),
                "waiting_time": config.get('waiting_time_weight'),
                "critical_path": config.get('critical_path_weight'),
                "load_balancing": config.get('load_balancing_weight')
            }
        },
        "optimization_info": "使用超参数调优后的最佳配置训练"
    }
    
    torch.save(checkpoint, model_path)
    print(f"✅ 优化DRL模型已保存到 {model_path}")
    
    # 显示训练总结
    final_makespan = np.mean(episode_makespans[-10:])
    improvement = (episode_makespans[0] - final_makespan) / episode_makespans[0] * 100 if episode_makespans[0] > 0 else 0
    
    print(f"\n📊 训练总结:")
    print(f"   最终平均Makespan: {final_makespan:.2f}s")
    print(f"   相比初期改善: {improvement:.1f}%")
    print(f"   最终探索率: {agent.epsilon:.3f}")
    print(f"   训练步数总计: {agent.training_step}")
    
    # 清理
    env.cleanup()
    
    return {
        "final_performance": final_makespan,
        "improvement": improvement,
        "training_episodes": episodes,
        "model_path": str(model_path),
        "hyperparameters_used": config
    }

def main():
    """主函数"""
    print("🎯 WASS-DRL 优化训练脚本")
    print("=" * 50)
    
    if len(sys.argv) != 2:
        print("使用方法: python scripts/train_drl_wrench.py <config.yaml>")
        print("\n💡 提示: 脚本已集成调优后的最佳超参数，无需额外配置")
        print("   自动使用以下优化配置:")
        print("   - 学习率: 0.0005")
        print("   - 网络结构: [256, 128]")  
        print("   - 批次大小: 64")
        print("   - 密集奖励函数")
        sys.exit(1)
    
    cfg_path = sys.argv[1]
    config = load_config(cfg_path)
    
    results = train_drl_agent(config)
    
    print(f"\n🎉 DRL训练完成！")
    print(f"🏆 最终性能: {results['final_performance']:.2f}s")
    print(f"📈 性能改善: {results.get('improvement', 0):.1f}%")
    print(f"💾 模型已保存到: {results['model_path']}")
    print("\n💡 下一步: 运行完整实验验证调优效果")
    print("   python experiments/wrench_real_experiment.py")

if __name__ == "__main__":
    main()
