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
    return cfg

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
        """计算奖励函数"""
        # 基础奖励：负的完成时间（越快越好）
        base_reward = -completion_time / 10.0
        
        # 节点效率奖励
        task_flops = task.get_flops()
        node_capacity = self.node_capacities[chosen_node]
        efficiency = node_capacity / 4.0  # 标准化到最高性能节点
        efficiency_bonus = efficiency * 2.0
        
        # 负载均衡奖励
        node_usage = sum(1 for t_name, t_node in getattr(self, 'task_node_mapping', {}).items() 
                        if t_node == chosen_node)
        balance_penalty = node_usage * 0.5
        
        total_reward = base_reward + efficiency_bonus - balance_penalty
        return total_reward
    
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
    """简单的DQN网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = SimpleDQN(state_dim, action_dim).to(self.device)
        self.target_network = SimpleDQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.99
        self.batch_size = 32
        
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
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_drl_agent(config: Dict):
    """训练DRL智能体"""
    print("🚀 开始基于WRENCH的DRL智能体训练...")
    
    # 创建环境
    platform_file = config['platform']['platform_file']
    env = WRENCHEnvironment(platform_file)
    
    # 创建智能体
    agent = DQNAgent(env.state_dim, env.action_dim)
    
    # 训练参数
    episodes = config.get('drl', {}).get('episodes', 50)
    max_steps = config.get('drl', {}).get('max_steps', 20)
    
    # 训练循环
    episode_rewards = []
    episode_makespans = []
    
    for episode in range(episodes):
        state = env.reset(num_tasks=random.randint(5, 15))
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
        
        # 更新目标网络
        if episode % 10 == 0:
            agent.update_target_network()
        
        # 记录性能
        makespan = env.get_final_makespan()
        episode_rewards.append(total_reward)
        episode_makespans.append(makespan)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_makespan = np.mean(episode_makespans[-10:])
            print(f"Episode {episode}: 平均奖励={avg_reward:.2f}, 平均Makespan={avg_makespan:.2f}s, ε={agent.epsilon:.3f}")
    
    # 保存模型
    model_path = Path("models/wass_models.pth")
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
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    torch.save(checkpoint, model_path)
    print(f"✅ DRL模型已保存到 {model_path}")
    
    # 清理
    env.cleanup()
    
    return {
        "final_performance": np.mean(episode_makespans[-10:]),
        "training_episodes": episodes,
        "model_path": str(model_path)
    }

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Usage: python scripts/train_drl_wrench.py <config.yaml>")
        sys.exit(1)
    
    cfg_path = sys.argv[1]
    config = load_config(cfg_path)
    
    results = train_drl_agent(config)
    print(f"\n🎉 DRL训练完成！最终性能: {results['final_performance']:.2f}s")

if __name__ == "__main__":
    main()
