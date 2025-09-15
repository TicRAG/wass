#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WASS-RAG 改进的DRL训练器
实现密集奖励函数以提高训练效果
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import yaml

# 先添加项目路径再导入本地包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.drl.reward import compute_step_reward, compute_final_reward, StepContext, EpisodeStats, WEIGHTS  # noqa: E402
import math

# 确保结果目录存在（冒烟运行可能尚未创建）
Path('results').mkdir(exist_ok=True)

@dataclass
class TaskState:
    """任务状态"""
    id: str
    computation_size: float
    parents: List[str]
    children: List[str]
    is_critical_path: bool
    data_locality_score: float

@dataclass
class NodeState:
    """节点状态"""
    id: str
    speed: float
    current_load: float
    available_time: float
    data_availability: Dict[str, float]  # 数据可用性评分

@dataclass
class EnvironmentState:
    """环境状态"""
    current_time: float
    pending_tasks: List[TaskState]
    node_states: List[NodeState]
    workflow_progress: float
    critical_path_length: float

class DenseRewardCalculator:
    """(Deprecated soon) 保留旧接口以防兼容问题，但内部委托到新 shaping。"""
    def calculate_step_reward(self, task: TaskState, chosen_node: NodeState, all_nodes: List[NodeState], environment: EnvironmentState) -> float:
        # 旧接口保持但现在直接返回0（避免误用），真实奖励在训练循环外部构造 StepContext
        return 0.0
    def calculate_final_reward(self, final_makespan: float, baseline_makespan: float) -> float:
        stats = EpisodeStats(makespan=final_makespan)
        return compute_final_reward(stats)

class AdvancedDQN(nn.Module):
    """高级DQN网络，引入注意力机制和更深层次结构"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]  # 更深的网络结构
        self.hidden_dims = hidden_dims
        
        # 特征提取层
        feature_layers = []
        current_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            feature_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 使用LayerNorm代替BatchNorm，对小批量更稳定
                nn.ReLU(),
                nn.Dropout(0.1 if i < 2 else 0.05)  # 前层使用更高dropout
            ])
            current_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.Tanh(),
            nn.Linear(current_dim // 2, 1)
        )
        
        # 价值流 (Value Stream)
        value_layers = []
        value_input_dim = current_dim
        for hidden_dim in [hidden_dims[-1], hidden_dims[-1] // 2]:
            value_layers.extend([
                nn.Linear(value_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.05)
            ])
            value_input_dim = hidden_dim
        value_layers.append(nn.Linear(value_input_dim, 1))
        self.value_stream = nn.Sequential(*value_layers)
        
        # 优势流 (Advantage Stream)
        advantage_layers = []
        advantage_input_dim = current_dim
        for hidden_dim in [hidden_dims[-1], hidden_dims[-1] // 2]:
            advantage_layers.extend([
                nn.Linear(advantage_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.05)
            ])
            advantage_input_dim = hidden_dim
        advantage_layers.append(nn.Linear(advantage_input_dim, action_dim))
        self.advantage_stream = nn.Sequential(*advantage_layers)
        
        # 使用Xavier初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 应用注意力机制
        attention_weights = self.attention(features)
        attention_weights = torch.softmax(attention_weights, dim=0)
        attended_features = features * attention_weights
        
        # Dueling DQN架构
        value = self.value_stream(attended_features)
        advantage = self.advantage_stream(attended_features)
        
        # Q值 = 价值 + (优势 - 平均优势)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class ImprovedDQN(nn.Module):
    """改进的DQN网络 - 保留原实现作为备选"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        current_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 使用Xavier初始化
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

class ImprovedDQNAgent:
    """改进的DQN智能体，支持多种网络架构"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 gamma: float = 0.99,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 device: str = None,
                 network_type: str = "advanced",  # 新增：网络类型选择
                 hidden_dims: List[int] = None,
                 exploration_strategy: str = "adaptive_epsilon",  # 新增：探索策略类型
                 use_ucb: bool = False,  # 新增：是否使用UCB探索
                 ucb_c: float = 2.0,  # 新增：UCB探索参数
                 use_boltzmann: bool = False,  # 新增：是否使用玻尔兹曼探索
                 boltzmann_tau: float = 1.0):  # 新增：玻尔兹曼温度参数
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.network_type = network_type
        
        # 探索策略参数
        self.exploration_strategy = exploration_strategy
        self.use_ucb = use_ucb
        self.ucb_c = ucb_c
        self.use_boltzmann = use_boltzmann
        self.boltzmann_tau = boltzmann_tau
        
        # 动作访问计数（用于UCB探索）
        self.action_counts = np.zeros(action_dim)
        self.action_values = np.zeros(action_dim)
        
        # 探索统计
        self.exploration_history = deque(maxlen=1000)
        self.exploitation_history = deque(maxlen=1000)
        self.recent_rewards = deque(maxlen=100)  # 用于自适应探索率调整
        
        # 根据网络类型选择网络架构
        if network_type == "advanced":
            # 使用高级DQN网络（带注意力和Dueling结构）
            self.q_network = AdvancedDQN(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_network = AdvancedDQN(state_dim, action_dim, hidden_dims).to(self.device)
        else:
            # 使用标准DQN网络
            self.q_network = ImprovedDQN(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_network = ImprovedDQN(state_dim, action_dim, hidden_dims).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        
        # 探索策略
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 自适应探索参数
        self.performance_window = deque(maxlen=50)  # 性能窗口
        self.stable_performance_threshold = 0.05  # 性能稳定阈值
        self.min_epsilon = epsilon_end
        self.max_epsilon = epsilon_start
        
        # 训练统计
        self.training_step = 0
        self.update_target()
        
        # 性能监控
        self.loss_history = deque(maxlen=1000)
        self.q_value_history = deque(maxlen=1000)
    
    def update_target(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_action_dim(self, new_action_dim):
        """更新动作维度，用于课程学习阶段切换"""
        if new_action_dim == self.action_dim:
            return  # 动作维度没有变化，无需更新
        
        print(f"🔄 更新动作维度: {self.action_dim} -> {new_action_dim}")
        
        # 保存当前网络状态
        q_network_state = self.q_network.state_dict()
        target_network_state = self.target_network.state_dict()
        
        # 更新动作维度
        self.action_dim = new_action_dim
        
        # 重新创建网络
        if self.network_type == "advanced":
            # 使用高级DQN网络（带注意力和Dueling结构）
            self.q_network = AdvancedDQN(self.state_dim, self.action_dim, self.q_network.hidden_dims).to(self.device)
            self.target_network = AdvancedDQN(self.state_dim, self.action_dim, self.target_network.hidden_dims).to(self.device)
        else:
            # 使用标准DQN网络
            self.q_network = ImprovedDQN(self.state_dim, self.action_dim, self.q_network.hidden_dims).to(self.device)
            self.target_network = ImprovedDQN(self.state_dim, self.action_dim, self.target_network.hidden_dims).to(self.device)
        
        # 尝试加载之前的状态（仅加载兼容的层）
        try:
            # 对于兼容的层，加载之前的状态
            q_compatible_state = {k: v for k, v in q_network_state.items() if k in self.q_network.state_dict() and v.shape == self.q_network.state_dict()[k].shape}
            target_compatible_state = {k: v for k, v in target_network_state.items() if k in self.target_network.state_dict() and v.shape == self.target_network.state_dict()[k].shape}
            
            self.q_network.load_state_dict(q_compatible_state, strict=False)
            self.target_network.load_state_dict(target_compatible_state, strict=False)
            
            print(f"✅ 成功迁移网络状态到新的动作维度")
        except Exception as e:
            print(f"⚠️ 迁移网络状态失败: {e}，将重新初始化网络")
        
        # 重新初始化优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        
        # 重置动作访问计数（用于UCB探索）
        self.action_counts = np.zeros(self.action_dim)
        self.action_values = np.zeros(self.action_dim)
        
        # 清空经验回放，因为旧经验可能不适用于新的动作空间
        self.memory.clear()
        print(f"🧹 清空经验回放，以适应新的动作空间")
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def act(self, state, training=True):
        """选择动作，支持多种探索策略"""
        if not training:
            # 在测试阶段，直接选择最优动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        
        # 根据探索策略选择动作
        if self.exploration_strategy == "adaptive_epsilon":
            return self._act_with_adaptive_epsilon(state)
        elif self.exploration_strategy == "boltzmann" or self.use_boltzmann:
            return self._act_with_boltzmann(state)
        elif self.exploration_strategy == "ucb" or self.use_ucb:
            return self._act_with_ucb(state)
        else:
            # 默认使用标准ε-贪婪策略
            if random.random() < self.epsilon:
                self.exploration_history.append(1)
                return random.randint(0, self.action_dim - 1)
            else:
                self.exploitation_history.append(1)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()
    
    def _act_with_adaptive_epsilon(self, state):
        """使用自适应ε-贪婪策略选择动作"""
        # 根据性能稳定性调整探索率
        if len(self.performance_window) >= 20:
            recent_performance = list(self.performance_window)[-20:]
            if np.std(recent_performance) < self.stable_performance_threshold * np.mean(recent_performance):
                # 性能稳定，减少探索
                adaptive_epsilon = max(self.min_epsilon, self.epsilon * 0.9)
            else:
                # 性能不稳定，增加探索
                adaptive_epsilon = min(self.max_epsilon, self.epsilon * 1.1)
        else:
            adaptive_epsilon = self.epsilon
        
        if random.random() < adaptive_epsilon:
            self.exploration_history.append(1)
            return random.randint(0, self.action_dim - 1)
        else:
            self.exploitation_history.append(1)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def _act_with_boltzmann(self, state):
        """使用玻尔兹曼探索策略选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze().cpu().numpy()
            
            # 应用玻尔兹曼分布
            # 温度参数随时间衰减，从探索转向利用
            current_tau = self.boltzmann_tau * (self.epsilon_decay ** self.training_step)
            current_tau = max(0.1, current_tau)  # 确保温度不会过低
            
            # 计算概率分布
            exp_q = np.exp(q_values / current_tau)
            probs = exp_q / np.sum(exp_q)
            
            # 根据概率分布选择动作
            action = np.random.choice(self.action_dim, p=probs)
            
            # 记录探索或利用
            if probs.max() < 0.8:  # 如果最大概率小于0.8，认为是探索
                self.exploration_history.append(1)
            else:
                self.exploitation_history.append(1)
            
            return action
    
    def _act_with_ucb(self, state):
        """使用UCB（Upper Confidence Bound）探索策略选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze().cpu().numpy()
            
            # 计算UCB值
            total_counts = np.sum(self.action_counts)
            if total_counts == 0:
                # 如果所有动作都未被尝试过，随机选择
                action = random.randint(0, self.action_dim - 1)
                self.exploration_history.append(1)
                return action
            
            ucb_values = q_values + self.ucb_c * np.sqrt(np.log(total_counts) / (self.action_counts + 1e-6))
            
            # 选择UCB值最大的动作
            action = ucb_values.argmax()
            
            # 更新动作计数
            self.action_counts[action] += 1
            
            # 记录探索或利用
            if self.action_counts[action] <= np.mean(self.action_counts):
                self.exploration_history.append(1)
            else:
                self.exploitation_history.append(1)
            
            return action
    
    def replay(self):
        """经验回放训练，加入性能监控和改进训练过程"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 采样经验
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # 计算Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN：使用当前网络选择动作，目标网络评估Q值
        if self.network_type == "advanced":
            # 对于高级网络，使用Double DQN
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).detach()
        else:
            # 对于标准网络，使用原始方法
            next_q = self.target_network(next_states).max(1)[0].detach()
        
        # 计算目标Q值
        target_q = rewards + (self.gamma * next_q.squeeze() * ~dones)
        
        # 计算损失，使用Huber损失代替MSE损失，对异常值更鲁棒
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，使用动态阈值
        if self.network_type == "advanced":
            # 对于更深的网络，使用更严格的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        else:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # 记录性能指标
        self.loss_history.append(loss.item())
        avg_q = current_q.mean().item()
        self.q_value_history.append(avg_q)
        
        # 更新探索率，使用指数衰减
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # 定期更新目标网络，使用软更新
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            if self.network_type == "advanced":
                # 对于高级网络，使用软更新
                tau = 0.001  # 软更新系数
                for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            else:
                # 对于标准网络，使用硬更新
                self.update_target()
        
        return loss.item()
    
    def update_exploration_parameters(self, episode_reward=None):
        """更新探索参数，根据训练进度和性能调整探索策略"""
        # 更新探索率（ε-贪婪相关）
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # 更新玻尔兹曼温度参数
        if self.use_boltzmann:
            # 温度随时间衰减，从探索转向利用
            self.boltzmann_tau *= 0.9995
            self.boltzmann_tau = max(0.1, self.boltzmann_tau)
        
        # 更新性能窗口
        if episode_reward is not None:
            self.performance_window.append(episode_reward)
            
            # 基于性能自适应调整探索策略
            if len(self.performance_window) >= 20:
                recent_performance = list(self.performance_window)[-20:]
                perf_std = np.std(recent_performance)
                perf_mean = np.mean(recent_performance)
                
                # 如果性能波动大，增加探索
                if perf_std > self.stable_performance_threshold * perf_mean:
                    if self.exploration_strategy == "adaptive_epsilon":
                        self.epsilon = min(self.max_epsilon, self.epsilon * 1.05)
                    elif self.use_boltzmann:
                        self.boltzmann_tau = min(2.0, self.boltzmann_tau * 1.05)
                # 如果性能稳定，减少探索
                elif perf_std < 0.5 * self.stable_performance_threshold * perf_mean:
                    if self.exploration_strategy == "adaptive_epsilon":
                        self.epsilon = max(self.min_epsilon, self.epsilon * 0.95)
                    elif self.use_boltzmann:
                        self.boltzmann_tau = max(0.1, self.boltzmann_tau * 0.95)
    
    def get_exploration_stats(self):
        """获取探索统计信息"""
        exploration_rate = sum(self.exploration_history) / max(1, len(self.exploration_history))
        exploitation_rate = sum(self.exploitation_history) / max(1, len(self.exploitation_history))
        
        return {
            'exploration_rate': exploration_rate,
            'exploitation_rate': exploitation_rate,
            'current_epsilon': self.epsilon,
            'boltzmann_tau': self.boltzmann_tau if self.use_boltzmann else None,
            'ucb_c': self.ucb_c if self.use_ucb else None,
            'action_distribution': self.action_counts / max(1, np.sum(self.action_counts)),
            'exploration_strategy': self.exploration_strategy
        }
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.loss_history:
            return {}
        
        return {
            'avg_loss': sum(self.loss_history) / len(self.loss_history),
            'avg_q_value': sum(self.q_value_history) / len(self.q_value_history),
            'current_epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'network_type': self.network_type
        }

class WRENCHDRLTrainer:
    """WRENCH DRL训练器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.reward_calculator = DenseRewardCalculator()
        self.agent = None
        self.training_history = []
        # 新增：读取扩展训练/日志配置
        self.drl_cfg = self.config.get('drl', {})
        self.checkpoint_cfg = self.config.get('checkpoint', {})
        self.logging_cfg = self.config.get('logging', {})
        Path(self.checkpoint_cfg.get('dir', 'models/checkpoints/')).mkdir(parents=True, exist_ok=True)
        Path(self.logging_cfg.get('metrics_file', 'results/training_metrics.jsonl')).parent.mkdir(parents=True, exist_ok=True)
        self.best_makespan = float('inf')
        
        # 课程学习相关参数
        self.curriculum_stages = [
            {"name": "入门场景", "tasks": 5, "nodes": 4, "complexity": 0.3, "episodes": 300},
            {"name": "中级场景", "tasks": 10, "nodes": 4, "complexity": 0.6, "episodes": 400},
            {"name": "高级场景", "tasks": 15, "nodes": 4, "complexity": 0.8, "episodes": 300},
            {"name": "真实场景", "tasks": 20, "nodes": 4, "complexity": 1.0, "episodes": 200}
        ]
        self.current_stage = 0
        self.stage_episodes_completed = 0
        self.stage_performance_history = []
        
        # 自适应学习率参数
        self.learning_rate_schedule = [
            {"episodes": 200, "lr": 0.001},
            {"episodes": 400, "lr": 0.0005},
            {"episodes": 700, "lr": 0.0001},
            {"episodes": 1000, "lr": 0.00005}
        ]
        
        # 先验知识集成参数
        self.use_heuristic_guidance = self.drl_cfg.get('use_heuristic_guidance', True)
        self.heuristic_weight = self.drl_cfg.get('heuristic_weight', 0.3)  # 启发式指导的初始权重
        self.heuristic_decay = self.drl_cfg.get('heuristic_decay', 0.995)  # 启发式权重衰减率
    
    def create_mock_environment(self, stage: int = None) -> Tuple[EnvironmentState, List[TaskState], List[NodeState]]:
        """创建模拟训练环境，支持课程学习的不同阶段"""
        # 获取当前课程阶段
        if stage is None:
            stage = self.current_stage
        
        curriculum = self.curriculum_stages[stage]
        num_tasks = curriculum["tasks"]
        num_nodes = curriculum["nodes"]
        complexity = curriculum["complexity"]
        
        print(f"📚 创建课程学习环境: {curriculum['name']} (任务数: {num_tasks}, 节点数: {num_nodes}, 复杂度: {complexity})")
        
        # 基于platform.xml的真实节点配置，但根据课程阶段调整
        base_node_configs = [
            {"id": "ComputeHost1", "speed": 2.0, "cores": 4, "disk_speed": 200},
            {"id": "ComputeHost2", "speed": 3.0, "cores": 8, "disk_speed": 250},
            {"id": "ComputeHost3", "speed": 2.5, "cores": 6, "disk_speed": 220},
            {"id": "ComputeHost4", "speed": 4.0, "cores": 16, "disk_speed": 300}
        ]
        
        # 根据课程阶段选择节点
        node_configs = base_node_configs[:num_nodes]
        
        # 根据复杂度调整节点性能
        for config in node_configs:
            # 在简单阶段，节点性能差异较小
            if complexity < 0.5:
                speed_factor = 0.8 + 0.4 * random.random()  # 0.8-1.2倍
            else:
                speed_factor = 0.5 + 1.5 * random.random()  # 0.5-2.0倍
            
            config["speed"] *= speed_factor
            config["cores"] = max(2, int(config["cores"] * speed_factor))
            config["disk_speed"] *= speed_factor
        
        # 创建节点状态
        node_states = []
        for config in node_configs:
            # 根据复杂度调整初始负载
            if complexity < 0.5:
                initial_load = random.uniform(0, 0.2)  # 简单阶段负载较低
            else:
                initial_load = random.uniform(0, 0.4)  # 复杂阶段负载较高
            
            node_states.append(NodeState(
                id=config["id"],
                speed=config["speed"],
                current_load=initial_load,
                available_time=random.uniform(0, 5 * (2 - complexity)),  # 复杂度越高，初始可用时间越短
                data_availability={f"task_{j}": random.random() for j in range(num_tasks)}
            ))
        
        # 创建任务状态，考虑工作流结构和复杂度
        task_states = []
        
        # 根据复杂度调整任务大小分布
        if complexity < 0.3:
            # 简单阶段：任务大小差异小
            task_sizes = [(1e9, 3e9)]
        elif complexity < 0.7:
            # 中等阶段：任务大小有差异
            task_sizes = [(1e9, 3e9), (3e9, 1e10)]
        else:
            # 复杂阶段：任务大小差异大
            task_sizes = [(1e9, 3e9), (3e9, 1e10), (1e10, 5e10)]
        
        # 生成任务，考虑依赖关系和复杂度
        for i in range(num_tasks):
            # 随机选择任务大小
            min_size, max_size = random.choice(task_sizes)
            computation_size = random.uniform(min_size, max_size)
            
            # 创建依赖关系，复杂度越高依赖关系越复杂
            parents = []
            # 基础依赖概率随复杂度增加
            base_prob = 0.2 + 0.3 * complexity
            # 远距离依赖概率随复杂度增加
            long_prob = 0.05 + 0.15 * complexity
            if i > 0:
                if random.random() < base_prob:
                    parents.append(f"task_{i-1}")
                
                if i > 2 and random.random() < long_prob:
                    parents.append(f"task_{random.randint(0, i-2)}")
            
            # 创建子任务关系
            children = []
            if i < num_tasks - 1:
                if random.random() < base_prob:
                    children.append(f"task_{i+1}")
                
                if i < num_tasks - 3 and random.random() < long_prob:
                    children.append(f"task_{random.randint(i+2, num_tasks-1)}")
            
            # 判断是否为关键路径任务，复杂度越高关键路径任务比例越高
            critical_prob = 0.1 + 0.2 * complexity
            is_critical_path = random.random() < critical_prob
            
            # 计算数据局部性分数，复杂度越高数据局部性越差
            if complexity < 0.5:
                data_locality_score = random.uniform(0.6, 1.0)  # 简单阶段数据局部性较好
            else:
                data_locality_score = random.uniform(0.3, 0.9)  # 复杂阶段数据局部性较差
            
            # 根据复杂度调整数据大小
            if complexity < 0.5:
                data_size = random.uniform(1e6, 5e7)  # 简单阶段数据量较小
            else:
                data_size = random.uniform(1e6, 1e8)  # 复杂阶段数据量较大
            
            task_states.append(TaskState(
                id=f"task_{i}",
                computation_size=computation_size,
                parents=parents,
                children=children,
                is_critical_path=is_critical_path,
                data_locality_score=data_locality_score
            ))
        
        # 创建环境状态，根据复杂度调整网络条件
        if complexity < 0.5:
            # 简单阶段网络条件好
            network_bandwidth = 1.0 + 0.5 * (1 - complexity)  # 1.0-1.5 GBps
            network_latency = 0.001 * complexity  # 0-0.0005 ms
        else:
            # 复杂阶段网络条件较差
            network_bandwidth = 0.5 + 0.5 * (1 - complexity)  # 0.5-1.0 GBps
            network_latency = 0.001 * complexity  # 0.0005-0.001 ms
        
        environment = EnvironmentState(
            current_time=0.0,
            pending_tasks=task_states,
            node_states=node_states,
            workflow_progress=0.0,
            critical_path_length=self._estimate_critical_path_length(task_states, node_states)
        )
        
        return environment, task_states, node_states
    
    def _estimate_critical_path_length(self, task_states: List[TaskState], node_states: List[NodeState]) -> float:
        """估算关键路径长度"""
        # 找出关键路径上的任务
        critical_tasks = [t for t in task_states if t.is_critical_path]
        
        if not critical_tasks:
            # 如果没有明确的关键路径任务，找出最长路径
            return self._find_longest_path(task_states, node_states)
        
        # 计算关键路径长度
        total_length = 0.0
        fastest_node = max(node_states, key=lambda n: n.speed)
        
        for task in critical_tasks:
            # 使用最快节点的执行时间作为估算
            execution_time = task.computation_size / fastest_node.speed
            total_length += execution_time
        
        return total_length
    
    def _find_longest_path(self, task_states: List[TaskState], node_states: List[NodeState]) -> float:
        """找出任务图中的最长路径（关键路径）"""
        # 构建任务图
        task_dict = {task.id: task for task in task_states}
        
        # 找出没有父任务的任务（起始任务）
        start_tasks = [task for task in task_states if not task.parents]
        
        if not start_tasks:
            # 如果没有起始任务（循环依赖），返回默认值
            return 100.0
        
        # 使用DFS找出最长路径
        max_path_length = 0.0
        fastest_node = max(node_states, key=lambda n: n.speed)
        
        def dfs(task_id, current_length, visited):
            nonlocal max_path_length
            
            if task_id in visited:
                return
            
            visited.add(task_id)
            task = task_dict[task_id]
            
            # 计算当前任务的执行时间
            execution_time = task.computation_size / fastest_node.speed
            current_length += execution_time
            
            # 更新最大路径长度
            max_path_length = max(max_path_length, current_length)
            
            # 递归处理子任务
            for child_id in task.children:
                dfs(child_id, current_length, visited.copy())
        
        # 从每个起始任务开始DFS
        for start_task in start_tasks:
            dfs(start_task.id, 0.0, set())
        
        return max_path_length or 100.0  # 如果没有找到路径，返回默认值
    
    def _get_heuristic_action(self, current_task: TaskState, node_states: List[NodeState], environment: EnvironmentState) -> int:
        """获取启发式算法建议的动作，用于指导DRL学习"""
        if not self.use_heuristic_guidance:
            return None
        
        # 计算每个节点的启发式分数
        node_scores = []
        
        for i, node in enumerate(node_states):
            score = 0.0
            
            # 1. 考虑节点速度（速度越快分数越高）
            max_speed = max(n.speed for n in node_states)
            speed_score = node.speed / max_speed
            score += speed_score * 0.3
            
            # 2. 考虑节点当前负载（负载越低分数越高）
            load_score = 1.0 - node.current_load
            score += load_score * 0.25
            
            # 3. 考虑节点可用时间（可用时间越短分数越高）
            min_available_time = min(n.available_time for n in node_states)
            if min_available_time > 0:
                availability_score = min_available_time / node.available_time
            else:
                availability_score = 1.0 if node.available_time == 0 else 0.0
            score += availability_score * 0.2
            
            # 4. 考虑数据局部性（数据可用性越高分数越高）
            data_availability = node.data_availability.get(current_task.id, 0.0)
            score += data_availability * 0.15
            
            # 5. 考虑任务是否在关键路径上（关键路径任务优先分配到高性能节点）
            if current_task.is_critical_path:
                performance_score = node.speed / max_speed
                score += performance_score * 0.1
            
            node_scores.append((i, score))
        
        # 按分数排序，选择最佳节点
        node_scores.sort(key=lambda x: x[1], reverse=True)
        return node_scores[0][0]
    
    def _adjust_learning_rate(self, episode: int) -> float:
        """根据训练进度调整学习率"""
        for schedule in self.learning_rate_schedule:
            if episode <= schedule["episodes"]:
                return schedule["lr"]
        return self.learning_rate_schedule[-1]["lr"]
    
    def _should_advance_curriculum(self) -> bool:
        """判断是否应该进入下一个课程阶段"""
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return False  # 已经是最后一个阶段
        
        # 检查当前阶段是否完成足够的训练轮数
        curriculum = self.curriculum_stages[self.current_stage]
        if self.stage_episodes_completed >= curriculum["episodes"]:
            print(f"🎓 完成阶段 {self.current_stage+1} 的 {self.stage_episodes_completed} episodes, 准备进入下一阶段")
            return True
        
        return False
    
    def _advance_curriculum_stage(self):
        """进入下一个课程阶段"""
        self.current_stage += 1
        self.stage_episodes_completed = 0
        self.stage_performance_history = []
        
        if self.current_stage < len(self.curriculum_stages):
            curriculum = self.curriculum_stages[self.current_stage]
            print(f"🎓 进入课程阶段 {self.current_stage+1}: {curriculum['name']}")
            
            # 更新智能体的动作维度
            new_action_dim = curriculum["nodes"]
            if hasattr(self.agent, 'update_action_dim'):
                self.agent.update_action_dim(new_action_dim)
            
            # 保存当前模型作为阶段检查点
            stage_ckpt_path = Path(self.checkpoint_cfg.get('dir', 'models/checkpoints/')) / f"stage_{self.current_stage}.pth"
            self.save_model(str(stage_ckpt_path))
            
            # 重置探索率，适应新环境
            if hasattr(self.agent, 'epsilon'):
                self.agent.epsilon = max(0.1, self.agent.epsilon * 1.5)  # 增加探索率以适应新环境
        else:
            print("🎓 所有课程阶段已完成！")
    
    def _update_heuristic_weight(self):
        """更新启发式指导权重，随着训练进行逐渐减少"""
        if self.use_heuristic_guidance and self.heuristic_weight > 0.01:
            self.heuristic_weight *= self.heuristic_decay
    
    def _find_longest_path(self, task_states: List[TaskState], node_states: List[NodeState]) -> float:
        """找出任务图中的最长路径（关键路径）"""
        # 构建任务图
        graph = {task.id: [] for task in task_states}
        for task in task_states:
            for child_id in task.children:
                graph[task.id].append(child_id)
        
        # 找出所有没有父节点的任务（起始任务）
        start_tasks = [task.id for task in task_states if not task.parents]
        
        # 使用DFS找出最长路径
        def dfs(task_id, visited):
            if task_id in visited:
                return 0.0
            
            visited.add(task_id)
            task = next(t for t in task_states if t.id == task_id)
            
            # 计算当前任务的执行时间
            fastest_node = max(node_states, key=lambda n: n.speed)
            execution_time = task.computation_size / fastest_node.speed
            
            # 递归计算子任务的最长路径
            max_child_path = 0.0
            for child_id in graph[task_id]:
                child_path = dfs(child_id, visited.copy())
                max_child_path = max(max_child_path, child_path)
            
            return execution_time + max_child_path
        
        # 计算所有起始任务的最长路径
        max_path = 0.0
        for start_task in start_tasks:
            path_length = dfs(start_task, set())
            max_path = max(max_path, path_length)
        
        return max_path

    def extract_state_features(self, 
                             current_task: TaskState, 
                             node_states: List[NodeState],
                             environment: EnvironmentState) -> np.ndarray:
        """提取状态特征，增强版"""
        features = []
        
        # 任务特征 (5维)
        # 1. 计算大小归一化（使用对数归一化，更好地处理不同大小的任务）
        features.append(np.log1p(current_task.computation_size / 1e9) / 10.0)  # 使用log1p避免log(0)
        # 2. 父任务数量归一化
        features.append(len(current_task.parents) / 5.0)  # 假设最多5个父任务
        # 3. 子任务数量归一化
        features.append(len(current_task.children) / 5.0)  # 假设最多5个子任务
        # 4. 是否在关键路径上
        features.append(float(current_task.is_critical_path))
        # 5. 数据局部性分数
        features.append(current_task.data_locality_score)
        
        # 节点特征 (每个节点4维，共nodes×4维)
        max_speed = max(node.speed for node in node_states)
        max_available_time = max(node.available_time for node in node_states) or 1.0
        
        for node in node_states:
            # 1. 节点速度归一化
            features.append(node.speed / max_speed)
            # 2. 节点当前负载归一化
            features.append(node.current_load)
            # 3. 节点可用时间归一化
            features.append(node.available_time / max_available_time)
            # 4. 数据可用性
            features.append(node.data_availability.get(current_task.id, 0.0))
        
        # 如果节点数少于4个，用0填充剩余特征
        num_nodes = len(node_states)
        if num_nodes < 4:
            for _ in range(4 - num_nodes):
                # 为每个缺失的节点添加4个0值特征
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 环境特征 (6维)
        # 1. 工作流进度
        features.append(environment.workflow_progress)
        # 2. 当前时间归一化（相对于关键路径长度）
        features.append(environment.current_time / (environment.critical_path_length or 1.0))
        # 3. 待处理任务数量归一化
        features.append(len(environment.pending_tasks) / 20.0)
        # 4. 平均节点负载
        avg_load = np.mean([node.current_load for node in node_states])
        features.append(avg_load)
        # 5. 节点负载标准差（反映负载均衡情况）
        load_std = np.std([node.current_load for node in node_states])
        features.append(load_std)
        # 6. 关键路径进度
        critical_tasks = [t for t in environment.pending_tasks if t.is_critical_path]
        total_critical_tasks = len([t for t in environment.pending_tasks + [current_task] if t.is_critical_path])
        critical_progress = 1.0 - (len(critical_tasks) / (total_critical_tasks or 1.0))
        features.append(critical_progress)
        
        # 数据传输特征 (5维)
        # 1. 当前任务数据大小归一化
        task_data_size = getattr(current_task, 'data_size', 0.0)
        features.append(np.log1p(task_data_size / 1e6) / 10.0)  # 使用log归一化
        # 2. 平均数据传输时间估算
        if task_data_size > 0:
            # 估算从存储节点到计算节点的平均传输时间
            avg_transfer_time = task_data_size / (0.5 * 1e9)  # 转换为字节/秒，假设0.5GBps
            features.append(min(avg_transfer_time / 10.0, 1.0))  # 归一化并限制最大值
        else:
            features.append(0.0)
        # 3. 数据局部性差异（反映数据在不同节点的分布情况）
        data_availability_values = [node.data_availability.get(current_task.id, 0.0) for node in node_states]
        data_locality_variance = np.var(data_availability_values) if len(data_availability_values) > 1 else 0.0
        features.append(data_locality_variance)
        # 4. 最佳数据可用性（反映哪个节点有最好的数据局部性）
        best_data_availability = max(data_availability_values) if data_availability_values else 0.0
        features.append(best_data_availability)
        # 5. 数据传输瓶颈指标（反映数据传输是否可能成为瓶颈）
        if task_data_size > 0:
            computation_time = current_task.computation_size / max_speed
            transfer_time = task_data_size / (0.5 * 1e9)  # 假设0.5GBps
            bottleneck_ratio = transfer_time / (computation_time + transfer_time)
            features.append(bottleneck_ratio)
        else:
            features.append(0.0)
        
        # 总特征维度：5(任务) + 16(节点) + 6(环境) + 5(数据传输) = 32维
        return np.array(features, dtype=np.float32)
    
    def simulate_step(self, 
                     task: TaskState, 
                     action: int, 
                     node_states: List[NodeState],
                     environment: EnvironmentState) -> Tuple[float, EnvironmentState, bool]:
        """模拟一步执行，考虑数据传输开销"""
        chosen_node = node_states[action]
        
        # 计算执行时间
        execution_time = task.computation_size / chosen_node.speed
        
        # 计算数据传输时间（如果任务有数据）
        transfer_time = 0.0
        task_data_size = getattr(task, 'data_size', 0.0)
        if task_data_size > 0:
            # 获取数据可用性
            data_availability = chosen_node.data_availability.get(task.id, 0.0)
            
            # 如果数据不完全在本地，需要传输
            if data_availability < 1.0:
                # 计算需要传输的数据量
                data_to_transfer = task_data_size * (1.0 - data_availability)
                
                # 考虑网络带宽
                effective_bandwidth = 1e9  # 转换为字节/秒，假设1GBps
                
                # 计算传输时间
                transfer_time = data_to_transfer / effective_bandwidth
                
                # 添加网络延迟（简化模型：每个传输操作都有一固定延迟）
                transfer_time += 0.001  # 假设1ms延迟
        
        # 总时间 = 执行时间 + 传输时间
        total_time = execution_time + transfer_time
        
        # 更新环境
        new_environment = EnvironmentState(
            current_time=environment.current_time + total_time,
            pending_tasks=[t for t in environment.pending_tasks if t.id != task.id],
            node_states=node_states.copy(),  # 创建副本以避免修改原始状态
            workflow_progress=environment.workflow_progress + 1.0/20.0,
            critical_path_length=environment.critical_path_length
        )
        
        # 更新节点状态
        # 找到新环境中的对应节点并更新
        for node in new_environment.node_states:
            if node.id == chosen_node.id:
                # 更新负载（考虑执行时间和传输时间）
                node.current_load += 0.1 * (1.0 + transfer_time / (execution_time + 1e-6))
                # 更新可用时间
                node.available_time += total_time
                # 更新数据可用性（任务数据现在在节点上）
                node.data_availability[task.id] = 1.0
                break
        
        # 检查是否结束
        done = len(new_environment.pending_tasks) == 0
        return 0.0, new_environment, done
    
    def train_episode(self) -> Dict[str, float]:
        """训练一个episode，集成课程学习和启发式指导"""
        # 根据当前课程阶段创建环境
        curriculum = self.curriculum_stages[self.current_stage]
        environment, task_states, node_states = self.create_mock_environment(
            stage=self.current_stage
        )
        
        step_rewards = []  # 记录每步的奖励
        total_makespan = 0.0
        step_count = 0
        
        current_tasks = task_states.copy()

        reward_debug_path = self.logging_cfg.get('reward_debug', 'results/reward_debug.log')
        debug_file = None
        try:
            debug_file = open(reward_debug_path, 'a')
        except Exception:
            debug_file = None

        # 准备增强型 StepContext 统计
        total_cp_tasks = sum(1 for t in task_states if t.is_critical_path) or 1
        baseline_avg_wait = np.mean([n.available_time for n in node_states]) or 1.0
        completed_ids = set()
        task_map = {t.id: t for t in task_states}

        while current_tasks and step_count < 50:  # 限制最大步数
            # 选择当前任务（简化：按顺序）
            current_task = current_tasks[0]
            
            # 提取状态特征
            state = self.extract_state_features(current_task, node_states, environment)
            
            # 获取启发式动作建议
            heuristic_action = self._get_heuristic_action(current_task, node_states, environment)
            
            # 智能体选择动作
            action = self.agent.act(state, training=True)
            
            # 如果使用启发式指导，根据概率决定是否采用启发式动作
            if self.use_heuristic_guidance and heuristic_action is not None and np.random.random() < self.heuristic_weight:
                # 使用启发式动作，但给予部分奖励以鼓励探索
                original_action = action
                action = heuristic_action
                
                # 给予额外奖励以鼓励智能体学习启发式策略
                heuristic_bonus = 0.1 * self.heuristic_weight
            else:
                heuristic_bonus = 0.0
            
            # 执行动作
            _, new_environment, done = self.simulate_step(
                current_task, action, node_states, environment
            )
            # 更新完成集
            completed_ids.add(current_task.id)

            # 构造增强 StepContext
            try:
                completed_cp = sum(1 for cid in completed_ids if task_map[cid].is_critical_path)
                ctx = StepContext(
                    completed_critical_path_tasks=completed_cp,
                    total_critical_path_tasks=total_cp_tasks,
                    node_busy_times={n.id: n.current_load for n in node_states},
                    ready_task_count=len(current_tasks)-1,  # 去掉当前即将调度的
                    total_nodes=len(node_states),
                    avg_queue_wait=np.mean([n.available_time for n in node_states]),
                    queue_wait_baseline=baseline_avg_wait
                )
                # 预测当前makespan和基准makespan
                predicted_makespan = total_makespan * (1.0 + (len(current_tasks) / len(task_states)))
                # 使用历史最佳makespan作为基准，如果没有则使用固定值
                baseline_makespan = self.best_makespan if self.best_makespan != float('inf') else 100.0
                step_reward, _metrics = compute_step_reward(ctx, predicted_makespan, baseline_makespan, debug_writer=debug_file)
                
                # 添加启发式奖励
                step_reward += heuristic_bonus
            except Exception:
                step_reward = 0.0
            
            # 记录步骤奖励
            step_rewards.append(step_reward)
            
            # 提取下一状态特征
            if len(current_tasks) > 1:
                next_task = current_tasks[1]
                next_state = self.extract_state_features(next_task, node_states, new_environment)
            else:
                next_state = np.zeros_like(state)
            
            # 存储经验
            self.agent.remember(state, action, step_reward, next_state, done)
            
            # 更新状态
            environment = new_environment
            current_tasks = current_tasks[1:]
            total_makespan = environment.current_time
            step_count += 1
            
            # 训练智能体
            if step_count % 4 == 0:  # 每4步训练一次
                loss = self.agent.replay()
            
            if done:
                break

        # 计算最终奖励 (新 makespan 稀疏奖励)
        episode_stats = EpisodeStats(makespan=total_makespan)
        final_reward, final_metrics = compute_final_reward(
            makespan=total_makespan,
            stats=episode_stats,
            temperature=0.75,
            baseline_makespan=self.baseline_makespan if hasattr(self, 'baseline_makespan') else None
        )
        
        # 计算平均奖励而不是累加奖励
        avg_step_reward = np.mean(step_rewards) if step_rewards else 0.0
        total_reward = avg_step_reward + final_reward  # 平均步骤奖励 + 最终奖励
        
        # 更新最佳makespan记录
        if total_makespan < self.best_makespan:
            self.best_makespan = total_makespan
        
        # 记录阶段性能历史
        self.stage_performance_history.append({
            'makespan': total_makespan,
            'reward': total_reward,
            'step_count': step_count
        })
        
        # 更新课程阶段计数
        self.stage_episodes_completed += 1
        
        # 更新启发式权重
        self._update_heuristic_weight()
        
        if debug_file:
            try:
                debug_file.write(f"FINAL\tstage={self.current_stage+1}\tmakespan={total_makespan:.4f}\tavg_step_reward={avg_step_reward:.4f}\tfinal_reward={final_reward:.4f}\ttotal_reward={total_reward:.4f}\n")
                debug_file.close()
            except Exception:
                pass

        return {
            'total_reward': total_reward,
            'avg_step_reward': avg_step_reward,
            'final_reward': final_reward,
            'makespan': total_makespan,
            'step_count': step_count,
            'epsilon': self.agent.epsilon,
            'stage': self.current_stage,
            'heuristic_weight': self.heuristic_weight
        }
    
    def train(self, episodes: int = 1000) -> Dict[str, Any]:
        """训练DRL智能体，使用课程学习策略和启发式指导"""
        print(f"🚀 开始高级DRL训练: {episodes} episodes (配置 episodes={self.drl_cfg.get('episodes', episodes)})")
        print(f"📊 训练特性: 高级网络架构，课程学习策略，启发式指导")
        print(f"🎓 课程阶段: {len(self.curriculum_stages)}个阶段，当前阶段: {self.current_stage+1}")
        
        # 初始化智能体
        state_dim = 32  # 更新后的特征维度：5(任务) + 16(节点) + 6(环境) + 5(数据传输)
        curriculum = self.curriculum_stages[self.current_stage]
        action_dim = curriculum["nodes"]  # 根据当前课程阶段的节点数设置动作维度
        
        # 获取DRL配置参数
        drl_config = self.config.get('drl', {}).copy()
        
        # 只提取ImprovedDQNAgent接受的参数
        agent_params = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'learning_rate': drl_config.get('learning_rate', 0.001),
            'epsilon_start': drl_config.get('epsilon_start', 1.0),
            'epsilon_end': drl_config.get('epsilon_end', 0.1),
            'epsilon_decay': drl_config.get('epsilon_decay', 0.995),
            'gamma': drl_config.get('gamma', 0.99),
            'memory_size': drl_config.get('memory_size', 10000),
            'batch_size': drl_config.get('batch_size', 64),
            'target_update_freq': drl_config.get('target_update_freq', 100),
            'network_type': "advanced",  # 使用高级网络
            'hidden_dims': [512, 256, 128, 64],  # 更深的网络结构
            'exploration_strategy': "adaptive",  # 使用自适应探索策略
            'use_ucb': True,  # 启用UCB探索
            'ucb_c': 2.0,  # UCB置信度参数
            'use_boltzmann': True,  # 启用玻尔兹曼探索
            'boltzmann_tau': 1.0,  # 初始温度参数
        }
        
        # 使用高级网络架构和多样化探索策略
        self.agent = ImprovedDQNAgent(**agent_params)
        
        # 训练循环
        best_makespan = float('inf')
        recent_rewards = deque(maxlen=100)
        recent_losses = deque(maxlen=100)
        
        log_interval = self.drl_cfg.get('log_interval', 50)
        eval_interval = self.drl_cfg.get('eval_interval', 100)
        checkpoint_interval = self.drl_cfg.get('checkpoint_interval', 100)
        rolling_window = self.drl_cfg.get('rolling_window', 100)
        metrics_path = self.logging_cfg.get('metrics_file', 'results/training_metrics.jsonl')
        ckpt_dir = Path(self.checkpoint_cfg.get('dir', 'models/checkpoints/'))
        keep_last = self.checkpoint_cfg.get('keep_last', 5)
        save_best = self.checkpoint_cfg.get('save_best', True)
        kept_ckpts = []

        for episode in range(episodes):
            # 动态调整学习率
            current_lr = self._adjust_learning_rate(episode)
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            episode_results = self.train_episode()
            self.training_history.append(episode_results)
            recent_rewards.append(episode_results['total_reward'])
            
            # 获取智能体性能统计
            perf_stats = self.agent.get_performance_stats()
            if perf_stats.get('avg_loss') is not None:
                recent_losses.append(perf_stats['avg_loss'])
            
            # 更新最佳性能
            if episode_results['makespan'] < best_makespan:
                best_makespan = episode_results['makespan']
            
            # 获取智能体探索统计
            exploration_stats = self.agent.get_exploration_stats()
            
            # 打印进度
            if episode % log_interval == 0:
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                curriculum = self.curriculum_stages[self.current_stage]
                print(f"Episode {episode}: "
                      f"阶段={self.current_stage+1}/{len(self.curriculum_stages)} ({curriculum['name']}), "
                      f"平均奖励={avg_reward:.3f}, "
                      f"平均损失={avg_loss:.4f}, "
                      f"Makespan={episode_results['makespan']:.2f}, "
                      f"最佳Makespan={best_makespan:.2f}, "
                      f"探索率={exploration_stats.get('exploration_rate', 0.0):.3f}, "
                      f"利用率={exploration_stats.get('exploitation_rate', 0.0):.3f}, "
                      f"ε={exploration_stats.get('epsilon', 0.0):.3f}, "
                      f"温度={exploration_stats.get('boltzmann_tau', 0.0):.3f}, "
                      f"Q值={perf_stats.get('avg_q_value', 0.0):.3f}, "
                      f"启发式权重={episode_results['heuristic_weight']:.3f}, "
                      f"学习率={current_lr:.6f}")
            
            # 写入流式指标日志
            try:
                with open(metrics_path, 'a') as mf:
                    mf.write(json.dumps({
                        'episode': episode,
                        'stage': self.current_stage,
                        'stage_name': curriculum['name'],
                        'reward': episode_results['total_reward'],
                        'makespan': episode_results['makespan'],
                        'best_makespan': best_makespan,
                        'exploration_rate': exploration_stats.get('exploration_rate', 0.0),
                        'exploitation_rate': exploration_stats.get('exploitation_rate', 0.0),
                        'epsilon': exploration_stats.get('epsilon', 0.0),
                        'boltzmann_tau': exploration_stats.get('boltzmann_tau', 0.0),
                        'heuristic_weight': episode_results['heuristic_weight'],
                        'learning_rate': current_lr,
                        'avg_loss': perf_stats.get('avg_loss', 0.0),
                        'avg_q_value': perf_stats.get('avg_q_value', 0.0),
                        'network_type': perf_stats.get('network_type', 'unknown'),
                        'exploration_strategy': exploration_stats.get('strategy', 'unknown'),
                        'timestamp': time.time()
                    }) + '\n')
            except Exception as e:
                print(f"⚠️ 写入指标日志失败: {e}")
            
            # 更新探索参数
            self.agent.update_exploration_parameters(episode_results['total_reward'])
            
            # 检查是否应该进入下一个课程阶段
            if self._should_advance_curriculum():
                self._advance_curriculum_stage()
            
            # 保存检查点
            if episode % checkpoint_interval == 0 and episode > 0:
                ckpt_path = ckpt_dir / f"checkpoint_ep{episode}.pth"
                self.save_model(str(ckpt_path))
                kept_ckpts.append(ckpt_path)
                
                # 清理旧检查点
                if len(kept_ckpts) > keep_last:
                    try:
                        oldest = kept_ckpts.pop(0)
                        oldest.unlink(missing_ok=True)
                    except Exception:
                        pass
                
                # 保存最佳模型
                if save_best and episode_results['makespan'] == best_makespan:
                    best_path = ckpt_dir / "best_model.pth"
                    self.save_model(str(best_path))
                    print(f"💾 保存最佳模型 (makespan={best_makespan:.2f})")
        
        # 训练完成摘要
        curriculum = self.curriculum_stages[self.current_stage]
        exploration_stats = self.agent.get_exploration_stats()
        
        summary = {
            'episodes_trained': episodes,
            'best_makespan': best_makespan,
            'final_stage': self.current_stage,
            'final_stage_name': curriculum['name'],
            'final_reward': recent_rewards[-1] if recent_rewards else 0,
            'final_epsilon': exploration_stats.get('epsilon', 0.0),
            'final_exploration_rate': exploration_stats.get('exploration_rate', 0.0),
            'final_exploitation_rate': exploration_stats.get('exploitation_rate', 0.0),
            'final_boltzmann_tau': exploration_stats.get('boltzmann_tau', 0.0),
            'final_heuristic_weight': episode_results['heuristic_weight'],
            'network_type': 'advanced',
            'exploration_strategy': exploration_stats.get('strategy', 'adaptive'),
            'curriculum_learning': True,
            'heuristic_guidance': self.use_heuristic_guidance
        }
        
        print(f"✅ 高级DRL训练完成！最佳makespan: {best_makespan:.2f}")
        print(f"🎓 完成课程阶段: {self.current_stage+1}/{len(self.curriculum_stages)} ({curriculum['name']})")
        print(f"🔍 最终探索策略: {exploration_stats.get('strategy', 'adaptive')}")
        print(f"📊 最终探索率: {exploration_stats.get('exploration_rate', 0.0):.3f}")
        print(f"📊 最终利用率: {exploration_stats.get('exploitation_rate', 0.0):.3f}")
        
        # 保存最终模型
        final_path = ckpt_dir / "final_model.pth"
        self.save_model(str(final_path))
        print(f"💾 保存最终模型到 {final_path}")
        
        return summary
    
    def save_model(self, model_path: str):
        """保存训练好的模型"""
        if self.agent is None:
            raise ValueError("智能体尚未初始化或训练")
        
        checkpoint = {
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'training_step': self.agent.training_step,
            'epsilon': self.agent.epsilon,
            'config': self.config,
            'training_history': self.training_history,
            'exploration_params': {
                'exploration_strategy': self.agent.exploration_strategy,
                'use_ucb': self.agent.use_ucb,
                'ucb_c': self.agent.ucb_c,
                'use_boltzmann': self.agent.use_boltzmann,
                'boltzmann_tau': self.agent.boltzmann_tau,
                'action_counts': self.agent.action_counts.tolist() if hasattr(self.agent.action_counts, 'tolist') else self.agent.action_counts,
                'action_values': self.agent.action_values.tolist() if hasattr(self.agent.action_values, 'tolist') else self.agent.action_values,
            },
            'metadata': {
                'state_dim': self.agent.state_dim,
                'action_dim': self.agent.action_dim,
                'training_completed': True,
                'final_performance': {
                    'avg_makespan': np.mean([h['makespan'] for h in self.training_history[-100:]]),
                    'best_makespan': min(h['makespan'] for h in self.training_history)
                }
            }
        }
        
        torch.save(checkpoint, model_path)
        print(f"📁 模型已保存: {model_path}")
        
        # 保存训练历史
        history_path = model_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"📊 训练历史已保存: {history_path}")

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def main():
    # Set seed for reproducibility
    set_seed(42)

    import argparse

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    
    parser = argparse.ArgumentParser(description='WASS-RAG 改进DRL训练器 (集成多样化探索策略)')
    parser.add_argument('--config', default='configs/experiment.yaml', help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=1000, help='训练episode数')
    parser.add_argument('--output', default='models/improved_wass_drl.pth', help='输出模型路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化训练器
    trainer = WRENCHDRLTrainer(args.config)
    
    # 训练
    results = trainer.train(args.episodes)
    
    # 保存模型
    trainer.save_model(args.output)
    
    # 保存训练结果摘要
    summary_path = args.output.replace('.pth', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"🎉 训练完成! 模型和结果已保存到 {Path(args.output).parent}")
    print(f"📈 训练特性: 集成多样化探索策略和makespan预测奖励")

if __name__ == "__main__":
    main()
