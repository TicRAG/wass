import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any
from collections import deque
import random
from src.reward_fix import RewardFix


class SchedulingState:
    def __init__(self, features: np.ndarray):
        self.features = features


class DQNAgent:
    """修复后的DQN Agent，支持改进的奖励机制"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001, 
                 buffer_size: int = 10000, batch_size: int = 32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=buffer_size)
        
        # 修复后的奖励计算器
        self.reward_fix = RewardFix()
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        # 初始化目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 训练统计
        self.episode_count = 0
        self.reward_history = []
        
    def act(self, state: SchedulingState, use_teacher: bool = False, teacher_action: Optional[int] = None, epsilon: float = 0.1) -> int:
        """选择动作，使用epsilon-greedy探索"""
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0)
        
        if use_teacher and teacher_action is not None:
            # 在RAG模式下，使用教师指导
            return teacher_action
        
        # Epsilon-greedy探索
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def compute_reward(self, teacher_makespan: float, student_makespan: float, 
                       task_scale: float = 1.0, simulation: Optional[Any] = None, 
                       task: Optional[Any] = None, chosen_node: Optional[str] = None) -> float:
        """
        修复后的奖励计算，使用归一化和多目标设计
        
        Args:
            teacher_makespan: 老师（预测器）建议的makespan
            student_makespan: 学生（Agent）选择的makespan
            task_scale: 任务规模，用于归一化
            simulation: 仿真环境（可选）
            task: 当前任务（可选）
            chosen_node: 选择的节点（可选）
        
        Returns:
            计算后的奖励值
        """
        # 如果提供了仿真环境和任务信息，使用多目标奖励
        if simulation is not None and task is not None and chosen_node is not None:
            reward = self.reward_fix.calculate_multi_objective_reward(
                simulation, task, chosen_node, teacher_makespan, student_makespan
            )
        else:
            # 否则使用基本的归一化奖励
            reward = self.reward_fix.calculate_normalized_reward(
                teacher_makespan, student_makespan, task_scale
            )
        
        # 记录奖励历史
        self.reward_history.append(reward)
        
        # 记录调试信息
        task_id = getattr(task, 'id', 'unknown') if task else 'unknown'
        self.reward_fix.debug_reward_info(task_id, teacher_makespan, student_makespan, reward)
        
        return reward
    
    def store_transition(self, state: SchedulingState, action: int, reward: float, next_state: Optional[SchedulingState] = None, done: bool = False):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, training_progress: float = 0.0):
        """
        修复后的经验回放学习，使用目标网络
        
        Args:
            training_progress: 训练进度 [0, 1]
        """
        if len(self.memory) < self.batch_size:
            return
            
        # 采样一批经验
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e[0].features for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # 处理下一个状态
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, [e[3] for e in batch])), dtype=torch.bool)
        non_final_next_states = torch.FloatTensor(np.array([e[3].features for e in batch if e[3] is not None]))
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值（使用目标网络）
        next_q_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            if non_final_next_states.size(0) > 0:
                next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        
        # 根据训练进度调整折扣因子
        gamma = 0.99 - 0.09 * training_progress  # 从0.99逐渐减少到0.9
        
        target_q_values = (rewards + gamma * next_q_values * ~dones)
        
        # 计算损失并更新
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def learn(self, state: SchedulingState, action: int, reward: float, next_state: Optional[SchedulingState] = None):
        """学习更新"""
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        target_q_values = q_values.clone()
        
        # 简化的Q学习更新
        target = reward
        if next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state.features).unsqueeze(0)
            with torch.no_grad():
                next_q_values = self.q_network(next_state_tensor)
                target += 0.99 * next_q_values.max().item()
        
        target_q_values[0, action] = target
        
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_q_values(self, state: SchedulingState) -> np.ndarray:
        """获取当前状态的Q值"""
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.numpy()[0]
    
    def get_average_reward(self, window_size: int = 100) -> float:
        """获取最近window_size个回合的平均奖励"""
        if len(self.reward_history) < window_size:
            return np.mean(self.reward_history) if self.reward_history else 0.0
        return np.mean(self.reward_history[-window_size:])
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'reward_history': self.reward_history
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.reward_history = checkpoint.get('reward_history', [])