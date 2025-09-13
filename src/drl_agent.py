import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class SchedulingState:
    def __init__(self, features: np.ndarray):
        self.features = features


class DQNAgent:
    """简化版的DQN Agent，支持奖励机制"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
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
    
    def compute_reward(self, teacher_makespan: float, student_makespan: float) -> float:
        """计算RAG奖励：教师预测与当前选择的差异"""
        return teacher_makespan - student_makespan
    
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
        
    def save(self, path: str):
        """保存模型"""
        torch.save(self.q_network.state_dict(), path)
        
    def load(self, path: str):
        """加载模型"""
        self.q_network.load_state_dict(torch.load(path))