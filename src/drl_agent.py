import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

# [FIX 1] 恢复评估脚本所需的 SchedulingState 数据类
@dataclass
class SchedulingState:
    """一个用来封装DRL智能体决策时所需状态信息的数据结构。"""
    task_features: np.ndarray
    node_features: np.ndarray
    
    def to_array(self) -> np.ndarray:
        """将所有状态特征拼接成一个扁平的numpy数组。"""
        return np.concatenate([self.task_features, self.node_features.flatten()]).astype(np.float32)


class DQNAgent(nn.Module):
    """
    统一的、简单的DQN智能体模型。
    训练脚本和评估脚本都使用这个完全相同的定义。
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        layers = []
        current_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU()])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, action_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """根据状态选择动作。"""
        if np.random.rand() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state_tensor)
                return q_values.argmax().item()
        else:
            action_dim = self.network[-1].out_features
            return np.random.randint(0, action_dim)