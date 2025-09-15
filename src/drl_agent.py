import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, List
from collections import deque, namedtuple
import random

# From improved_drl_trainer.py
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

class SchedulingState:
    def __init__(self, features: np.ndarray):
        self.features = features

# Renamed from ImprovedDQNAgent
class DQNAgent:
    """改进的DQN智能体，支持多种网络架构"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 gamma: float = 0.99,
                 memory_size: int = 10000,
                 batch_size: int = 32,
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

    def act(self, state, training=True, epsilon=0.1):
        """选择动作，支持多种探索策略"""
        if not training:
            # 在测试阶段，直接选择最优动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def get_q_values(self, state: SchedulingState) -> np.ndarray:
        """获取当前状态的Q值"""
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]

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
        states = torch.FloatTensor(np.array([e[0].features for e in batch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # 处理下一个状态
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, [e[3] for e in batch])), dtype=torch.bool, device=self.device)
        non_final_next_states_list = [e[3].features for e in batch if e[3] is not None]
        if not non_final_next_states_list:
            return # No non-final states in batch

        non_final_next_states = torch.FloatTensor(np.array(non_final_next_states_list)).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值（使用目标网络）
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_next_states.size(0) > 0:
                next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        
        # 根据训练进度调整折扣因子
        gamma = 0.99 - 0.09 * training_progress  # 从0.99逐渐减少到0.9
        
        target_q_values = (rewards + gamma * next_q_values * ~dones)
        
        # 计算损失并更新
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str, network_type="advanced", state_dim=None, action_dim=None, hidden_dims=None):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Re-create networks if dimensions differ
        if state_dim is not None and action_dim is not None:
            if self.state_dim != state_dim or self.action_dim != action_dim:
                print(f"ℹ️ Rebuilding model to match checkpoint state_dim={state_dim} action_dim={action_dim}")
                self.state_dim = state_dim
                self.action_dim = action_dim
                if network_type == "advanced":
                    self.q_network = AdvancedDQN(state_dim, action_dim, hidden_dims).to(self.device)
                    self.target_network = AdvancedDQN(state_dim, action_dim, hidden_dims).to(self.device)
                else:
                    self.q_network = ImprovedDQN(state_dim, action_dim, hidden_dims).to(self.device)
                    self.target_network = ImprovedDQN(state_dim, action_dim, hidden_dims).to(self.device)
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.optimizer.param_groups[0]['lr'])

        try:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Model loaded successfully (strict).")
        except RuntimeError as e:
            print(f"⚠️ Strict loading failed, trying partial load: {e}")
            # Partial load
            q_state_dict = checkpoint.get('q_network_state_dict', checkpoint)
            self.q_network.load_state_dict(q_state_dict, strict=False)
            if 'target_network_state_dict' in checkpoint:
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'], strict=False)
            else: # Fallback for older models
                self.target_network.load_state_dict(q_state_dict, strict=False)
            
            # Report missing/unexpected keys
            q_model_keys = set(self.q_network.state_dict().keys())
            q_checkpoint_keys = set(q_state_dict.keys())
            missing_keys = q_model_keys - q_checkpoint_keys
            unexpected_keys = q_checkpoint_keys - q_model_keys
            if missing_keys:
                print(f"  - Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"  - Unexpected keys: {len(unexpected_keys)}")
            print(f"✅ Partial load completed.")
