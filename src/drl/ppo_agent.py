# src/drl/ppo_agent.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    PPO算法的Actor-Critic网络。
    - Actor (策略网络): 输入状态，输出一个动作概率分布。
    - Critic (价值网络): 输入状态，输出该状态的评价值 (Value)。
    """
    def __init__(self, state_dim: int, action_dim: int):
        """
        初始化Actor-Critic网络。
        Args:
            state_dim (int): 输入状态向量的维度 (应与GNNEncoder的输出维度一致)。
            action_dim (int): 动作空间的维度 (即可选的主机数量)。
        """
        super(ActorCritic, self).__init__()

        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # 输出动作的概率
        )

        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出一个标量评价值
        )

    def act(self, state: torch.Tensor, deterministic: bool = False) -> tuple[int, torch.Tensor]:
        """
        根据状态选择一个动作。
        Args:
            state (torch.Tensor): 当前状态的嵌入向量。
            deterministic (bool): 是否采用确定性策略 (选择概率最高的动作)，用于评估阶段。
        Returns:
            tuple[int, torch.Tensor]: 选择的动作索引, 该动作的对数概率。
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        if deterministic:
            action = torch.argmax(action_probs).item()
        else:
            action = dist.sample()

        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在PPO更新时，评估给定的状态和动作。
        Args:
            state (torch.Tensor): 一批状态。
            action (torch.Tensor): 一批动作。
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 动作的对数概率, 状态价值, 分布的熵。
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy