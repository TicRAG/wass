# src/drl/replay_buffer.py
import torch

class ReplayBuffer:
    """
    一个简单的经验回放池，用于收集一个episode内的轨迹数据。
    PPO的更新是on-policy的，所以每个episode结束后会清空buffer。
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def add(self, state: torch.Tensor, action: int, logprob: torch.Tensor, reward: float):
        """添加一条经验"""
        self.states.append(state)
        self.actions.append(torch.tensor(action))
        self.logprobs.append(logprob)
        self.rewards.append(torch.tensor(reward))

    def get_all(self) -> tuple[list, list, list, list]:
        """获取所有存储的经验"""
        return self.states, self.actions, self.logprobs, self.rewards

    def clear(self):
        """清空缓存"""
        self.__init__()