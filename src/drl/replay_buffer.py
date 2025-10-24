# src/drl/replay_buffer.py
import torch
from typing import List, Union, Any
try:
    from torch_geometric.data import Data  # type: ignore
except ImportError:
    Data = Any  # fallback placeholder

class ReplayBuffer:
    """Single-episode on-policy storage for PPO.

    Refactored: store raw PyG ``Data`` graphs (or embedding tensors) instead of only detached embeddings
    so we can re-encode with the current GNN to maintain gradient flow.
    """
    def __init__(self):
        # Raw graph states (PyG Data) or embedding tensors preserved for re-encoding
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def add(self, state, action, logprob, reward):
        self.states.append(state)
        self.actions.append(torch.tensor(action, dtype=torch.long))
        self.logprobs.append(logprob if isinstance(logprob, torch.Tensor) else torch.tensor(logprob, dtype=torch.float32))
        self.rewards.append(reward if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32))

    def __len__(self):
        return len(self.actions)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()