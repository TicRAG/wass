import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, List

@dataclass
class PPOConfig:
    gamma: float = 0.99
    epochs: int = 10
    eps_clip: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    reward_mode: Literal['dense', 'final'] = 'dense'  # 'dense' uses provided per-step rewards; 'final' uses only last reward

class PPOTrainer:
    def __init__(self, policy_net: nn.Module, optimizer, config: PPOConfig):
        self.policy = policy_net
        self.optimizer = optimizer
        self.cfg = config
        self.mse_loss = nn.MSELoss()

    def _build_returns(self, rewards: List[torch.Tensor]) -> torch.Tensor:
        if not rewards:
            return torch.zeros(0)
        # Standard (dense) discounted returns over provided reward sequence
        discounted = []
        running = torch.zeros(1)
        for r in reversed(rewards):
            r_t = r if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.float32)
            running = r_t + self.cfg.gamma * running
            discounted.insert(0, running)
        return torch.stack(discounted)

    def update(self, memory) -> None:
        rewards = memory.rewards  # list[Tensor]
        if not rewards:
            return
        # Build returns depending on reward shaping mode
        if self.cfg.reward_mode == 'final':
            # Sparse terminal reward: discount it back across all steps
            T_steps = len(memory.actions)
            if T_steps == 0:
                return
            final_r = rewards[-1].detach() if isinstance(rewards[-1], torch.Tensor) else torch.tensor(rewards[-1], dtype=torch.float32)
            # returns[t] = gamma^(T-1 - t) * final_r
            returns_vals = [final_r * (self.cfg.gamma ** (T_steps - 1 - t)) for t in range(T_steps)]
            returns = torch.stack(returns_vals)
        else:
            returns = self._build_returns(rewards).detach()  # shape: [reward_len]
        # States now are precomputed embeddings (Tensor)
        old_actions = torch.stack(memory.actions)
        old_logprobs = torch.stack(memory.logprobs).detach()

        T_returns = returns.shape[0]
        T_steps = old_actions.shape[0]

        # Align lengths:
        if T_returns != T_steps:
            # Truncate or pad (pad with last) to match when using dense mode or unexpected mismatch
            if T_returns > T_steps:
                returns = returns[:T_steps]
            else:
                pad_val = returns[-1]
                pad = pad_val.repeat(T_steps - T_returns)
                returns = torch.cat([returns, pad], dim=0)

        # Normalize for stability if >1
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        target_values_full = returns.view(-1, 1)  # [T,1]

        # Re-embed graphs if raw Data objects stored; preserve gradient flow through GNN encoder
        reencoded_states = []
        for s in memory.states:
            if hasattr(self.policy, 'gnn_encoder') and self.policy.gnn_encoder is not None:
                try:
                    from torch_geometric.data import Data  # type: ignore
                    if isinstance(s, Data):
                        # Clone to avoid in-place status mutations interfering with autograd versioning
                        cloned = s.clone()
                        if hasattr(cloned, 'x'):
                            cloned.x = cloned.x.clone()
                        emb = self.policy.gnn_encoder(cloned)
                        reencoded_states.append(emb)
                        continue
                except Exception:
                    pass
            # Fallback: assume s already an embedding tensor
            reencoded_states.append(s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32))
        # We only want gradients flowing once through the GNN; multiple backward passes on the same
        # graph cause 'backward through the graph a second time'. Strategy:
        # 1. First epoch uses full graph with grad.
        # 2. Subsequent epochs reuse detached embeddings (no second backward through GNN).
        embedded_states_full = torch.cat(reencoded_states)
        original_embedded_states = embedded_states_full
        for epoch_i in range(self.cfg.epochs):
            if epoch_i == 0:
                embedded_states = original_embedded_states  # gradient flows through encoder
            else:
                embedded_states = original_embedded_states.detach()  # cut graph for later epochs
            logprobs, state_values, dist_entropy = self.policy.evaluate(embedded_states, old_actions)
            # Ensure state_values shape matches
            if state_values.shape[0] != target_values_full.shape[0]:
                min_len = min(state_values.shape[0], target_values_full.shape[0])
                state_values_use = state_values[:min_len]
                target_use = target_values_full[:min_len]
                logprobs_use = logprobs[:min_len]
                old_logprobs_use = old_logprobs[:min_len]
            else:
                state_values_use = state_values
                target_use = target_values_full
                logprobs_use = logprobs
                old_logprobs_use = old_logprobs

            advantages = (target_use - state_values_use.detach())
            ratios = torch.exp(logprobs_use - old_logprobs_use)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mse_loss(state_values_use, target_use)
            entropy_loss = -dist_entropy.mean()
            loss = policy_loss + self.cfg.value_coeff * value_loss + self.cfg.entropy_coeff * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
