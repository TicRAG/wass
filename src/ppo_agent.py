import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

@dataclass
class PPOTrajectory:
    states: List[torch.Tensor]
    actions: List[torch.Tensor]
    log_probs: List[torch.Tensor]
    rewards: List[torch.Tensor]
    dones: List[torch.Tensor]
    values: List[torch.Tensor]

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x):
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = cfg.get('gamma', 0.99)
        self.lam = cfg.get('gae_lambda', 0.95)
        self.clip_eps = cfg.get('clip_eps', 0.2)
        self.entropy_coef = cfg.get('entropy_coef', 0.01)
        self.value_coef = cfg.get('value_coef', 0.5)
        self.lr = cfg.get('learning_rate', 3e-4)
        self.batch_size = cfg.get('ppo_batch_size', 64)
        self.epochs = cfg.get('ppo_update_epochs', 4)
        self.model = ActorCritic(state_dim, action_dim, hidden=cfg.get('hidden_dim', 256)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, state: torch.Tensor):
        state = state.to(self.device)
        logits, value = self.model(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).detach(), value.detach()

    def compute_advantages(self, rewards, values, dones):
        advantages: List[torch.Tensor] = []
        gae = 0.0
        values = [v.view(1) for v in values] + [torch.zeros(1, device=self.device)]
        for t in reversed(range(len(rewards))):
            r_t = rewards[t]
            v_t = values[t]
            v_next = values[t + 1]
            d_t = dones[t]
            delta = r_t + self.gamma * v_next * (1 - d_t) - v_t
            gae = delta + self.gamma * self.lam * (1 - d_t) * gae
            advantages.insert(0, gae.view(1))
        advantages_tensor = torch.cat(advantages, dim=0)
        returns = advantages_tensor + torch.cat(values[:-1], dim=0)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        return advantages_tensor, returns

    def update(self, traj: PPOTrajectory):
        states = torch.stack(traj.states).to(self.device)
        actions = torch.stack(traj.actions).to(self.device)
        old_log_probs = torch.stack(traj.log_probs).to(self.device)
        rewards = torch.stack(traj.rewards).to(self.device).view(-1)
        dones = torch.stack(traj.dones).to(self.device).view(-1)
        values = [v.to(self.device) for v in traj.values]

        advantages, returns = self.compute_advantages(list(rewards), values, list(dones))

        dataset_size = states.size(0)
        idx = torch.arange(dataset_size)
        last_policy_loss = last_value_loss = last_entropy = torch.tensor(0.0)
        for _ in range(self.epochs):
            perm = idx[torch.randperm(dataset_size)]
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logp = old_log_probs[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

                logits, value = self.model(b_states)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_actions)
                ratio = torch.exp(logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                target_value = b_ret if b_ret.dim()==1 else b_ret.view(-1)
                # Optional value clipping (like PPO-clip for value)
                with torch.no_grad():
                    old_value = value.detach()
                clipped_value = old_value + torch.clamp(value - old_value, -0.2, 0.2)
                v_loss_unclipped = (value - target_value).pow(2)
                v_loss_clipped = (clipped_value - target_value).pow(2)
                value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self, 'max_grad_norm', 1.0))
                self.optimizer.step()

                last_policy_loss = policy_loss.detach()
                last_value_loss = value_loss.detach()
                last_entropy = entropy.detach()

        return {
            'policy_loss': float(last_policy_loss.cpu()),
            'value_loss': float(last_value_loss.cpu()),
            'entropy': float(last_entropy.cpu())
        }
