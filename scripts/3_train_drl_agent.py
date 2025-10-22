import os
import sys

# --- 路径修正 ---
# 将项目根目录 (上一级目录) 添加到 Python 的 sys.path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from pathlib import Path
import json

# --- Path fix ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------

from src.workflows.manager import WorkflowManager
from src.drl.gnn_encoder import GNNEncoder
from src.drl.agent import ActorCritic
from src.drl.replay_buffer import ReplayBuffer
from src.simulation.schedulers import WASS_RAG_Scheduler_Trainable
from src.simulation.experiment_runner import WrenchExperimentRunner

# --- Config ---
GNN_IN_CHANNELS = 4
GNN_HIDDEN_CHANNELS = 64
GNN_OUT_CHANNELS = 32
ACTION_DIM = 4
LEARNING_RATE = 3e-4
GAMMA = 0.99
EPOCHS = 10
EPS_CLIP = 0.2
TOTAL_EPISODES = 200
MODEL_SAVE_DIR = "models/saved_models"
AGENT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "drl_agent_no_rag.pth")
WORKFLOW_CONFIG_FILE = "configs/workflow_config.yaml"

class PPO:
    """Handles the PPO update."""
    def __init__(self, policy_net, lr, gamma, epochs, eps_clip):
        self.policy = policy_net
        self.optimizer = Adam(policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epochs = epochs
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        # Reward calculation is based only on the final makespan
        final_reward = memory.rewards[0]
        rewards = [final_reward] * len(memory.actions)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        # Normalize rewards only if there's more than one to avoid division by zero
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.cat(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        for _ in range(self.epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            rewards_reshaped = rewards.view(-1, 1)
            advantages = rewards_reshaped - state_values.detach()
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards_reshaped) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def main():
    print("🚀 [Phase 3.1] Starting DRL Agent Training (NO RAG)...")
    
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n[Step 1/4] Initializing components...")
    workflow_manager = WorkflowManager(WORKFLOW_CONFIG_FILE)
    platform_file = workflow_manager.get_platform_file()
    gnn_encoder = GNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    
    state_dim = GNN_OUT_CHANNELS
    policy_agent = ActorCritic(state_dim=state_dim, action_dim=ACTION_DIM)
    ppo_updater = PPO(policy_agent, LEARNING_RATE, GAMMA, EPOCHS, EPS_CLIP)
    
    replay_buffer = ReplayBuffer()
    
    config_params = {"platform_file": platform_file}
    wrench_runner = WrenchExperimentRunner(schedulers={}, config=config_params)
    print("✅ Components initialized.")

    print("\n[Step 2/4] Generating a pool of training workflows...")
    training_workflows = workflow_manager.generate_training_workflows()
    if not training_workflows:
        print("❌ No training workflows generated.")
        return
    print(f"✅ Generated {len(training_workflows)} workflows.")

    print("\n[Step 3/4] Starting main training loop...")
    for episode in range(1, TOTAL_EPISODES + 1):
        workflow_file = np.random.choice(training_workflows)
        
        # --- THIS IS THE FIX: The lambda now accepts all keyword arguments from the caller ---
        trainable_scheduler_factory = lambda simulation, compute_services, hosts, workflow_obj, workflow_file: WASS_RAG_Scheduler_Trainable(
            simulation=simulation,
            compute_services=compute_services,
            hosts=hosts,
            workflow_obj=workflow_obj,
            agent=policy_agent,
            teacher=None, # Teacher is disabled
            replay_buffer=replay_buffer,
            gnn_encoder=gnn_encoder,
            workflow_file=workflow_file
        )
        
        makespan, _ = wrench_runner.run_single_seeding_simulation(
            scheduler_class=trainable_scheduler_factory,
            workflow_file=workflow_file
        )

        if makespan < 0 or not replay_buffer.actions:
            print(f"  Episode {episode}, Workflow: {Path(workflow_file).name}, Status: FAILED. Skipping.")
            replay_buffer.clear()
            continue

        reward = -makespan / 1000.0
        replay_buffer.rewards = [torch.tensor(reward)]
        
        ppo_updater.update(replay_buffer)
        replay_buffer.clear()

        print(f"  Episode {episode}, Workflow: {Path(workflow_file).name}, Makespan: {makespan:.2f}s, Reward: {reward:.4f}")

        if episode % 50 == 0:
            torch.save(policy_agent.state_dict(), AGENT_MODEL_PATH)
            print(f"💾 NO-RAG Model saved at episode {episode}")

    print("\n[Step 4/4] Training finished.")
    print(f"✅ Final NO-RAG model saved to: {AGENT_MODEL_PATH}")
    print("\n🎉 [Phase 3.1] DRL Agent (NO RAG) Training Completed! 🎉")

if __name__ == "__main__":
    main()