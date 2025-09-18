# train.py
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from pathlib import Path
import json

# --- 修正导入路径问题 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------

from scripts.workflow_manager import WorkflowManager
from src.drl.gnn_encoder import GNNEncoder
from src.drl.ppo_agent import ActorCritic
from src.drl.replay_buffer import ReplayBuffer
from src.drl.knowledge_teacher import KnowledgeBase, KnowledgeableTeacher
from src.wrench_schedulers import WASS_RAG_Scheduler_Trainable
from src.utils import WrenchExperimentRunner


# --- 配置 ---
GNN_IN_CHANNELS = 4
GNN_HIDDEN_CHANNELS = 64
GNN_OUT_CHANNELS = 32
ACTION_DIM = 4
LEARNING_RATE = 3e-4
GAMMA = 0.99
EPOCHS = 10
EPS_CLIP = 0.2
TOTAL_EPISODES = 500
MODEL_SAVE_DIR = "models/saved_models"
AGENT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "drl_agent.pth")
PLATFORM_FILE = "configs/test_platform.xml"
WORKFLOW_CONFIG_FILE = "configs/workflow_config.yaml"

class PPO:
    """处理PPO更新的独立类。"""
    def __init__(self, policy_net, lr, gamma, epochs, eps_clip):
        self.policy = policy_net
        self.optimizer = Adam(policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epochs = epochs
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            # 确保 reward 是一个 tensor
            reward_tensor = reward if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32)
            discounted_reward = reward_tensor + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
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
    print("🚀 [Phase 3] Starting DRL Agent Training (with Dynamic State Update)...")
    
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n[Step 1/4] Initializing components...")
    workflow_manager = WorkflowManager(WORKFLOW_CONFIG_FILE)
    gnn_encoder = GNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    
    state_dim = GNN_OUT_CHANNELS
    policy_agent = ActorCritic(state_dim=state_dim, action_dim=ACTION_DIM)
    ppo_updater = PPO(policy_agent, LEARNING_RATE, GAMMA, EPOCHS, EPS_CLIP)
    
    replay_buffer = ReplayBuffer()
    
    print("🧠 Initializing Knowledge Base and Teacher...")
    kb = KnowledgeBase(dimension=GNN_OUT_CHANNELS)
    teacher = KnowledgeableTeacher(state_dim=state_dim, knowledge_base=kb)
    
    config_params = {"platform_file": PLATFORM_FILE}
    wrench_runner = WrenchExperimentRunner(schedulers={}, config=config_params)
    print("✅ Components initialized.")

    print("\n[Step 2/4] Generating a pool of training workflows...")
    training_workflows = workflow_manager.generate_training_workflows()
    if not training_workflows:
        print("❌ No training workflows generated. Please check your config.")
        return
    print(f"✅ Generated {len(training_workflows)} workflows for training pool.")

    print("\n[Step 3/4] Starting main training loop...")
    for episode in range(1, TOTAL_EPISODES + 1):
        workflow_file = np.random.choice(training_workflows)
        
        trainable_scheduler = lambda sim, cs, h: WASS_RAG_Scheduler_Trainable(
            simulation=sim, 
            compute_services=cs, 
            hosts=h,
            agent=policy_agent,
            teacher=teacher,
            replay_buffer=replay_buffer,
            gnn_encoder=gnn_encoder,
            workflow_file=workflow_file
        )

        makespan, _ = wrench_runner.run_single_seeding_simulation(
            scheduler_class=trainable_scheduler,
            workflow_file=workflow_file
        )

        if makespan < 0 or not replay_buffer.rewards:
            print(f"  Episode {episode}, Workflow: {Path(workflow_file).name}, Status: FAILED. Skipping update.")
            replay_buffer.clear()
            continue

        # --- 核心修改：在更新前计算平均RAG奖励 ---
        # 提取所有RAG奖励（除了最后一个可能是终局奖励）
        rag_rewards = [r.item() for r in replay_buffer.rewards]
        avg_rag_reward = np.mean(rag_rewards) if rag_rewards else 0
        
        # 终局奖励：负的makespan
        final_reward = -makespan / 1000.0
        # 将其加到最后一步的奖励上
        replay_buffer.rewards[-1] = torch.tensor(replay_buffer.rewards[-1].item() + final_reward)
        # --- 修改结束 ---
        
        ppo_updater.update(replay_buffer)
        replay_buffer.clear()

        print(f"  Episode {episode}, Workflow: {Path(workflow_file).name}, Makespan: {makespan:.2f}s, Avg RAG Reward: {avg_rag_reward:.4f}")

        if episode % 50 == 0:
            torch.save(policy_agent.state_dict(), AGENT_MODEL_PATH)
            print(f"💾 Model saved at episode {episode}")

    print("\n[Step 4/4] Training finished.")
    print(f"✅ Final model saved to: {AGENT_MODEL_PATH}")
    print("\n🎉 [Phase 3] DRL Agent Training Completed! 🎉")

if __name__ == "__main__":
    main()