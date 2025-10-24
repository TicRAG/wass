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
import xml.etree.ElementTree as ET

# --- Path fix ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------

from src.workflows.manager import WorkflowManager
from src.drl.gnn_encoder import GNNEncoder
from src.drl.agent import ActorCritic
from src.drl.ppo import PPOTrainer, PPOConfig
from src.drl.replay_buffer import ReplayBuffer
from src.simulation.schedulers import WASS_RAG_Scheduler_Trainable
import joblib
from src.simulation.experiment_runner import WrenchExperimentRunner
from src.utils.config import load_training_config

WORKFLOW_CONFIG_FILE = "configs/workflow_config.yaml"

def _merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base or {})
    merged.update(override or {})
    return merged


TRAINING_CFG = load_training_config()
COMMON_CFG = TRAINING_CFG.get("common", {})
DRL_CFG = TRAINING_CFG.get("drl_training", {})

GNN_CFG = _merge_dicts(COMMON_CFG.get("gnn", {}), DRL_CFG.get("gnn", {}))
PPO_CFG = _merge_dicts(COMMON_CFG.get("ppo", {}), DRL_CFG.get("ppo", {}))
MODEL_CFG = DRL_CFG.get("model", {})
REWARD_CFG = DRL_CFG.get("reward_scaling", {})

GNN_IN_CHANNELS = int(GNN_CFG.get("in_channels", 4))
GNN_HIDDEN_CHANNELS = int(GNN_CFG.get("hidden_channels", 64))
GNN_OUT_CHANNELS = int(GNN_CFG.get("out_channels", 32))

LEARNING_RATE = float(PPO_CFG.get("learning_rate", 3e-4))
GAMMA = float(PPO_CFG.get("gamma", 0.99))
EPOCHS = int(PPO_CFG.get("epochs", 10))
EPS_CLIP = float(PPO_CFG.get("eps_clip", 0.2))

TOTAL_EPISODES = int(DRL_CFG.get("total_episodes", 200))
SAVE_INTERVAL = int(DRL_CFG.get("save_interval", 50)) if DRL_CFG.get("save_interval") is not None else 0

MODEL_SAVE_DIR = MODEL_CFG.get("save_dir", "models/saved_models")
MODEL_FILENAME = MODEL_CFG.get("filename", "drl_agent_no_rag.pth")
AGENT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

MAKESPAN_NORMALIZER = float(REWARD_CFG.get("makespan_normalizer", 1000.0)) or 1.0


def infer_action_dim(platform_path: str) -> int:
    tree = ET.parse(platform_path)
    host_ids = {
        host.get('id')
        for host in tree.getroot().iter('host')
        if host.get('id') not in {"ControllerHost", "StorageHost"}
    }
    if not host_ids:
        raise ValueError(f"No compute hosts found in platform XML: {platform_path}")
    return len(host_ids)


import argparse

def parse_args():
    ap = argparse.ArgumentParser(description="Train DRL scheduler (no RAG) with unified PPO.")
    ap.add_argument('--max_episodes', type=int, default=None, help='Override TOTAL_EPISODES for quick runs.')
    ap.add_argument('--reward_mode', choices=['dense','final'], default='final', help='Reward shaping mode: dense (discount across steps) or final (single terminal reward).')
    return ap.parse_args()

def main():
    args = parse_args()
    print("🚀 [Phase 3.1] Starting DRL Agent Training (NO RAG)...")
    
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n[Step 1/4] Initializing components...")
    workflow_manager = WorkflowManager(WORKFLOW_CONFIG_FILE)
    platform_file = workflow_manager.get_platform_file()
    action_dim = infer_action_dim(platform_file)
    gnn_encoder = GNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    # Load feature scaler for consistency with KB seeding
    feature_scaler = None
    scaler_path = Path("models/saved_models/feature_scaler.joblib")
    if scaler_path.exists():
        try:
            feature_scaler = joblib.load(scaler_path)
            print(f"🔄 Loaded feature scaler from {scaler_path}")
        except Exception as e:
            print(f"⚠️ Failed to load feature scaler ({e}); proceeding without scaling.")
    else:
        print(f"⚠️ Feature scaler not found at {scaler_path}; proceeding without scaling.")
    
    state_dim = GNN_OUT_CHANNELS
    policy_agent = ActorCritic(state_dim=state_dim, action_dim=action_dim, gnn_encoder=gnn_encoder)
    ppo_cfg = PPOConfig(gamma=GAMMA, epochs=EPOCHS, eps_clip=EPS_CLIP, reward_mode=args.reward_mode)
    ppo_updater = PPOTrainer(policy_agent, Adam(policy_agent.parameters(), lr=LEARNING_RATE), ppo_cfg)
    
    replay_buffer = ReplayBuffer()  # stores raw graph states for PPO re-embedding
    
    config_params = {"platform_file": platform_file}
    wrench_runner = WrenchExperimentRunner(schedulers={}, config=config_params)
    print("✅ Components initialized.")

    print("\n[Step 2/4] Loading converted wfcommons training workflows (data/workflows/training)...")
    workflows_dir = Path("data/workflows/training")
    training_workflows = sorted(str(p) for p in workflows_dir.glob("*.json"))
    if not training_workflows:
        print(f"❌ No training workflows found in {workflows_dir}. Ensure files are placed under data/workflows/training.")
        return
    print(f"✅ Loaded {len(training_workflows)} converted workflows.")

    effective_total = args.max_episodes if args.max_episodes is not None else TOTAL_EPISODES
    print(f"\n[Step 3/4] Starting main training loop... total_episodes={effective_total} mode={args.reward_mode}")
    for episode in range(1, effective_total + 1):
        workflow_file = np.random.choice(training_workflows)
        
        # --- THIS IS THE FIX: The lambda now accepts all keyword arguments from the caller ---
        trainable_scheduler_factory = lambda simulation, compute_services, hosts, workflow_obj, workflow_file: WASS_RAG_Scheduler_Trainable(
            simulation=simulation,
            compute_services=compute_services,
            hosts=hosts,
            workflow_obj=workflow_obj,
            agent=policy_agent,
            teacher=None,
            replay_buffer=replay_buffer,
            policy_gnn_encoder=gnn_encoder,
            workflow_file=workflow_file,
            rag_gnn_encoder=None,
            feature_scaler=feature_scaler
        )
        
        makespan, _ = wrench_runner.run_single_seeding_simulation(
            scheduler_class=trainable_scheduler_factory,
            workflow_file=workflow_file
        )

        if makespan < 0 or not replay_buffer.actions:
            print(f"  Episode {episode}, Workflow: {Path(workflow_file).name}, Status: FAILED. Skipping.")
            replay_buffer.clear()
            continue

        final_reward = -makespan / MAKESPAN_NORMALIZER
        if args.reward_mode == 'final':
            replay_buffer.rewards = [torch.tensor(final_reward)]
        else:
            steps = len(replay_buffer.actions)
            if steps <= 1:
                replay_buffer.rewards = [torch.tensor(final_reward)]
            else:
                per_step = final_reward / steps
                replay_buffer.rewards = [torch.tensor(per_step) for _ in range(steps)]

        # PPO update (now re-embeds graph states via policy_agent.gnn_encoder)
        ppo_updater.update(replay_buffer)
        replay_buffer.clear()

        if args.reward_mode == 'final':
            reported_reward = final_reward
        else:
            reported_reward = final_reward  # aggregate of per-step rewards equals final_reward
        print(f"  Episode {episode}, Workflow: {Path(workflow_file).name}, Makespan: {makespan:.2f}s, Reward: {reported_reward:.4f}")

        if SAVE_INTERVAL and episode % SAVE_INTERVAL == 0:
            torch.save(policy_agent.state_dict(), AGENT_MODEL_PATH)
            print(f"💾 NO-RAG Model saved at episode {episode}")

    print("\n[Step 4/4] Training finished.")
    print(f"✅ Final NO-RAG model saved to: {AGENT_MODEL_PATH}")
    print("\n🎉 [Phase 3.1] DRL Agent (NO RAG) Training Completed! 🎉")

if __name__ == "__main__":
    main()