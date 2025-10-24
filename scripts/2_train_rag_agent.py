import os
import sys
import time
import argparse

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
import joblib

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
from src.rag.teacher import KnowledgeBase, KnowledgeableTeacher
from src.simulation.schedulers import WASS_RAG_Scheduler_Trainable
from src.simulation.experiment_runner import WrenchExperimentRunner
from src.utils.config import load_training_config


WORKFLOW_CONFIG_FILE = "configs/workflow_config.yaml"

def _merge_dicts(base: dict, override: dict) -> dict:
    """Return a shallow merge of two dictionaries with override precedence."""
    merged = dict(base or {})
    merged.update(override or {})
    return merged


TRAINING_CFG = load_training_config()
COMMON_CFG = TRAINING_CFG.get("common", {})
RAG_CFG = TRAINING_CFG.get("rag_training", {})

GNN_CFG = _merge_dicts(COMMON_CFG.get("gnn", {}), RAG_CFG.get("gnn", {}))
PPO_CFG = _merge_dicts(COMMON_CFG.get("ppo", {}), RAG_CFG.get("ppo", {}))
MODEL_CFG = RAG_CFG.get("model", {})
REWARD_CFG = RAG_CFG.get("reward_scaling", {})
TEACHER_CFG = RAG_CFG.get("teacher", {})

GNN_IN_CHANNELS = int(GNN_CFG.get("in_channels", 4))
GNN_HIDDEN_CHANNELS = int(GNN_CFG.get("hidden_channels", 64))
GNN_OUT_CHANNELS = int(GNN_CFG.get("out_channels", 32))

LEARNING_RATE = float(PPO_CFG.get("learning_rate", 3e-4))
GAMMA = float(PPO_CFG.get("gamma", 0.99))
EPOCHS = int(PPO_CFG.get("epochs", 10))
EPS_CLIP = float(PPO_CFG.get("eps_clip", 0.2))

TOTAL_EPISODES = int(RAG_CFG.get("total_episodes", 200))
SAVE_INTERVAL = int(RAG_CFG.get("save_interval", 50)) if RAG_CFG.get("save_interval") is not None else 0

MODEL_SAVE_DIR = MODEL_CFG.get("save_dir", "models/saved_models")
MODEL_FILENAME = MODEL_CFG.get("filename", "drl_agent.pth")
AGENT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

RAG_REWARD_MULTIPLIER = float(REWARD_CFG.get("rag_multiplier", 10.0))
FINAL_REWARD_NORMALIZER = float(REWARD_CFG.get("final_normalizer", 5000.0))
GNN_SEED_WEIGHTS_PATH = "models/saved_models/gnn_encoder_kb.pth"


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


FEATURE_SCALER_PATH = "models/saved_models/feature_scaler.joblib"

def parse_args():
    parser = argparse.ArgumentParser(description="Train RAG-enabled scheduling agent.")
    parser.add_argument('--max_tasks', type=int, default=None, help='Skip workflows with task count greater than this value.')
    parser.add_argument('--max_episodes', type=int, default=None, help='Override TOTAL_EPISODES for quick diagnostic runs.')
    parser.add_argument('--profile', action='store_true', help='Print timing for major phases to diagnose hangs.')
    parser.add_argument('--freeze_gnn', action='store_true', help='Freeze GNN encoder parameters to avoid embedding drift relative to KB.')
    parser.add_argument('--kb_refresh_interval', type=int, default=0, help='If >0, periodically rebuild KB entries using current GNN every N episodes (costly).')
    parser.add_argument('--reward_mode', choices=['dense','final'], default='dense', help='dense: use per-task RAG rewards + discount; final: only final aggregated reward.')
    parser.add_argument('--grad_check', action='store_true', help='Perform a one-off gradient flow check for policy GNN before PPO update.')
    parser.add_argument('--include_aug', action='store_true', help='Include augmented workflows from data/workflows/training_aug.')
    return parser.parse_args()


def count_tasks_in_workflow(path: str) -> int:
    try:
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        wf = data.get('workflow', {})
        if 'tasks' in wf and isinstance(wf['tasks'], list):
            return len(wf['tasks'])
        spec_tasks = wf.get('specification', {}).get('tasks', [])
        return len(spec_tasks) if isinstance(spec_tasks, list) else 0
    except Exception:
        return -1


def main():
    args = parse_args()
    print("🚀 [Phase 3] Starting DRL Agent Training (with Re-balanced Rewards)...")
    
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n[Step 1/4] Initializing components...")
    workflow_manager = WorkflowManager(WORKFLOW_CONFIG_FILE)
    platform_file = workflow_manager.get_platform_file()
    action_dim = infer_action_dim(platform_file)
    # Dual encoders: policy (trainable) vs rag (frozen, matches KB space)
    policy_gnn_encoder = GNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    rag_gnn_encoder = GNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    if Path(GNN_SEED_WEIGHTS_PATH).exists():
        try:
            rag_gnn_encoder.load_state_dict(torch.load(GNN_SEED_WEIGHTS_PATH, map_location=torch.device('cpu')))
            print(f"🔐 Loaded seed GNN weights into frozen RAG encoder from {GNN_SEED_WEIGHTS_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load seed GNN weights ({e}); continuing with random init.")
    for p in rag_gnn_encoder.parameters():
        p.requires_grad = False
    rag_gnn_encoder.eval()
    if args.freeze_gnn:
        for p in policy_gnn_encoder.parameters():
            p.requires_grad = False
        print("🧊 Policy GNN encoder frozen; using identical frozen encoder for RAG queries.")
    else:
        print("🔀 Using separate trainable policy GNN and frozen RAG GNN (pre-seeding weights).")
    state_dim = GNN_OUT_CHANNELS
    policy_agent = ActorCritic(state_dim=state_dim, action_dim=action_dim, gnn_encoder=policy_gnn_encoder)
    ppo_cfg = PPOConfig(gamma=GAMMA, epochs=EPOCHS, eps_clip=EPS_CLIP, reward_mode=args.reward_mode)
    ppo_updater = PPOTrainer(policy_agent, Adam(policy_agent.parameters(), lr=LEARNING_RATE), ppo_cfg)
    replay_buffer = ReplayBuffer()  # now stores raw PyG Data graphs (not detached embeddings)
    
    print("🧠 Initializing Knowledge Base and Teacher...")
    kb = KnowledgeBase(dimension=GNN_OUT_CHANNELS)
    teacher = KnowledgeableTeacher(state_dim=state_dim, knowledge_base=kb, gnn_encoder=rag_gnn_encoder, reward_config=TEACHER_CFG)

    # Load feature scaler for consistent preprocessing with seeding phase
    feature_scaler = None
    if Path(FEATURE_SCALER_PATH).exists():
        try:
            feature_scaler = joblib.load(FEATURE_SCALER_PATH)
            print(f"🔄 Loaded feature scaler from {FEATURE_SCALER_PATH} for embedding normalization.")
        except Exception as e:
            print(f"⚠️ Failed to load feature scaler ({e}); proceeding without scaling.")
    else:
        print(f"⚠️ Feature scaler file not found at {FEATURE_SCALER_PATH}; proceeding without scaling.")
    
    config_params = {"platform_file": platform_file}
    wrench_runner = WrenchExperimentRunner(schedulers={}, config=config_params)
    print("✅ Components initialized.")

    t_load_start = time.time()
    print("\n[Step 2/4] Loading converted wfcommons training workflows (data/workflows/training)...")
    workflows_dir = Path("data/workflows/training")
    training_workflows_all = sorted(str(p) for p in workflows_dir.glob("*.json"))
    # Filter by max_tasks if requested
    if args.max_tasks is not None:
        filtered = []
        for wf_path in training_workflows_all:
            tc = count_tasks_in_workflow(wf_path)
            if tc < 0:
                print(f"[WARN] Could not parse {wf_path}, skipping.")
                continue
            if tc <= args.max_tasks:
                filtered.append(wf_path)
            else:
                print(f"[SKIP] {Path(wf_path).name} task_count={tc} > max_tasks={args.max_tasks}")
        training_workflows = filtered
    else:
        training_workflows = training_workflows_all
    if not training_workflows:
        print(f"❌ No training workflows found in {workflows_dir}. Ensure files are placed under data/workflows/training.")
        return

    # Optionally include augmented workflows
    if args.include_aug:
        aug_dir = Path("data/workflows/training_aug")
        aug_files = sorted(str(p) for p in aug_dir.glob("*.json"))
        if aug_files:
            # Deduplicate while preserving order
            training_workflows = list(dict.fromkeys(training_workflows + aug_files))
            # Avoid surrogate escape issues in some terminals; use ASCII indicator
            print(f"[AUG] Included {len(aug_files)} augmented workflows. Total training workflow count: {len(training_workflows)}")
        else:
            print(f"\u26a0\ufe0f No augmented workflows found in {aug_dir}; continuing without them.")
    load_elapsed = time.time() - t_load_start
    print(f"✅ Loaded {len(training_workflows)} workflows (elapsed {load_elapsed:.2f}s).")
    if args.profile:
        for wf in training_workflows:
            print(f"    • {Path(wf).name} tasks={count_tasks_in_workflow(wf)}")

    effective_total_episodes = args.max_episodes if args.max_episodes is not None else TOTAL_EPISODES
    print(f"\n[Step 3/4] Starting main training loop... total_episodes={effective_total_episodes}")
    loop_start = time.time()
    for episode in range(1, effective_total_episodes + 1):
        workflow_file = np.random.choice(training_workflows)
        task_count_selected = count_tasks_in_workflow(workflow_file)
        if args.profile:
            print(f"[EP] {episode}/{effective_total_episodes} -> {Path(workflow_file).name} tasks={task_count_selected}")
        
        # --- THIS IS THE FIX: The lambda now accepts all keyword arguments from the caller ---
        # Inject feature_scaler & gnn_encoder so scheduler/teacher can produce scaled embeddings
        trainable_scheduler_factory = lambda simulation, compute_services, hosts, workflow_obj, workflow_file: WASS_RAG_Scheduler_Trainable(
            simulation=simulation,
            compute_services=compute_services,
            hosts=hosts,
            workflow_obj=workflow_obj,
            agent=policy_agent,
            teacher=teacher,
            replay_buffer=replay_buffer,
            policy_gnn_encoder=policy_gnn_encoder,
            rag_gnn_encoder=rag_gnn_encoder,
            workflow_file=workflow_file,
            feature_scaler=feature_scaler
        )
        
        t_sim_start = time.time()
        makespan, _ = wrench_runner.run_single_seeding_simulation(
            scheduler_class=trainable_scheduler_factory,
            workflow_file=workflow_file
        )
        sim_elapsed = time.time() - t_sim_start

        if makespan < 0 or not replay_buffer.rewards:
            print(f"  Episode {episode}, Workflow: {Path(workflow_file).name}, Status: FAILED. Skipping update.")
            replay_buffer.clear()
            continue

        rag_rewards_for_logging = [r.item() for r in replay_buffer.rewards]
        avg_rag_reward = np.mean(rag_rewards_for_logging) if rag_rewards_for_logging else 0.0
        # Additional reward distribution diagnostics (Option A):
        if rag_rewards_for_logging:
            min_rag = float(np.min(rag_rewards_for_logging))
            max_rag = float(np.max(rag_rewards_for_logging))
            std_rag = float(np.std(rag_rewards_for_logging))
            clamped_floor = TEACHER_CFG.get('min_clamped_reward', -0.05)
            clamped_count = sum(1 for v in rag_rewards_for_logging if v <= clamped_floor + 1e-9)
            clamped_pct = (clamped_count / len(rag_rewards_for_logging)) * 100.0
        else:
            min_rag = max_rag = std_rag = 0.0
            clamped_count = clamped_pct = 0.0

        normalizer = FINAL_REWARD_NORMALIZER if FINAL_REWARD_NORMALIZER != 0 else 1.0
        final_penalty = - (makespan / normalizer)
        if args.reward_mode == 'dense':
            # Distribute final penalty evenly across all collected RAG rewards for fairness
            steps = len(replay_buffer.rewards)
            if steps == 0:
                replay_buffer.rewards = [torch.tensor(final_penalty)]
            else:
                per_step_penalty = final_penalty / steps
                replay_buffer.rewards = [torch.tensor(r.item() + per_step_penalty) for r in replay_buffer.rewards]
        else:
            # Final mode collapses RAG intermediates + final penalty to single scalar
            total_rag = sum(r.item() for r in replay_buffer.rewards) if replay_buffer.rewards else 0.0
            combined = total_rag + final_penalty
            replay_buffer.rewards = [torch.tensor(combined)]
        
        # Gradient norm diagnostics (now meaningful because graphs are re-embedded during PPO update)
        if not args.freeze_gnn:
            with torch.no_grad():
                pre_norm = sum(p.norm().item() for p in policy_gnn_encoder.parameters() if p.requires_grad)
        ppo_updater.update(replay_buffer)
        if not args.freeze_gnn:
            with torch.no_grad():
                post_norm = sum(p.norm().item() for p in policy_gnn_encoder.parameters() if p.requires_grad)
            print(f"    [GradCheck] GNN param norm: pre={pre_norm:.4f} post={post_norm:.4f} delta={(post_norm - pre_norm):.4f}")
        replay_buffer.clear()

        # Optional periodic KB refresh (expensive: re-encode all workflows with current GNN)
        if args.kb_refresh_interval and episode % args.kb_refresh_interval == 0:
            if args.freeze_gnn:
                print("[KB Refresh] Skipped because GNN is frozen (no drift).")
            else:
                try:
                    print(f"[KB Refresh] Rebuilding Knowledge Base embeddings at episode {episode}...")
                    from src.drl.utils import workflow_json_to_pyg_data
                    new_vectors = []
                    new_meta = []
                    for wf in training_workflows:
                        try:
                            data_obj = workflow_json_to_pyg_data(wf, feature_scaler)
                            emb = policy_gnn_encoder(data_obj).detach().cpu().numpy().flatten()
                            new_vectors.append(emb)
                            new_meta.append({
                                'workflow_file': Path(wf).name,
                                'scheduler_used': 'HEFT',
                                'makespan': 0.0,
                                'decisions': '{}'
                            })
                        except Exception as e:
                            print(f"  [KB Refresh] Failed to encode {wf}: {e}")
                    if new_vectors:
                        # Replace KB (re-init FAISS index for simplicity)
                        kb.index.reset()
                        kb.metadata = kb.metadata.iloc[0:0]
                        # Use already imported numpy at module level (avoid shadowing / UnboundLocal)
                        kb.add(np.array(new_vectors, dtype=np.float32), new_meta)
                        kb.save()
                        print(f"[KB Refresh] Completed with {len(new_vectors)} embeddings.")
                    else:
                        print("[KB Refresh] No embeddings produced; KB unchanged.")
                except Exception as e:
                    print(f"[KB Refresh] Failed: {e}")

        print(
            f"  Episode {episode}, Workflow: {Path(workflow_file).name}, tasks={task_count_selected}, Makespan: {makespan:.2f}s, "
            f"AvgRAG={avg_rag_reward:.4f} min={min_rag:.4f} max={max_rag:.4f} std={std_rag:.4f} "
            f"clamped={clamped_count} ({clamped_pct:.1f}%), sim_time={sim_elapsed:.2f}s"
        )

        if SAVE_INTERVAL and episode % SAVE_INTERVAL == 0:
            torch.save(policy_agent.state_dict(), AGENT_MODEL_PATH)
            print(f"💾 Model saved at episode {episode}")

    total_loop_elapsed = time.time() - loop_start
    print(f"\n[Step 4/4] Training finished. Total loop time: {total_loop_elapsed:.2f}s")
    print(f"✅ Final model saved to: {AGENT_MODEL_PATH}")
    print("\n🎉 [Phase 3] DRL Agent Training Completed! 🎉")

if __name__ == "__main__":
    main()