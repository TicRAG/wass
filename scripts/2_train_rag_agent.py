import os
import sys
import time
import argparse
import random
from collections import defaultdict, deque
from datetime import datetime

# --- 路径修正 ---
# 将项目根目录 (上一级目录) 添加到 Python 的 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import joblib

# --- Path fix ---
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# -----------------

from src.workflows.manager import WorkflowManager
from src.drl.gnn_encoder import DecoupledGNNEncoder
from src.drl.agent import ActorCritic
from src.drl.ppo import PPOTrainer, PPOConfig
from src.drl.replay_buffer import ReplayBuffer
from src.rag.teacher import KnowledgeBase, KnowledgeableTeacher
from src.simulation.schedulers import WASS_RAG_Scheduler_Trainable
from src.simulation.experiment_runner import WrenchExperimentRunner
from src.utils.config import load_training_config
from src.utils.reward_normalizer import FamilyRewardNormalizer
from src.utils.training_logger import TrainingLogger


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

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, MODEL_CFG.get("save_dir", "models/saved_models"))
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
    parser.add_argument('--disable_rag', action='store_true', help='Disable RAG-based potential shaping and fall back to vanilla DRL rewards.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed used for workflow sampling and PPO updates.')
    parser.add_argument('--log_dir', default="results/training_runs", help='Directory where per-episode metrics will be recorded.')
    parser.add_argument('--run_label', default=None, help='Custom label for this training run (overrides strategy name in logs).')
    parser.add_argument('--trace_log_dir', default=None, help='If set, write interpretability trace JSONL logs to this directory.')
    parser.add_argument('--randomize_hosts', action='store_true', help='Shuffle host ordering each episode to reduce index-specific bias.')
    parser.add_argument('--resume-from', dest='resume_from', default=None, help='Optional path to an existing policy checkpoint for warm-start fine-tuning.')
    parser.add_argument(
        '--family-filter',
        nargs='+',
        default=None,
        help='Optional whitelist of workflow families to train on (case-insensitive).',
    )
    parser.add_argument('--rag-multiplier', type=float, default=None, help='Override dense reward multiplier applied after normalization.')
    parser.add_argument('--final-normalizer', type=float, default=None, help='Override final makespan reward normalizer (base 5000).')
    parser.add_argument('--teacher-lambda', type=float, default=None, help='Override teacher lambda scaling factor.')
    parser.add_argument('--teacher-top-k', type=int, default=None, help='Override teacher neighbor top-k during retrieval.')
    parser.add_argument('--teacher-temperature', type=float, default=None, help='Override teacher softmax temperature for similarity weighting.')
    parser.add_argument('--teacher-gamma', type=float, default=None, help='Override teacher exponential moving average gamma.')
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
    trace_log_path: Path | None = None

    rag_multiplier = RAG_REWARD_MULTIPLIER if args.rag_multiplier is None else float(args.rag_multiplier)
    if args.rag_multiplier is not None:
        print(f"[Override] rag_multiplier set to {rag_multiplier:.4f}")
    final_normalizer = FINAL_REWARD_NORMALIZER if args.final_normalizer is None else float(args.final_normalizer)
    if args.final_normalizer is not None:
        print(f"[Override] final_normalizer set to {final_normalizer:.4f}")
    teacher_config = dict(TEACHER_CFG)
    if args.teacher_lambda is not None:
        teacher_config["lambda"] = float(args.teacher_lambda)
        print(f"[Override] teacher.lambda set to {teacher_config['lambda']:.4f}")
    if args.teacher_top_k is not None:
        teacher_config["top_k"] = int(args.teacher_top_k)
        print(f"[Override] teacher.top_k set to {teacher_config['top_k']}")
    if args.teacher_temperature is not None:
        teacher_config["temperature"] = float(args.teacher_temperature)
        print(f"[Override] teacher.temperature set to {teacher_config['temperature']:.4f}")
    if args.teacher_gamma is not None:
        teacher_config["gamma"] = float(args.teacher_gamma)
        print(f"[Override] teacher.gamma set to {teacher_config['gamma']:.4f}")
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n[Step 1/4] Initializing components...")
    workflow_manager = WorkflowManager(WORKFLOW_CONFIG_FILE)
    platform_file = workflow_manager.get_platform_file()
    action_dim = infer_action_dim(platform_file)
    # Dual encoders: policy (trainable) vs rag (frozen, matches KB space)
    print(f"🧮 Inferred action dimension from platform '{platform_file}': {action_dim}")
    dual_gnn_encoder = DecoupledGNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    policy_gnn_encoder = dual_gnn_encoder.policy_encoder
    rag_gnn_encoder = dual_gnn_encoder.retrieval_encoder
    if Path(GNN_SEED_WEIGHTS_PATH).exists():
        try:
            state_dict = torch.load(GNN_SEED_WEIGHTS_PATH, map_location=torch.device('cpu'))
            policy_gnn_encoder.load_state_dict(state_dict)
            dual_gnn_encoder.sync_retrieval_encoder()
            print(f"🔐 Loaded seed GNN weights into policy encoder and mirrored retrieval encoder from {GNN_SEED_WEIGHTS_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load seed GNN weights ({e}); continuing with random init.")
    dual_gnn_encoder.freeze_retrieval_encoder()
    if args.freeze_gnn:
        dual_gnn_encoder.freeze_policy_encoder()
        print("🧊 Policy GNN encoder frozen; both encoders remain static for drift-free experiments.")
    else:
        print("🔀 Using separate trainable policy GNN and frozen RAG GNN (pre-seeding weights).")
    total_policy_params = sum(param.numel() for param in policy_gnn_encoder.parameters())
    trainable_policy_params = sum(param.numel() for param in policy_gnn_encoder.parameters() if param.requires_grad)
    trainable_retrieval_params = sum(param.numel() for param in rag_gnn_encoder.parameters() if param.requires_grad)
    print(
        f"🧮 Encoder status -> policy trainable {trainable_policy_params}/{total_policy_params} params; "
        f"retrieval trainable {trainable_retrieval_params}/{total_policy_params} params"
    )
    state_dim = GNN_OUT_CHANNELS
    policy_agent = ActorCritic(state_dim=state_dim, action_dim=action_dim, gnn_encoder=policy_gnn_encoder)
    checkpoint_candidate = None
    if args.resume_from:
        checkpoint_candidate = Path(args.resume_from).expanduser()
    else:
        saved_path = Path(AGENT_MODEL_PATH)
        if saved_path.exists():
            checkpoint_candidate = saved_path
    if checkpoint_candidate and checkpoint_candidate.exists():
        try:
            state_dict = torch.load(checkpoint_candidate, map_location=torch.device('cpu'))
            policy_agent.load_state_dict(state_dict, strict=True)
            print(f"♻️  Warm-started policy weights from {checkpoint_candidate}")
        except Exception as exc:
            print(f"⚠️ Failed to load checkpoint {checkpoint_candidate} ({exc}); continuing with fresh initialization.")
    ppo_cfg = PPOConfig(gamma=GAMMA, epochs=EPOCHS, eps_clip=EPS_CLIP, reward_mode=args.reward_mode)
    ppo_updater = PPOTrainer(policy_agent, Adam(policy_agent.parameters(), lr=LEARNING_RATE), ppo_cfg)
    replay_buffer = ReplayBuffer()  # now stores raw PyG Data graphs (not detached embeddings)
    
    print("🧠 Initializing Knowledge Base and Teacher...")
    if args.disable_rag:
        kb = None
        teacher = None
        rag_gnn_for_scheduler = None
        print("🚫 RAG potential shaping disabled via CLI; training will rely on vanilla rewards only.")
        if args.trace_log_dir:
            print("⚠️ Trace logging directory provided but RAG is disabled; no trace logs will be generated.")
    else:
        kb = KnowledgeBase(dimension=GNN_OUT_CHANNELS)
        teacher = KnowledgeableTeacher(
            state_dim=state_dim,
            knowledge_base=kb,
            gnn_encoder=rag_gnn_encoder,
            reward_config=teacher_config,
        )
        if args.trace_log_dir:
            trace_dir = Path(args.trace_log_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            run_slug = args.run_label or "rag"
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            trace_log_path = trace_dir / f"{run_slug}_trace_{timestamp}.jsonl"
            teacher.enable_trace_logging(trace_log_path)
        rag_gnn_for_scheduler = rag_gnn_encoder

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
    reward_normalizer = FamilyRewardNormalizer(
        metadata_path="data/knowledge_base/workflow_metadata.csv"
    )
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

    if args.family_filter:
        allowed_families = {fam.lower() for fam in args.family_filter}
        filtered_workflows: list[str] = []
        for wf in training_workflows:
            family_name = reward_normalizer.get_family(wf)
            if family_name.lower() in allowed_families:
                filtered_workflows.append(wf)
        removed = len(training_workflows) - len(filtered_workflows)
        training_workflows = filtered_workflows
        print(
            f"[Filter] Retained {len(training_workflows)} workflows after family filter {sorted(allowed_families)} (removed {removed})."
        )
        if not training_workflows:
            print("❌ Family filter removed all workflows; aborting.")
            return

    band_assignments = reward_normalizer.assign_bands(training_workflows)
    band_to_workflows: defaultdict[str, list[str]] = defaultdict(list)
    for wf in training_workflows:
        band = band_assignments.get(wf, "medium")
        band_to_workflows[band].append(wf)
    if args.profile:
        for band_name in ("short", "medium", "long"):
            print(f"[Curriculum] band={band_name} size={len(band_to_workflows.get(band_name, []))}")
    band_cycle = deque(["short", "medium", "long"])

    effective_total_episodes = args.max_episodes if args.max_episodes is not None else TOTAL_EPISODES
    print(f"\n[Step 3/4] Starting main training loop... total_episodes={effective_total_episodes}")
    strategy_label = "WASS-DRL (Vanilla)" if args.disable_rag else "WASS-RAG (Full)"
    training_logger: TrainingLogger | None = None
    metadata = {
        "reward_mode": args.reward_mode,
        "disable_rag": args.disable_rag,
        "include_aug": args.include_aug,
        "max_tasks": args.max_tasks,
        "freeze_gnn": args.freeze_gnn,
        "rag_multiplier": rag_multiplier,
        "final_normalizer": final_normalizer,
        "teacher_lambda": teacher_config.get("lambda"),
        "teacher_top_k": teacher_config.get("top_k"),
        "teacher_temperature": teacher_config.get("temperature"),
        "teacher_gamma": teacher_config.get("gamma"),
    }
    if args.family_filter:
        metadata["family_filter"] = list(args.family_filter)
    if args.trace_log_dir:
        metadata["trace_log_dir"] = args.trace_log_dir
    if trace_log_path is not None:
        metadata["trace_log_path"] = str(trace_log_path)
    try:
        training_logger = TrainingLogger(
            strategy_label=strategy_label,
            output_dir=args.log_dir,
            seed=args.seed,
            run_label=args.run_label,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Failed to initialize training logger: {exc}")
        training_logger = None

    loop_start = time.time()
    for episode in range(1, effective_total_episodes + 1):
        episode_start = time.time()
        selected_workflow: str | None = None
        for _ in range(len(band_cycle)):
            candidate_band = band_cycle[0]
            candidates = band_to_workflows.get(candidate_band, [])
            if candidates:
                selected_workflow = str(np.random.choice(candidates))
                band_cycle.rotate(-1)
                break
            band_cycle.rotate(-1)
        if selected_workflow is None:
            selected_workflow = str(np.random.choice(training_workflows))
        workflow_file = selected_workflow
        workflow_band = band_assignments.get(workflow_file, "medium")
        family_name = reward_normalizer.get_family(workflow_file)
        task_count_selected = count_tasks_in_workflow(workflow_file)
        if args.profile:
            print(
                f"[EP] {episode}/{effective_total_episodes} -> {Path(workflow_file).name} "
                f"tasks={task_count_selected} band={workflow_band}"
            )
        if teacher is not None:
            teacher.set_trace_context(
                episode=episode,
                workflow_file=Path(workflow_file).name,
                workflow_band=workflow_band,
                run_label=args.run_label,
                seed=args.seed,
                workflow_family=family_name,
            )
        
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
            rag_gnn_encoder=rag_gnn_for_scheduler,
            workflow_file=workflow_file,
            feature_scaler=feature_scaler,
            randomize_host_order=args.randomize_hosts,
        )
        
        t_sim_start = time.time()
        makespan, sim_details = wrench_runner.run_single_seeding_simulation(
            scheduler_class=trainable_scheduler_factory,
            workflow_file=workflow_file
        )
        sim_elapsed = time.time() - t_sim_start

        potential_summary = (sim_details or {}).get('potential_summary') if sim_details else None

        episode_duration = time.time() - episode_start

        if makespan < 0 or not replay_buffer.rewards:
            print(f"  Episode {episode}, Workflow: {Path(workflow_file).name}, Status: FAILED. Skipping update.")
            if training_logger:
                training_logger.log_episode(
                    episode=episode,
                    workflow=workflow_file,
                    metrics={
                        "status": "failed",
                        "task_count": task_count_selected,
                        "makespan": makespan,
                        "episode_wallclock": episode_duration,
                        "reward_mode": args.reward_mode,
                        "rag_enabled": 0 if args.disable_rag else 1,
                    },
                )
            replay_buffer.clear()
            continue

        raw_rag_rewards = [float(r.item()) for r in replay_buffer.rewards]
        scaled_rag_rewards = reward_normalizer.normalize_dense_rewards(
            workflow_file,
            raw_rag_rewards,
            multiplier=rag_multiplier,
        )
        dense_stats = reward_normalizer.summarize_rewards(workflow_file, scaled_rag_rewards)
        scaled_array = np.asarray(scaled_rag_rewards, dtype=np.float64)
        rag_min = float(scaled_array.min()) if scaled_array.size else 0.0
        rag_max = float(scaled_array.max()) if scaled_array.size else 0.0
        final_reward_component = reward_normalizer.compute_final_reward(workflow_file, makespan)
        if final_normalizer and final_normalizer > 0:
            final_reward_component *= (1000.0 / final_normalizer)

        if args.reward_mode == 'dense':
            steps = len(scaled_rag_rewards)
            if steps == 0:
                replay_buffer.rewards = [torch.tensor(final_reward_component, dtype=torch.float32)]
            else:
                per_step_bonus = final_reward_component / steps
                replay_buffer.rewards = [
                    torch.tensor(val + per_step_bonus, dtype=torch.float32)
                    for val in scaled_rag_rewards
                ]
        else:
            total_dense = float(np.sum(scaled_rag_rewards)) if scaled_rag_rewards else 0.0
            combined = total_dense + final_reward_component
            replay_buffer.rewards = [torch.tensor(combined, dtype=torch.float32)]

        normalized_rewards = [float(r.item()) for r in replay_buffer.rewards]
        reward_stats = reward_normalizer.summarize_rewards(workflow_file, normalized_rewards)
        clip_low, clip_high = reward_normalizer.get_ratio_clip(workflow_file)
        print(
            "    [RewardNorm] family=%s band=%s mean=%.3f std=%.3f pos=%.1f%% neg=%.1f%% clip=(%.2f,%.2f) final=%.3f"
            % (
                family_name,
                workflow_band,
                reward_stats["mean"],
                reward_stats["std"],
                reward_stats["positive_frac"] * 100.0,
                reward_stats["negative_frac"] * 100.0,
                clip_low,
                clip_high,
                final_reward_component,
            )
        )

        reported_reward = float(sum(normalized_rewards)) if normalized_rewards else 0.0
        
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

        if potential_summary:
            phi_min = potential_summary.get('phi_min')
            phi_max = potential_summary.get('phi_max')
            delta_min = potential_summary.get('delta_min')
            delta_max = potential_summary.get('delta_max')
            samples = potential_summary.get('samples', [])
            sample_preview = []
            for sample in samples[:2]:
                task_name = sample.get('task_name')
                delta_val = sample.get('delta')
                reward_val = sample.get('reward')
                sample_preview.append(f"{task_name}:Δφ={delta_val:.4f},r={reward_val:.4f}")
            sample_str = '; '.join(sample_preview) if sample_preview else 'n/a'
            print(
                "    [PBRS] φ_range=[{:.4f}, {:.4f}] Δφ_range=[{:.4f}, {:.4f}] samples={}".format(
                    phi_min if phi_min is not None else float('nan'),
                    phi_max if phi_max is not None else float('nan'),
                    delta_min if delta_min is not None else float('nan'),
                    delta_max if delta_max is not None else float('nan'),
                    sample_str,
                )
            )

        # Optional periodic KB refresh (expensive: re-encode all workflows with current GNN)
        if args.kb_refresh_interval and episode % args.kb_refresh_interval == 0:
            if kb is None:
                print("[KB Refresh] Skipped because RAG is disabled for this run.")
            elif args.freeze_gnn:
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
            f"ScaledRAG mean={dense_stats['mean']:.4f} min={rag_min:.4f} max={rag_max:.4f} std={dense_stats['std']:.4f} "
            f"pos={dense_stats['positive_frac']*100:.1f}% neg={dense_stats['negative_frac']*100:.1f}% "
            f"reported={reported_reward:.4f}, sim_time={sim_elapsed:.2f}s"
        )

        if training_logger:
            training_logger.log_episode(
                episode=episode,
                workflow=workflow_file,
                metrics={
                    "status": "success",
                    "task_count": task_count_selected,
                    "makespan": makespan,
                    "episode_reward": reported_reward,
                    "rag_mean": dense_stats["mean"],
                    "rag_std": dense_stats["std"],
                    "rag_positive_frac": dense_stats["positive_frac"],
                    "rag_negative_frac": dense_stats["negative_frac"],
                    "rag_min": rag_min,
                    "rag_max": rag_max,
                    "sim_time": sim_elapsed,
                    "episode_wallclock": episode_duration,
                    "reward_mode": args.reward_mode,
                    "rag_enabled": 0 if args.disable_rag else 1,
                    "workflow_band": workflow_band,
                    "workflow_family": family_name,
                    "final_reward_component": final_reward_component,
                    "reward_mean": reward_stats["mean"],
                    "reward_std": reward_stats["std"],
                    "reward_positive_frac": reward_stats["positive_frac"],
                    "reward_negative_frac": reward_stats["negative_frac"],
                    "replay_steps": len(scaled_rag_rewards),
                },
            )

        if SAVE_INTERVAL and episode % SAVE_INTERVAL == 0:
            torch.save(policy_agent.state_dict(), AGENT_MODEL_PATH)
            print(f"💾 Model saved at episode {episode}")

    total_loop_elapsed = time.time() - loop_start
    print(f"\n[Step 4/4] Training finished. Total loop time: {total_loop_elapsed:.2f}s")
    torch.save(policy_agent.state_dict(), AGENT_MODEL_PATH)
    print(f"✅ Final model saved to: {AGENT_MODEL_PATH}")
    print("\n🎉 [Phase 3] DRL Agent Training Completed! 🎉")

    if training_logger:
        training_logger.close()

if __name__ == "__main__":
    main()