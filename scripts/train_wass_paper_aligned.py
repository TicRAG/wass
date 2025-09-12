#!/usr/bin/env python3
"""
Paper-aligned WASS training script (Prototype)
Implements:
  - Graph workflow encoder (lightweight GNN-style message passing) -> graph embedding
  - PPO (policy gradient) instead of DQN
  - RAG Teacher providing R_RAG reward shaping (retrieval-based action quality signal)
Reward = w_rag * R_RAG + (1 - w_rag) * R_env
Where R_env is a simplified dense reward (data locality, critical path proxy, load balance)

NOTE: This is an incremental prototype to reduce gap with the paper.
      It intentionally keeps the existing WRENCH environment logic minimal.
"""
import os
import sys
import time
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---- Project Path ----
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# ---- Optional WRENCH Import ----
try:
    import wrench  # type: ignore
except Exception:
    print("⚠️  WRENCH not available. Install wrench-python-api before running actual training.")
    wrench = None  # type: ignore

# ---- Internal Modules (added) ----
from src.graph_encoder import GraphFeatureEncoder
from src.ppo_agent import PPOAgent
from src.rag_teacher import RAGTeacher
from src.workflow_generator_shared import generate_workflow, DEFAULT_FLOPS_RANGE, DEFAULT_FILE_SIZE_RANGE, DEFAULT_DEP_PROB

# Reuse / fallback: lightweight performance predictor optional
try:
    from src.performance_predictor import PerformancePredictor
except Exception:
    PerformancePredictor = None  # type: ignore

# ---------------- Environment Wrapper -----------------
class PaperWRENCHEnv:
    """Simplified WRENCH RL environment for PPO + GNN encoder.
    Generates same task/node style as original DQN environment.
    """
    def __init__(self, platform_file: str, controller_host: str = "ControllerHost"):
        self.platform_file = platform_file
        self.controller_host = controller_host
        self.sim = None
        self.workflow = None
        self.compute_service = None
        self.storage_service = None
        self.compute_nodes = ["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"]
        self.node_capacities = {
            "ComputeHost1": 2.0,
            "ComputeHost2": 3.0,
            "ComputeHost3": 2.5,
            "ComputeHost4": 4.0,
        }
        # Basic (non-graph) state dim reused for compatibility if needed
        self.state_dim_flat = 5 + len(self.compute_nodes) * 3
        self.task_list = []
        self.files = []
        self.current_time = 0.0
        self.task_completion_times: Dict[str, float] = {}
        self.scheduled: set[str] = set()

        with open(platform_file, 'r', encoding='utf-8') as f:
            self.platform_xml = f.read()

    def reset(self, num_tasks: int = 12):
        if wrench is None:
            raise RuntimeError("WRENCH not available.")
        if self.sim:
            try:
                self.sim.terminate()
            except Exception:
                pass
        self.sim = wrench.Simulation()
        self.sim.start(self.platform_xml, self.controller_host)
        self.storage_service = self.sim.create_simple_storage_service("StorageHost", ["/storage"])
        compute_resources = {n: (4, 8_589_934_592) for n in self.compute_nodes}
        self.compute_service = self.sim.create_bare_metal_compute_service(
            "ComputeHost1", compute_resources, "/scratch", {}, {}
        )
        # Unified workflow generation (shared with evaluation)
        self.workflow, self.task_list, self.files = generate_workflow(
            self.sim,
            size=num_tasks,
            flops_range=DEFAULT_FLOPS_RANGE,
            dep_prob=DEFAULT_DEP_PROB,
            file_size_range=DEFAULT_FILE_SIZE_RANGE
        )
        for f in self.files:
            self.storage_service.create_file_copy(f)
        self.current_time = 0.0
        self.task_completion_times.clear()
        self.scheduled.clear()
        return self._basic_state()

    def _basic_state(self) -> np.ndarray:
        ready = self.workflow.get_ready_tasks()
        if not ready:
            return np.zeros(self.state_dim_flat, dtype=np.float32)
        task = ready[0]
        task_features = [
            task.get_flops() / 1e9,
            len(task.get_input_files()),
            task.get_number_of_children(),
            len(self.task_list),
            len(self.scheduled) / max(1, len(self.task_list)),
        ]
        node_feats = []
        for n in self.compute_nodes:
            cap = self.node_capacities[n]
            avail = 0.0  # placeholder (no queue modeling)
            exec_time = task.get_flops() / (cap * 1e9)
            node_feats.extend([cap / 4.0, avail / 100.0, exec_time / 10.0])
        return np.array(task_features + node_feats, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        ready = self.workflow.get_ready_tasks()
        if not ready:
            return self._basic_state(), {"done": self.workflow.is_done(), "skipped": True}
        task = ready[0]
        node = self.compute_nodes[action % len(self.compute_nodes)]
        file_locations = {f: self.storage_service for f in task.get_input_files()}
        for f in task.get_output_files():
            file_locations[f] = self.storage_service
        job = self.sim.create_standard_job([task], file_locations)
        self.compute_service.submit_standard_job(job)
        while True:
            event = self.sim.wait_for_next_event()
            if event["event_type"] == "standard_job_completion" and event["standard_job"] == job:
                break
            if event["event_type"] == "simulation_termination":
                break
        comp_time = self.sim.get_simulated_time()
        self.current_time = comp_time
        self.scheduled.add(task.get_name())
        self.task_completion_times[task.get_name()] = comp_time
        return self._basic_state(), {
            "done": self.workflow.is_done(),
            "task": task,
            "node": node,
            "completion_time": comp_time,
        }

    def makespan(self) -> float:
        return max(self.task_completion_times.values()) if self.task_completion_times else 0.0

    def graph_encoding(self, encoder: GraphFeatureEncoder) -> torch.Tensor:
        g = encoder.build_graph(self.workflow)
        return encoder(g['node_feat'], g['adj'])  # [2H]

    def cleanup(self):
        if self.sim:
            try:
                self.sim.terminate()
            except Exception:
                pass
            self.sim = None

# ---------------- Reward Components -----------------
class RewardComposer:
    def __init__(self, rag_weight: float = 0.7):
        self.rag_weight = rag_weight

    def dense_env_reward(self, task, info: Dict[str, Any]) -> float:
        # Proxy heuristics (scaled small to let R_RAG dominate)
        in_deg = len(task.get_input_files())
        out_deg = task.get_number_of_children()
        locality = 0.6  # placeholder (could inspect file locations)
        critical_bonus = min(out_deg / 4.0, 1.0)
        locality_bonus = locality
        dependency_penalty = -0.1 * (in_deg > 2)
        return 0.5 * critical_bonus + 0.3 * locality_bonus + dependency_penalty

    def combine(self, r_env: float, r_rag: float) -> float:
        return self.rag_weight * r_rag + (1 - self.rag_weight) * r_env

# ---------------- Training Loop -----------------
def train(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        base_cfg = json.load(f) if config_path.endswith('.json') else __import__('yaml').safe_load(f)
    drl_cfg = base_cfg.get('drl', {})
    seed = base_cfg.get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ppo_cfg = {
        'learning_rate': drl_cfg.get('learning_rate', 3e-4),
        'gamma': drl_cfg.get('gamma', 0.99),
        'gae_lambda': drl_cfg.get('gae_lambda', 0.95),
        'clip_eps': drl_cfg.get('clip_eps', 0.2),
        'entropy_coef': drl_cfg.get('entropy_coef', 0.01),
        'value_coef': drl_cfg.get('value_coef', 0.5),
        'ppo_batch_size': drl_cfg.get('ppo_batch_size', 64),
        'ppo_update_epochs': drl_cfg.get('ppo_update_epochs', 4),
        'hidden_dim': drl_cfg.get('hidden_dim', 256)
    }
    rag_weight = drl_cfg.get('rag_reward_weight', 0.7)
    rag_schedule = drl_cfg.get('rag_weight_schedule', {
        'initial': rag_weight,
        'final': rag_weight,
        'warmup_episodes': 0
    })
    rag_reward_scale = drl_cfg.get('rag_reward_scale', 0.3)
    dynamic_rag_scale = drl_cfg.get('dynamic_rag_scale', True)
    topk_sched = drl_cfg.get('teacher_top_k_schedule', {
        'initial': 5, 'final': 5, 'warmup_episodes': 0
    })
    episodes = drl_cfg.get('episodes', 60)
    horizon = drl_cfg.get('rollout_horizon', 64)
    reward_norm = drl_cfg.get('reward_norm', True)
    lr_decay = drl_cfg.get('lr_decay', True)
    max_grad_norm = drl_cfg.get('max_grad_norm', 1.0)

    platform_file = base_cfg.get('platform', {}).get('platform_file', 'configs/platform.xml')

    env = PaperWRENCHEnv(platform_file)
    encoder = GraphFeatureEncoder(in_dim=5, hidden_dim=64, layers=2)
    graph_dim = encoder.out_dim  # 128

    # Combined state: flat basic state + graph embedding
    combined_state_dim = env.state_dim_flat + graph_dim
    agent = PPOAgent(combined_state_dim, len(env.compute_nodes), ppo_cfg)
    teacher = RAGTeacher(top_k=topk_sched.get('initial',5))
    composer = RewardComposer(rag_weight=rag_weight)

    metrics = []

    running_return_mean = 0.0
    beta = 0.02  # moving average factor for logging (not used in update)
    for ep in range(episodes):
        # Optional linear LR decay
        if lr_decay:
            frac = 1 - ep / max(1, episodes)
            for g in agent.optimizer.param_groups:
                g['lr'] = agent.lr * frac
        # Dynamic rag weight schedule
        if rag_schedule['warmup_episodes'] > 0:
            progress = min(1.0, ep / max(1, rag_schedule['warmup_episodes']))
            composer.rag_weight = rag_schedule['initial'] + progress * (rag_schedule['final'] - rag_schedule['initial'])
        # teacher top-k schedule
        if topk_sched.get('warmup_episodes',0) > 0:
            tk_prog = min(1.0, ep / max(1, topk_sched['warmup_episodes']))
            cur_topk = int(round(topk_sched['initial'] + tk_prog * (topk_sched['final'] - topk_sched['initial'])))
            teacher.top_k = max(1, cur_topk)
        env.reset(num_tasks=random.randint(10, 20))
        traj_states: List[torch.Tensor] = []
        traj_actions: List[torch.Tensor] = []
        traj_logps: List[torch.Tensor] = []
        traj_rewards: List[torch.Tensor] = []
        traj_dones: List[torch.Tensor] = []
        traj_values: List[torch.Tensor] = []

        steps = 0
        ep_reward = 0.0
        ep_r_env = 0.0
        ep_r_rag = 0.0
        while steps < horizon:
            flat_state = env._basic_state()  # numpy
            graph_emb = env.graph_encoding(encoder).detach().cpu().numpy()
            state_vec = np.concatenate([flat_state, graph_emb], axis=0)
            state_t = torch.from_numpy(state_vec).float()

            action, logp, value = agent.act(state_t.unsqueeze(0))
            next_state_flat, info = env.step(action)

            # R_RAG computation (needs current graph emb & action quality)
            task = info.get('task')
            if task is None:
                break
            task_flops = task.get_flops()
            node_caps = [env.node_capacities[n] for n in env.compute_nodes]
            scale_now = rag_reward_scale
            if dynamic_rag_scale:
                # grow scale with memory richness (sqrt for diminishing returns)
                scale_now = min(rag_reward_scale * (1 + 0.5 * (len(teacher.cases)**0.5 / 100.0)), rag_reward_scale*1.8)
            r_rag = teacher.rag_reward(graph_emb, task_flops, node_caps, action, scale=scale_now)
            r_env = composer.dense_env_reward(task, info)
            reward = composer.combine(r_env, r_rag)

            # Store teacher case (uses observed exec time as completion_time - current_time delta ≈ exec)
            teacher.add_case(graph_emb, task_flops, action, info['completion_time'], node_caps[action])

            traj_states.append(state_t)
            traj_actions.append(torch.tensor(action))
            traj_logps.append(logp)
            traj_values.append(value)
            traj_rewards.append(torch.tensor(reward, dtype=torch.float32))
            traj_dones.append(torch.tensor(float(info['done'])))

            ep_reward += reward
            ep_r_env += r_env
            ep_r_rag += r_rag
            steps += 1
            if info['done']:
                break

        # PPO update
        from src.ppo_agent import PPOTrajectory  # local import to avoid circular
        # Optional reward normalization (center & scale by std per trajectory)
        if reward_norm and traj_rewards:
            rew_tensor = torch.stack(traj_rewards)
            std = rew_tensor.std()
            if std > 1e-6:
                mean = rew_tensor.mean()
                normed = (rew_tensor - mean) / (std + 1e-8)
                traj_rewards = [r for r in normed]

        # Attach grad clip into agent temporarily
        agent.max_grad_norm = max_grad_norm

        stats = agent.update(PPOTrajectory(
            states=traj_states,
            actions=traj_actions,
            log_probs=traj_logps,
            rewards=traj_rewards,
            dones=traj_dones,
            values=traj_values
        ))

        ms = env.makespan()
        # Update moving average total reward
        running_return_mean = (1 - beta) * running_return_mean + beta * (ep_reward / max(1, steps))
        metrics.append({
            'episode': ep,
            'reward': ep_reward / max(1, steps),
            'avg_r_env': ep_r_env / max(1, steps),
            'avg_r_rag': ep_r_rag / max(1, steps),
            'rag_weight': composer.rag_weight,
            'steps': steps,
            'makespan': ms,
            'policy_loss': stats['policy_loss'],
            'value_loss': stats['value_loss'],
            'entropy': stats['entropy'],
            'teacher_cases': len(teacher.cases)
        })
        if ep % 10 == 0:
            print(
                f"EP {ep:03d} | avgR={metrics[-1]['reward']:.3f} (env={metrics[-1]['avg_r_env']:.3f} rag={metrics[-1]['avg_r_rag']:.3f}) "
                f"mksp={ms:.2f} cases={len(teacher.cases)}"
            )
        env.cleanup()

    # ---- Save Model + Metrics ----
    out_dir = Path('models')
    out_dir.mkdir(exist_ok=True)
    ckpt = {
        'actor_critic': agent.model.state_dict(),
        'config': {'ppo': ppo_cfg, 'rag_weight': rag_weight},
        'paper_alignment': {
            'gnn_encoder': True,
            'ppo': True,
            'rag_teacher_reward': True,
            'combined_reward_formula': 'w_rag * R_RAG + (1-w_rag) * R_env'
        }
    }
    torch.save(ckpt, out_dir / 'wass_paper_aligned.pth')
    with open(out_dir / 'wass_paper_aligned_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Saved paper-aligned PPO model -> models/wass_paper_aligned.pth")
    print("📊 Metrics -> models/wass_paper_aligned_metrics.json")

    return metrics

# ---------------- Entry -----------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_wass_paper_aligned.py <config.yaml>")
        sys.exit(1)
    train(sys.argv[1])
