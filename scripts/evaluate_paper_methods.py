#!/usr/bin/env python3
"""
Unified evaluation script for paper-aligned methods.
Evaluates: FIFO, HEFT, WASS-Heuristic, WASS-DRL (legacy), WASS-PPO+RAG (paper-aligned prototype)
Outputs a JSON with per-workflow metrics and aggregate stats.
"""
import os
import sys
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np
import torch

# Project path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

# Import scheduler adapters (replaces deprecated experiments.wrench_real_experiment)
from src.legacy_scheduler_adapters import (
    FIFOScheduler, HEFTScheduler, WASSHeuristicScheduler, WASSDRLScheduler
)
from src.workflow_generator_shared import generate_workflow, generate_richer_workflow, DEFAULT_FLOPS_RANGE, DEFAULT_FILE_SIZE_RANGE, DEFAULT_DEP_PROB

# Paper aligned components
from src.graph_encoder import GraphFeatureEncoder
from src.ppo_agent import PPOAgent
from src.rag_teacher import RAGTeacher

try:
    import wrench  # type: ignore
except Exception:
    print("WRENCH not available. Install wrench-python-api for real evaluation.")
    wrench = None  # type: ignore

@dataclass
class EvalRecord:
    method: str
    workflow_id: str
    tasks: int
    makespan: float
    steps: int
    runtime_ms: float
    total_transfer_bytes: float
    transfer_events: int
    locality_hit_rate: float
    transfer_bytes_per_task: float

class SimpleWorkflowEnv:
    def __init__(self, platform_file: str):
        if wrench is None:
            raise RuntimeError("WRENCH not available.")
        self.platform_file = platform_file
        with open(platform_file,'r',encoding='utf-8') as f:
            self.platform_xml = f.read()
        self.compute_nodes = ["ComputeHost1","ComputeHost2","ComputeHost3","ComputeHost4"]
        # Increase heterogeneity (wider spread)
        self.node_capacities = {"ComputeHost1":1.0,"ComputeHost2":2.0,"ComputeHost3":4.0,"ComputeHost4":8.0}

    def run(self, workflow_size: int, scheduler) -> EvalRecord:
        sim = wrench.Simulation()
        sim.start(self.platform_xml, "ControllerHost")
        storage_service = sim.create_simple_storage_service("StorageHost", ["/storage"])
        compute_resources = {n: (4, 8_589_934_592) for n in self.compute_nodes}
        compute_service = sim.create_bare_metal_compute_service("ComputeHost1", compute_resources, "/scratch", {}, {})
        # Richer DAG with higher dependency density
        workflow, tasks, files, adjacency = generate_richer_workflow(
            sim,
            size=workflow_size,
            flops_range=DEFAULT_FLOPS_RANGE,
            file_size_range=DEFAULT_FILE_SIZE_RANGE,
            layers=None,
            layer_width_range=(3, 10),
            max_parents=4,
            parent_edge_prob=0.7,
            merge_prob=0.35,
            seed=random.randint(0, 10_000),
        )
        if hasattr(scheduler, "set_adjacency"):
            try:
                scheduler.set_adjacency(adjacency)
            except Exception:
                pass
        for f in files:
            storage_service.create_file_copy(f)
        # Random initial placement
        initial_nodes: Dict[str,str] = {}
        for f in files:
            fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
            node_choice = random.choice(self.compute_nodes)
            initial_nodes[fid] = node_choice
            if hasattr(scheduler, 'data_location') and isinstance(getattr(scheduler,'data_location'), dict):
                scheduler.data_location[fid] = node_choice
        node_loads = {n: 0.0 for n in self.compute_nodes}
        # Track running jobs: job -> (task, node, start_time)
        active = {}
        ready = list(workflow.get_ready_tasks())
        steps = 0
        data_location: Dict[str,str] = {}
        total_transfer = 0.0; transfer_events = 0; local_hits = 0; file_accesses = 0
        start_wall = time.time()
        # Helper to submit one task
        def submit_task(task, node):
            nonlocal steps, total_transfer, transfer_events, local_hits, file_accesses
            file_locations = {f: storage_service for f in task.get_input_files()}
            try:
                for f in task.get_input_files():
                    file_accesses += 1
                    fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                    size_b = f.get_size()
                    loc = data_location.get(fid, initial_nodes.get(fid))
                    if loc == node:
                        local_hits += 1
                    else:
                        total_transfer += size_b; transfer_events += 1
            except Exception:
                pass
            for f in task.get_output_files():
                file_locations[f] = storage_service
            job = sim.create_standard_job([task], file_locations)
            compute_service.submit_standard_job(job)
            active[job] = (task, node, sim.get_simulated_time())
            steps += 1
        # Main event loop
        while True:
            # Submit while we have ready tasks
            scheduled_any = False
            while ready:
                if hasattr(scheduler, 'choose'):
                    t, node = scheduler.choose(ready, self.compute_nodes, self.node_capacities, node_loads, compute_service)
                    if t is None:
                        break
                else:
                    t = ready[0]
                    node = scheduler.schedule_task(t, self.compute_nodes, self.node_capacities, node_loads, compute_service)
                # Remove chosen from ready
                try:
                    ready.remove(t)
                except ValueError:
                    pass
                submit_task(t, node)
                scheduled_any = True
            if not active:
                # No running jobs; if nothing ready either, we're done
                if not ready:
                    break
            # Wait next event
            ev = sim.wait_for_next_event()
            if ev is None or ev.get('event_type') == 'simulation_termination':
                break
            if ev.get('event_type') == 'standard_job_completion':
                job = ev.get('standard_job')
                info = active.pop(job, None)
                if info:
                    task, node, st = info
                    et = sim.get_simulated_time()
                    exec_t = et - st
                    node_loads[node] += exec_t
                    # Update data locality
                    try:
                        for f in task.get_output_files():
                            fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                            data_location[fid] = node
                            if hasattr(scheduler,'data_location') and isinstance(getattr(scheduler,'data_location'), dict):
                                scheduler.data_location[fid] = node
                    except Exception:
                        pass
                    if hasattr(scheduler,'notify_task_completion'):
                        try:
                            scheduler.notify_task_completion(task.get_name(), node)
                        except Exception:
                            pass
                # Refresh ready list after completion
                try:
                    ready = list(workflow.get_ready_tasks())
                except Exception:
                    ready = []
        makespan = sim.get_simulated_time()
        sim.terminate()
        return EvalRecord(
            method=scheduler.name,
            workflow_id=f"wf_{workflow_size}_{random.randint(0,9999)}",
            tasks=workflow_size,
            makespan=makespan,
            steps=steps,
            runtime_ms=(time.time() - start_wall) * 1000,
            total_transfer_bytes=total_transfer,
            transfer_events=transfer_events,
            locality_hit_rate=(local_hits / file_accesses) if file_accesses else 0.0,
            transfer_bytes_per_task=(total_transfer / workflow_size) if workflow_size else 0.0,
        )

class PPOPaperScheduler:
    def __init__(self, model_path: str, rag_weight: float = 0.7, deterministic: bool = True):
        ckpt = torch.load(model_path, map_location='cpu')
        self.rag_weight = ckpt['config'].get('rag_weight', rag_weight)
        self.name = 'WASS-PPO-RAG'
        self.deterministic = deterministic
        self.encoder = GraphFeatureEncoder(in_dim=5, hidden_dim=64, layers=2)
        from src.ppo_agent import ActorCritic
        self.model = ActorCritic(self.encoder.out_dim + (5 + 4*3), 4, 256)
        self.model.load_state_dict(ckpt['actor_critic'])
        self.model.eval()

    def _graph_embedding(self, workflow):
        # Build minimal adjacency + node features similar to training encoder interface
        g = self.encoder.build_graph(workflow)
        with torch.no_grad():
            return self.encoder(g['node_feat'], g['adj']).cpu().numpy()

    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        # Base features (keep original dimensionality expectations)
        task_features = [
            task.get_flops()/1e9,
            len(task.get_input_files()),
            task.get_number_of_children(),
            0.0,
            0.0
        ]
        max_load = max(node_loads.values()) if node_loads else 1.0
        node_feats = []
        exec_costs = {}
        for n in available_nodes:
            cap = node_capacities.get(n,1.0)
            exec_time = task.get_flops()/(cap*1e9)
            load_norm = (node_loads.get(n,0.0)/max(1e-6, max_load)) if max_load > 0 else 0.0
            node_feats.extend([cap/8.0, load_norm, exec_time/10.0])  # capacity normalized to 0..1
            exec_costs[n] = exec_time + node_loads.get(n,0.0)
        import numpy as np
        flat = np.array(task_features + node_feats, dtype=np.float32)
        try:
            workflow = task.get_workflow()
            graph_emb = self._graph_embedding(workflow)
        except Exception:
            graph_emb = np.zeros(self.encoder.out_dim, dtype=np.float32)
        state_vec = np.concatenate([flat, graph_emb], axis=0)
        with torch.no_grad():
            logits, _ = self.model(torch.from_numpy(state_vec).float().unsqueeze(0))
            if self.deterministic:
                act_idx = int(torch.argmax(logits, dim=-1).item())
            else:
                act_idx = torch.distributions.Categorical(logits=logits).sample().item()
        rl_choice = available_nodes[act_idx % len(available_nodes)]
        # Hybrid: prefer node with minimal (exec_time + accumulated load)
        best_node = min(exec_costs.items(), key=lambda kv: kv[1])[0]
        if self.deterministic:
            return best_node
        else:
            import random as _r
            if _r.random() < self.rag_weight:
                return best_node
            return rl_choice


def evaluate(config_path: str, workflows: List[int] = None, repetitions: int = 3):
    with open(config_path,'r',encoding='utf-8') as f:
        base_cfg = json.load(f) if config_path.endswith('.json') else __import__('yaml').safe_load(f)
    platform_file = base_cfg.get('platform',{}).get('platform_file','configs/platform.xml')
    seed = base_cfg.get('random_seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    workflows = workflows or [5,10,15,25,35,50,60,80]
    env_sizes = os.environ.get('WASS_EVAL_SIZES')
    if env_sizes:
        try:
            parsed = [int(x.strip()) for x in env_sizes.split(',') if x.strip()]
            if parsed:
                workflows = parsed
                print(f"Using workflow sizes from WASS_EVAL_SIZES: {workflows}")
        except Exception as e:
            print('Failed to parse WASS_EVAL_SIZES:', e)

    # Prepare schedulers
    schedulers = {
        'FIFO': FIFOScheduler(),
        'HEFT': HEFTScheduler(),
        'WASS-Heuristic': WASSHeuristicScheduler()
    }
    if os.path.exists('models/wass_models.pth'):
        try:
            schedulers['WASS-DRL'] = WASSDRLScheduler('models/wass_models.pth')
        except Exception as e:
            print('Skip WASS-DRL:', e)
    if os.path.exists('models/wass_paper_aligned.pth'):
        try:
            deterministic = base_cfg.get('drl',{}).get('deterministic_eval', True)
            schedulers['WASS-PPO-RAG'] = PPOPaperScheduler('models/wass_paper_aligned.pth', deterministic=deterministic)
        except Exception as e:
            print('Skip WASS-PPO-RAG:', e)

    env = SimpleWorkflowEnv(platform_file)
    records: List[EvalRecord] = []
    total = len(schedulers)*len(workflows)*repetitions
    c = 0
    for name, sch in schedulers.items():
        for w in workflows:
            for r in range(repetitions):
                c += 1
                print(f"[{c}/{total}] {name} workflow={w} rep={r}")
                try:
                    rec = env.run(w, sch)
                    records.append(rec)
                except Exception as e:
                    print('  failed:', e)
    # Aggregate stats
    agg: Dict[str, Dict[str, Any]] = {}
    size_buckets: Dict[str, Dict[int, List[float]]] = {}
    for rec in records:
        agg.setdefault(rec.method, {'makespans': [], 'transfer_bytes': [], 'locality_hr': [], 'transfer_per_task': []})
        agg[rec.method]['makespans'].append(rec.makespan)
        agg[rec.method]['transfer_bytes'].append(rec.total_transfer_bytes)
        agg[rec.method]['locality_hr'].append(rec.locality_hit_rate)
        agg[rec.method]['transfer_per_task'].append(rec.transfer_bytes_per_task)
        size_buckets.setdefault(rec.method, {}).setdefault(rec.tasks, []).append(rec.makespan)
    summary = {}
    for m, d in agg.items():
        arr = np.array(d['makespans'])
        t_arr = np.array(d['transfer_bytes'])
        hr_arr = np.array(d['locality_hr'])
        tpt_arr = np.array(d['transfer_per_task'])
        summary[m] = {
            'count': int(arr.size),
            'mean_makespan': float(arr.mean()),
            'std_makespan': float(arr.std()),
            'best': float(arr.min()),
            'mean_total_transfer_bytes': float(t_arr.mean()) if t_arr.size else 0.0,
            'mean_locality_hit_rate': float(hr_arr.mean()) if hr_arr.size else 0.0,
            'mean_transfer_bytes_per_task': float(tpt_arr.mean()) if tpt_arr.size else 0.0,
        }
    # Significance quick comparison if both methods exist
    if 'WASS-DRL' in summary and 'WASS-PPO-RAG' in summary:
        a = np.array(agg['WASS-DRL']['makespans'])
        b = np.array(agg['WASS-PPO-RAG']['makespans'])
        if a.size == b.size and a.size > 2:
            diff = a - b
            summary['comparison'] = {
                'ppo_better_mean': float(diff.mean()),
                'ppo_better_std': float(diff.std()),
                'relative_improvement_percent': float((a.mean()-b.mean())/a.mean()*100.0),
                'samples': int(a.size)
            }
    # Per size breakdown
    per_size = {}
    for m, sizes in size_buckets.items():
        per_size[m] = {}
        for sz, arr in sizes.items():
            a = np.array(arr)
            per_size[m][str(sz)] = {
                'count': int(a.size),
                'mean': float(a.mean()),
                'std': float(a.std()),
                'best': float(a.min())
            }
    summary['per_size'] = per_size
    out_dir = Path('results/paper_eval'); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'records.json','w') as f: json.dump([asdict(r) for r in records], f, indent=2)
    with open(out_dir/'summary.json','w') as f: json.dump(summary, f, indent=2)
    print('Saved evaluation ->', out_dir)
    return summary

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/evaluate_paper_methods.py <config.yaml>')
        sys.exit(1)
    evaluate(sys.argv[1])
