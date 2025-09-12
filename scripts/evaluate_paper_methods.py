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
from typing import List, Dict, Any, Tuple

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
    def __init__(self, platform_file: str, flops_range: Tuple[float,float]=None, file_size_range: Tuple[int,int]=None, capacity_scale: float=1.0,
                 sensitive_file_fraction: float = 0.0, sensitive_transfer_multiplier: float = 1.0):
        if wrench is None:
            raise RuntimeError("WRENCH not available.")
        self.platform_file = platform_file
        with open(platform_file,'r',encoding='utf-8') as f:
            self.platform_xml = f.read()
        self.compute_nodes = ["ComputeHost1","ComputeHost2","ComputeHost3","ComputeHost4"]
        base_caps = {"ComputeHost1":1.0,"ComputeHost2":2.0,"ComputeHost3":4.0,"ComputeHost4":8.0}
        self.node_capacities = {k: v*capacity_scale for k,v in base_caps.items()}
        self.flops_range = flops_range
        self.file_size_range = file_size_range
        self.sensitive_file_fraction = max(0.0, min(1.0, sensitive_file_fraction))
        self.sensitive_transfer_multiplier = max(1.0, sensitive_transfer_multiplier)
        self.sensitive_files: set[str] = set()

    def run(self, workflow_size: int, scheduler) -> EvalRecord:
        sim = wrench.Simulation()
        sim.start(self.platform_xml, "ControllerHost")
        storage_service = sim.create_simple_storage_service("StorageHost", ["/storage"])
        compute_resources = {n: (4, 8_589_934_592) for n in self.compute_nodes}
        compute_service = sim.create_bare_metal_compute_service("ComputeHost1", compute_resources, "/scratch", {}, {})

        workflow, tasks, files, adjacency = generate_richer_workflow(
            sim,
            size=workflow_size,
            flops_range=self.flops_range or DEFAULT_FLOPS_RANGE,
            file_size_range=self.file_size_range or DEFAULT_FILE_SIZE_RANGE,
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

        # Select sensitive files
        if self.sensitive_file_fraction > 0 and files:
            rng = random.Random(1234)
            shuffled = list(files); rng.shuffle(shuffled)
            take = int(len(shuffled) * self.sensitive_file_fraction)
            self.sensitive_files = {
                (ff.get_name() if hasattr(ff, 'get_name') else getattr(ff, 'name', str(ff))) for ff in shuffled[:take]
            }
        # Create baseline copies on storage
        for f in files:
            storage_service.create_file_copy(f)

        # Random initial placement knowledge for scheduler locality map
        initial_nodes: Dict[str, str] = {}
        for f in files:
            fid = f.get_name() if hasattr(f, 'get_name') else getattr(f, 'name', str(f))
            host_choice = random.choice(self.compute_nodes)
            initial_nodes[fid] = host_choice
            if hasattr(scheduler, 'data_location') and isinstance(getattr(scheduler, 'data_location'), dict):
                scheduler.data_location[fid] = host_choice

        node_loads = {n: 0.0 for n in self.compute_nodes}
        active: Dict[Any, Any] = {}
        ready = list(workflow.get_ready_tasks())
        steps = 0
        data_location: Dict[str, str] = {}
        total_transfer = 0.0; transfer_events = 0; local_hits = 0; file_accesses = 0
        zero_access_tasks = 0
        sensitive_transfer_bytes = 0.0; sensitive_transfer_events = 0
        start_wall = time.time()

        def submit_task(task, node):
            nonlocal steps, total_transfer, transfer_events, local_hits, file_accesses
            nonlocal zero_access_tasks, sensitive_transfer_bytes, sensitive_transfer_events
            file_locations = {f: storage_service for f in task.get_input_files()}
            try:
                ins = list(task.get_input_files())
                if not ins:
                    zero_access_tasks += 1
                for f in ins:
                    file_accesses += 1
                    fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                    size_b = f.get_size()
                    origin = data_location.get(fid, initial_nodes.get(fid))
                    if origin == node:
                        local_hits += 1
                    else:
                        mult = self.sensitive_transfer_multiplier if fid in self.sensitive_files else 1.0
                        added = size_b * mult
                        total_transfer += added
                        transfer_events += 1
                        if fid in self.sensitive_files:
                            sensitive_transfer_bytes += added
                            sensitive_transfer_events += 1
            except Exception:
                pass
            for f in task.get_output_files():
                file_locations[f] = storage_service
            job = sim.create_standard_job([task], file_locations)
            compute_service.submit_standard_job(job)
            active[job] = (task, node, sim.get_simulated_time())
            steps += 1

        # Main simulation loop
        while True:
            while ready:
                if hasattr(scheduler, 'choose'):
                    t, node = scheduler.choose(ready, self.compute_nodes, self.node_capacities, node_loads, compute_service)
                    if t is None:
                        break
                else:
                    t = ready[0]
                    node = scheduler.schedule_task(t, self.compute_nodes, self.node_capacities, node_loads, compute_service)
                try:
                    ready.remove(t)
                except ValueError:
                    pass
                submit_task(t, node)
            if not active:
                if not ready:
                    break
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
                    try:
                        for f in task.get_output_files():
                            fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                            data_location[fid] = node
                            if hasattr(scheduler, 'data_location') and isinstance(getattr(scheduler, 'data_location'), dict):
                                scheduler.data_location[fid] = node
                    except Exception:
                        pass
                    if hasattr(scheduler, 'notify_task_completion'):
                        try:
                            scheduler.notify_task_completion(task.get_name(), node)
                        except Exception:
                            pass
                try:
                    ready = list(workflow.get_ready_tasks())
                except Exception:
                    ready = []

        makespan = sim.get_simulated_time()
        sim.terminate()
        rec = EvalRecord(
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
        rec.sensitive_transfer_bytes = sensitive_transfer_bytes  # type: ignore
        rec.sensitive_transfer_events = sensitive_transfer_events  # type: ignore
        rec.sensitive_file_fraction = self.sensitive_file_fraction  # type: ignore
        rec.zero_access_tasks = zero_access_tasks  # type: ignore
        rec.total_file_accesses = file_accesses  # type: ignore
        return rec

class PPOPaperScheduler:
    def __init__(self, model_path: str, rag_weight: float = 0.7, deterministic: bool = True, cost_blend_epsilon: float = 0.05):
        ckpt = torch.load(model_path, map_location='cpu')
        self.rag_weight = ckpt['config'].get('rag_weight', rag_weight)
        self.name = 'WASS-PPO-RAG'
        self.deterministic = deterministic
        self.cost_blend_epsilon = cost_blend_epsilon
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
        # Sensitivity ratio (defaults 0 if env not tracking)
        sens_ratio = 0.0
        try:
            wf = task.get_workflow()
            # Attempt to locate env sensitive file set via attribute injection (not guaranteed)
            # If not available, remain 0.0
            ins = task.get_input_files()
            if ins:
                # Heuristic: treat larger input counts as more likely to include sensitive data -> not accurate without env
                # We keep 0 unless an attribute 'sensitive_files' is attached dynamically to workflow (future extension)
                if hasattr(wf, 'sensitive_files'):
                    sset = getattr(wf, 'sensitive_files', set())
                    hits = 0
                    for f in ins:
                        fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                        if fid in sset:
                            hits += 1
                    if hits:
                        sens_ratio = hits / max(1, len(ins))
        except Exception:
            sens_ratio = 0.0
        task_features = [
            task.get_flops()/1e9,
            len(task.get_input_files()),
            task.get_number_of_children(),
            sens_ratio,
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
        # Hybrid cost+RL blend
        best_node, best_cost = min(exec_costs.items(), key=lambda kv: kv[1])
        rl_cost = exec_costs.get(rl_choice, best_cost)
        if self.deterministic:
            if rl_cost <= best_cost * (1.0 + self.cost_blend_epsilon):
                return rl_choice
            return best_node
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

    # Evaluation scaling + sensitivity config
    eval_cfg = base_cfg.get('evaluation', {})
    flops_range = tuple(eval_cfg['flops_range']) if 'flops_range' in eval_cfg else None
    file_size_range = tuple(eval_cfg['file_size_range']) if 'file_size_range' in eval_cfg else None
    capacity_scale = float(eval_cfg.get('capacity_scale', 1.0))
    rl_cost_blend_epsilon = float(eval_cfg.get('rl_cost_blend_epsilon', 0.05))
    sensitive_file_fraction = float(eval_cfg.get('sensitive_file_fraction', 0.0))
    sensitive_transfer_multiplier = float(eval_cfg.get('sensitive_transfer_multiplier', 1.0))
    if eval_cfg.get('workflow_sizes'):
        workflows = eval_cfg['workflow_sizes']

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
            schedulers['WASS-PPO-RAG'] = PPOPaperScheduler('models/wass_paper_aligned.pth', deterministic=deterministic, cost_blend_epsilon=rl_cost_blend_epsilon)
        except Exception as e:
            print('Skip WASS-PPO-RAG:', e)

    env = SimpleWorkflowEnv(platform_file, flops_range=flops_range, file_size_range=file_size_range, capacity_scale=capacity_scale,
                            sensitive_file_fraction=sensitive_file_fraction, sensitive_transfer_multiplier=sensitive_transfer_multiplier)
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
        agg.setdefault(rec.method, {'makespans': [], 'transfer_bytes': [], 'locality_hr': [], 'transfer_per_task': [], 'sens_bytes': [], 'sens_events': []})
        agg[rec.method]['makespans'].append(rec.makespan)
        agg[rec.method]['transfer_bytes'].append(rec.total_transfer_bytes)
        agg[rec.method]['locality_hr'].append(rec.locality_hit_rate)
        agg[rec.method]['transfer_per_task'].append(rec.transfer_bytes_per_task)
        # Dynamic sensitive metrics
        if hasattr(rec, 'sensitive_transfer_bytes'):
            agg[rec.method]['sens_bytes'].append(getattr(rec, 'sensitive_transfer_bytes'))
        if hasattr(rec, 'sensitive_transfer_events'):
            agg[rec.method]['sens_events'].append(getattr(rec, 'sensitive_transfer_events'))
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
            'mean_sensitive_transfer_bytes': float(np.array(d['sens_bytes']).mean()) if d['sens_bytes'] else 0.0,
            'mean_sensitive_transfer_events': float(np.array(d['sens_events']).mean()) if d['sens_events'] else 0.0,
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
    # Convert records including dynamic sensitive metrics
    out_records = []
    for r in records:
        base = asdict(r)
        # merge dynamic attrs if present
        for extra in ['sensitive_transfer_bytes','sensitive_transfer_events','sensitive_file_fraction']:
            if hasattr(r, extra):
                base[extra] = getattr(r, extra)
        out_records.append(base)
    with open(out_dir/'records.json','w') as f: json.dump(out_records, f, indent=2)
    with open(out_dir/'summary.json','w') as f: json.dump(summary, f, indent=2)
    print('Saved evaluation ->', out_dir)
    return summary

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/evaluate_paper_methods.py <config.yaml>')
        sys.exit(1)
    evaluate(sys.argv[1])
