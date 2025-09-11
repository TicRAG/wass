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

# Import legacy experiment schedulers (reuse code)
from experiments.wrench_real_experiment import (
    FIFOScheduler, HEFTScheduler, WASSHeuristicScheduler, WASSDRLScheduler
)

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

class SimpleWorkflowEnv:
    def __init__(self, platform_file: str):
        if wrench is None:
            raise RuntimeError("WRENCH not available.")
        self.platform_file = platform_file
        with open(platform_file,'r',encoding='utf-8') as f:
            self.platform_xml = f.read()
        self.compute_nodes = ["ComputeHost1","ComputeHost2","ComputeHost3","ComputeHost4"]
        self.node_capacities = {"ComputeHost1":2.0,"ComputeHost2":3.0,"ComputeHost3":2.5,"ComputeHost4":4.0}

    def run(self, workflow_size: int, scheduler) -> EvalRecord:
        sim = wrench.Simulation(); sim.start(self.platform_xml, "ControllerHost")
        storage_service = sim.create_simple_storage_service("StorageHost", ["/storage"])
        compute_resources = {n:(4,8_589_934_592) for n in self.compute_nodes}
        compute_service = sim.create_bare_metal_compute_service("ComputeHost1", compute_resources, "/scratch", {}, {})
        workflow = sim.create_workflow()
        tasks = []
        files = []
        for i in range(workflow_size):
            flops = random.uniform(2e9, 10e9)
            task = workflow.add_task(f"task_{i}", flops, 1,1,0)
            tasks.append(task)
            if i < workflow_size -1:
                ofile = sim.add_file(f"f_{i}", random.randint(1024,10240))
                task.add_output_file(ofile); files.append(ofile)
        for i in range(1, len(tasks)):
            if random.random() < 0.3:
                dep = random.randint(0, i-1)
                if dep < len(files):
                    tasks[i].add_input_file(files[dep])
        for f in files: storage_service.create_file_copy(f)
        node_loads = {n:0.0 for n in self.compute_nodes}
        ready = workflow.get_ready_tasks()
        start_wall = time.time()
        steps = 0
        while ready:
            t = ready[0]
            chosen = scheduler.schedule_task(t, self.compute_nodes, self.node_capacities, node_loads, compute_service)
            file_locations = {f:storage_service for f in t.get_input_files()}
            for f in t.get_output_files(): file_locations[f]=storage_service
            job = sim.create_standard_job([t], file_locations)
            compute_service.submit_standard_job(job)
            st = sim.get_simulated_time()
            while True:
                ev = sim.wait_for_next_event()
                if ev['event_type'] == 'standard_job_completion' and ev['standard_job']==job: break
                if ev['event_type'] == 'simulation_termination': break
            et = sim.get_simulated_time(); exec_t = et - st
            node_loads[chosen] += exec_t
            ready = workflow.get_ready_tasks(); steps += 1
        makespan = sim.get_simulated_time(); sim.terminate()
        return EvalRecord(method=scheduler.name, workflow_id=f"wf_{workflow_size}_{random.randint(0,9999)}", tasks=workflow_size, makespan=makespan, steps=steps, runtime_ms=(time.time()-start_wall)*1000)

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
        task_features = [task.get_flops()/1e9, len(task.get_input_files()), task.get_number_of_children(), 0.0, 0.0]
        node_feats = []
        for n in available_nodes:
            cap = node_capacities.get(n,1.0); exec_time = task.get_flops()/(cap*1e9)
            node_feats.extend([cap/4.0, 0.0, exec_time/10.0])
        import numpy as np
        flat = np.array(task_features + node_feats, dtype=np.float32)
        # Try to get workflow reference from task (depends on wrench API); fallback zeros if unavailable
        try:
            workflow = task.get_workflow()  # if API provides
            graph_emb = self._graph_embedding(workflow)
        except Exception:
            graph_emb = np.zeros(self.encoder.out_dim, dtype=np.float32)
        state_vec = np.concatenate([flat, graph_emb], axis=0)
        with torch.no_grad():
            logits, _ = self.model(torch.from_numpy(state_vec).float().unsqueeze(0))
            if self.deterministic:
                act = int(torch.argmax(logits, dim=-1).item())
            else:
                act = torch.distributions.Categorical(logits=logits).sample().item()
        return available_nodes[act % len(available_nodes)]


def evaluate(config_path: str, workflows: List[int] = None, repetitions: int = 3):
    with open(config_path,'r',encoding='utf-8') as f:
        base_cfg = json.load(f) if config_path.endswith('.json') else __import__('yaml').safe_load(f)
    platform_file = base_cfg.get('platform',{}).get('platform_file','configs/platform.xml')
    seed = base_cfg.get('random_seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    workflows = workflows or [5,10,15,25,35,50,60,80]

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
        agg.setdefault(rec.method, {'makespans': []})['makespans'].append(rec.makespan)
        size_buckets.setdefault(rec.method, {}).setdefault(rec.tasks, []).append(rec.makespan)
    summary = {}
    for m, d in agg.items():
        arr = np.array(d['makespans'])
        summary[m] = {
            'count': int(arr.size),
            'mean_makespan': float(arr.mean()),
            'std_makespan': float(arr.std()),
            'best': float(arr.min()),
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
