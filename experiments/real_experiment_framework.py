#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WASS-RAG 真实实验框架 (V3 - 最终修复版)
核心逻辑修正：采用决策驱动的模拟循环，而非估算因子，以确保实验结果的科学有效性。
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    # --- 关键修复：添加 SchedulingAction 到导入列表 ---
    from ai_schedulers import create_scheduler, SchedulingState, SchedulingAction
    HAS_AI_SCHEDULERS = True
except ImportError as e:
    print(f"Warning: AI schedulers not available: {e}")
    HAS_AI_SCHEDULERS = False

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    workflow_sizes: List[int]
    scheduling_methods: List[str]
    cluster_sizes: List[int]
    repetitions: int
    output_dir: str
    ai_model_path: str = "models/wass_models.pth"
    knowledge_base_path: str = "data/knowledge_base.pkl"

@dataclass
class ExperimentResult:
    """单次实验结果"""
    method: str
    task_count: int
    cluster_size: int
    repetition: int
    makespan: float
    avg_cpu_util: float
    data_locality: float

class WassExperimentRunner:
    """WASS实验运行器 (修复版)"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ExperimentResult] = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ai_schedulers = {}
        if HAS_AI_SCHEDULERS:
            self._initialize_ai_schedulers()

    def _initialize_ai_schedulers(self):
        """预先加载所有需要的AI调度器"""
        ai_methods = ["WASS (Heuristic)", "WASS-DRL (w/o RAG)", "WASS-RAG"]
        for method in ai_methods:
            if method in self.config.scheduling_methods:
                print(f"Initializing {method} scheduler...")
                self.ai_schedulers[method] = create_scheduler(
                    method,
                    model_path=self.config.ai_model_path,
                    knowledge_base_path=self.config.knowledge_base_path
                )
                print(f"Successfully initialized {method}")

    def run_all_experiments(self):
        """运行所有实验配置"""
        total_experiments = len(self.config.workflow_sizes) * len(self.config.scheduling_methods) * \
                            len(self.config.cluster_sizes) * self.config.repetitions
        print(f"\nStarting experiments... Total experiments to run: {total_experiments}")
        
        exp_count = 0
        start_time = time.time()

        for task_count in self.config.workflow_sizes:
            for cluster_size in self.config.cluster_sizes:
                for rep in range(self.config.repetitions):
                    # 为每个重复实验生成一个固定的工作流和集群，确保所有算法面对相同的问题
                    workflow, cluster = self._generate_scenario(task_count, cluster_size, rep)
                    
                    for method in self.config.scheduling_methods:
                        exp_count += 1
                        progress = (exp_count / total_experiments) * 100
                        eta = ((time.time() - start_time) / exp_count) * (total_experiments - exp_count) if exp_count > 0 else 0
                        
                        print(f"\nRunning ({exp_count}/{total_experiments}): {method}, {task_count} tasks, {cluster_size} nodes, rep {rep}")
                        print(f"Progress: {progress:.1f}%, ETA: {eta:.1f}s")

                        result = self._run_single_simulation(workflow, cluster, method, rep)
                        self.results.append(result)

        print(f"\nCompleted all experiments in {time.time() - start_time:.2f}s")
        self._save_and_analyze_results()

    def _generate_scenario(self, task_count: int, cluster_size: int, seed: int):
        """生成一个固定的工作流和集群场景"""
        np.random.seed(seed * 1000 + task_count + cluster_size) # 确保可复现

        # --- FIX: Added 'current_load' to match the data structure used in training ---
        cluster = {f"node_{j}": {
            "cpu_capacity": round(np.random.uniform(2.0, 8.0), 2),
            "memory_capacity": round(np.random.uniform(8.0, 64.0), 2),
            "current_load": round(np.random.uniform(0.1, 0.9), 2) # This was the missing piece
        } for j in range(cluster_size)}

        tasks = []
        for i in range(task_count):
            task = {"id": f"task_{i}", "flops": float(np.random.uniform(1e9, 20e9)), "memory": round(np.random.uniform(1.0, 8.0), 2), "dependencies": []}
            tasks.append(task)

        for i in range(1, task_count):
            dep_candidate = np.random.randint(0, i)
            if f"task_{dep_candidate}" not in tasks[i]["dependencies"]:
                tasks[i]["dependencies"].append(f"task_{dep_candidate}")
            # Increase dependency density slightly for more complex graphs
            for j in range(i):
                if np.random.rand() < 0.2 and f"task_{j}" not in tasks[i]["dependencies"]:
                     tasks[i]["dependencies"].append(f"task_{j}")

        workflow = {"tasks": tasks}
        return workflow, cluster

    def _run_single_simulation(self, workflow: Dict, cluster: Dict, method: str, repetition: int) -> ExperimentResult:
        """运行一次完整的、决策驱动的仿真"""
        
        node_available_time = {node: 0.0 for node in cluster}
        task_finish_time = {}
        task_placements = {}
        total_cpu_work_done = 0

        if method == "FIFO":
            scheduler = self._get_fifo_scheduler()
        elif method == "HEFT":
            scheduler = self._get_heft_scheduler(workflow, cluster)
        else:
            scheduler = self.ai_schedulers.get(method)
            if scheduler is None:
                raise ValueError(f"Scheduler for method '{method}' not initialized.")
        
        pending_tasks = {task['id'] for task in workflow['tasks']}
        
        while pending_tasks:
            ready_tasks = []
            for task_id in sorted(list(pending_tasks)):
                task = next(t for t in workflow['tasks'] if t['id'] == task_id)
                if all(dep in task_finish_time for dep in task['dependencies']):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                if not pending_tasks: break
                raise RuntimeError("Simulation stuck: No ready tasks but pending tasks exist.")

            if method == "HEFT":
                next_task_obj = scheduler.get_next_task(ready_tasks)
                ready_tasks = [next_task_obj] if next_task_obj else []

            if not ready_tasks: continue

            for task in ready_tasks:
                current_task_id = task['id']

                earliest_start_times = {}
                for node in cluster:
                    data_ready_time = 0
                    for dep_id in task['dependencies']:
                        dep_finish_time = task_finish_time[dep_id]
                        transfer_time = 0.1 if task_placements.get(dep_id) != node else 0
                        data_ready_time = max(data_ready_time, dep_finish_time + transfer_time)
                    earliest_start_times[node] = max(node_available_time[node], data_ready_time)

                state = SchedulingState(
                    workflow_graph=workflow,
                    cluster_state={
                        "nodes": cluster, 
                        "earliest_start_times": earliest_start_times
                    },
                    pending_tasks=list(pending_tasks),
                    current_task=current_task_id,
                    available_nodes=list(cluster.keys()),
                    timestamp=min(node_available_time.values())
                )
                
                decision = scheduler.make_decision(state)
                chosen_node = decision.target_node
                
                task_flops = task['flops'] / 1e9
                node_cpu = cluster[chosen_node]['cpu_capacity']
                exec_time = task_flops / node_cpu
                
                start_time = earliest_start_times[chosen_node]
                finish_time = start_time + exec_time
                
                task_finish_time[current_task_id] = finish_time
                task_placements[current_task_id] = chosen_node
                node_available_time[chosen_node] = finish_time
                total_cpu_work_done += task_flops
                
                if current_task_id in pending_tasks:
                    pending_tasks.remove(current_task_id)

        makespan = max(task_finish_time.values()) if task_finish_time else 0
        total_cluster_cpu_seconds = sum(c['cpu_capacity'] for c in cluster.values()) * makespan
        avg_cpu_util = total_cpu_work_done / total_cluster_cpu_seconds if total_cluster_cpu_seconds > 0 else 0

        transfers = 0
        total_deps = 0
        for task in workflow['tasks']:
            total_deps += len(task['dependencies'])
            for dep in task['dependencies']:
                if task_placements.get(task['id']) != task_placements.get(dep):
                    transfers += 1
        data_locality = (1 - (transfers / total_deps)) if total_deps > 0 else 1.0

        return ExperimentResult(
            method=method,
            task_count=len(workflow['tasks']),
            cluster_size=len(cluster),
            repetition=repetition,
            makespan=round(makespan, 4),
            avg_cpu_util=round(avg_cpu_util, 4),
            data_locality=round(data_locality, 4)
        )

    def _get_fifo_scheduler(self):
        class FifoScheduler:
            def make_decision(self, state):
                earliest_finish_time, best_node = float('inf'), None
                task = next(t for t in state.workflow_graph['tasks'] if t['id'] == state.current_task)
                for node, est in state.cluster_state['earliest_start_times'].items():
                    exec_time = (task['flops'] / 1e9) / state.cluster_state['nodes'][node]['cpu_capacity']
                    if est + exec_time < earliest_finish_time:
                        earliest_finish_time, best_node = est + exec_time, node
                return SchedulingAction(state.current_task, best_node or state.available_nodes[0], 1.0)
        return FifoScheduler()

    def _get_heft_scheduler(self, workflow, cluster):
        class HeftScheduler:
            def __init__(self, workflow, cluster):
                self.ranks = self._calculate_ranks(workflow, cluster)
                self.task_order = sorted(self.ranks.keys(), key=lambda t: self.ranks[t], reverse=True)
                self.scheduled_tasks = set()

            def _calculate_ranks(self, workflow, cluster):
                tasks = {t['id']: t for t in workflow['tasks']}
                avg_exec = {tid: np.mean([(t['flops']/1e9)/n['cpu_capacity'] for n in cluster.values()]) for tid, t in tasks.items()}
                ranks = {}
                def get_rank(task_id):
                    if task_id in ranks: return ranks[task_id]
                    successors = [t['id'] for t in workflow['tasks'] if task_id in t['dependencies']]
                    max_succ_rank = max((get_rank(s) for s in successors), default=0)
                    ranks[task_id] = avg_exec[task_id] + max_succ_rank
                    return ranks[task_id]
                for task_id in tasks: get_rank(task_id)
                return ranks
            
            def get_next_task(self, ready_tasks):
                for task_id in self.task_order:
                    if task_id not in self.scheduled_tasks:
                        for ready_task in ready_tasks:
                            if ready_task['id'] == task_id:
                                self.scheduled_tasks.add(task_id)
                                return ready_task
                return None

            def make_decision(self, state):
                earliest_finish_time, best_node = float('inf'), None
                task = next(t for t in state.workflow_graph['tasks'] if t['id'] == state.current_task)
                for node, est in state.cluster_state['earliest_start_times'].items():
                    exec_time = (task['flops'] / 1e9) / state.cluster_state['nodes'][node]['cpu_capacity']
                    if est + exec_time < earliest_finish_time:
                        earliest_finish_time, best_node = est + exec_time, node
                return SchedulingAction(state.current_task, best_node or state.available_nodes[0], 1.0)
        return HeftScheduler(workflow, cluster)
        
    def _save_and_analyze_results(self):
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"Results saved to: {results_file}")

        table_data = {}
        fifo_results = [r for r in self.results if r.method == "FIFO"]
        baseline_makespan = np.mean([r.makespan for r in fifo_results]) if fifo_results else 0

        for method in self.config.scheduling_methods:
            method_results = [r for r in self.results if r.method == method]
            if not method_results: continue
            
            avg_makespan = np.mean([r.makespan for r in method_results])
            improvement = ((baseline_makespan - avg_makespan) / baseline_makespan) * 100 if baseline_makespan > 0 else 0
            
            table_data[method] = {
                "makespan": f"{avg_makespan:.2f}",
                "improvement": f"{improvement:.1f}%",
                "cpu_util": f"{np.mean([r.avg_cpu_util for r in method_results]) * 100:.1f}%",
                "data_locality": f"{np.mean([r.data_locality for r in method_results]) * 100:.1f}%"
            }
        
        print("\n--- Final Performance Comparison ---")
        print(f"{'Method':<18} {'Makespan (s)':<15} {'Improvement':<15} {'CPU Util':<12} {'Data Locality':<15}")
        print("-" * 75)
        for method, data in table_data.items():
            print(f"{method:<18} {data['makespan']:<15} {data['improvement']:<15} {data['cpu_util']:<12} {data['data_locality']:<15}")

def main():
    """主函数"""
    config = ExperimentConfig(
        name="WASS-RAG Performance Evaluation (Corrected)",
        workflow_sizes=[10, 20, 49, 100],
        scheduling_methods=[
            "FIFO", 
            "HEFT",
            "WASS (Heuristic)",
            "WASS-DRL (w/o RAG)",
            "WASS-RAG"
        ],
        cluster_sizes=[4, 8, 16],
        repetitions=3,
        output_dir="results/final_experiments"
    )
    
    runner = WassExperimentRunner(config)
    runner.run_all_experiments()

if __name__ == "__main__":
    main()