#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WASS-RAG 真实实验框架 (V4 - 离散事件仿真修复版)
核心逻辑修正：采用决策驱动的、真正的离散事件模拟循环，取代了原有的批处理逻辑，
以确保实验结果的科学有效性、数据多样性和决策准确性。
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
        # --- 修复：创建一个基础种子，确保每次运行的实验集都不同 ---
        base_seed = int(time.time())

        for task_count in self.config.workflow_sizes:
            for cluster_size in self.config.cluster_sizes:
                for rep in range(self.config.repetitions):
                    # --- 修复：为每次重复实验生成一个唯一的、可复现的场景 ---
                    # 这样所有算法在同一次重复中面对相同的问题，但不同重复之间的问题是不同的
                    scenario_seed = base_seed + task_count * 1000 + cluster_size * 100 + rep
                    workflow, cluster = self._generate_scenario(task_count, cluster_size, scenario_seed)
                    
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
        """生成一个固定的、可复现的工作流和集群场景"""
        # --- 修复：使用传入的唯一种子 ---
        np.random.seed(seed)

        cluster = {f"node_{j}": {
            "cpu_capacity": round(np.random.uniform(2.0, 8.0), 2),
            "memory_capacity": round(np.random.uniform(8.0, 64.0), 2),
            "current_load": round(np.random.uniform(0.1, 0.5), 2) # 初始负载
        } for j in range(cluster_size)}

        tasks = []
        for i in range(task_count):
            task = {
                "id": f"task_{i}", 
                "flops": float(np.random.uniform(1e9, 20e9)), 
                "memory": round(np.random.uniform(1.0, 8.0), 2), 
                "dependencies": []
            }
            tasks.append(task)
        
        # 创建一个更真实的DAG（有向无环图）结构
        for i in range(task_count):
            # 每个任务最多依赖于它之前的5个任务中的随机几个
            num_dependencies = np.random.randint(0, min(i, 5) + 1)
            if i > 0 and num_dependencies > 0:
                dependencies = np.random.choice(range(i), size=num_dependencies, replace=False)
                tasks[i]["dependencies"] = [f"task_{dep_idx}" for dep_idx in dependencies]

        workflow = {"tasks": tasks}
        return workflow, cluster

    def _run_single_simulation(self, workflow: Dict, cluster: Dict, method: str, repetition: int) -> ExperimentResult:
        """
        --- 核心重构：运行一次完整的、真正的离散事件驱动仿真 ---
        取代了旧的批处理逻辑，确保状态实时更新和决策准确性。
        """
        node_available_time = {node: 0.0 for node in cluster}
        task_finish_time = {}
        task_placements = {}
        total_cpu_work_done = 0

        # 初始化调度器
        if method == "FIFO": scheduler = self._get_fifo_scheduler()
        elif method == "HEFT": scheduler = self._get_heft_scheduler(workflow, cluster)
        else:
            scheduler = self.ai_schedulers.get(method)
            if scheduler is None: raise ValueError(f"Scheduler for method '{method}' not initialized.")
            scheduler.reset()

        pending_tasks_set = {task['id'] for task in workflow['tasks']}
        
        # --- 主模拟循环：只要还有未完成的任务就继续 ---
        while pending_tasks_set:
            ready_tasks = []
            # 1. 查找所有依赖项均已完成的任务
            for task_id in sorted(list(pending_tasks_set)): # 排序以保证FIFO的确定性
                task = next(t for t in workflow['tasks'] if t['id'] == task_id)
                if all(dep in task_finish_time for dep in task['dependencies']):
                    ready_tasks.append(task)

            if not ready_tasks:
                if not pending_tasks_set: break # 所有任务完成
                # 如果没有就绪任务，但仍有待处理任务，说明工作流定义有误（例如循环依赖）
                raise RuntimeError(f"Simulation stuck for method {method}: No ready tasks but {len(pending_tasks_set)} tasks pending. Check workflow for cycles.")
            
            # 2. 从就绪任务中选择一个进行调度 (这是离散事件的核心)
            # 对于HEFT，它有自己的优先级顺序。对于其他算法，我们一次只处理一个，以确保状态更新。
            if method == "HEFT":
                task_to_schedule = scheduler.get_next_task(ready_tasks)
            else:
                task_to_schedule = ready_tasks[0] # FIFO 和 AI 调度器一次处理一个

            if not task_to_schedule:
                # 这种情况可能在HEFT中发生，如果最高优先级的任务还未就绪
                # 我们需要等待，但在这个简化模拟中，我们直接进入下个循环
                continue

            current_task_id = task_to_schedule['id']
            
            # 3. 为选定的任务创建当前的系统状态
            current_sim_time = min(node_available_time.values()) # 推进时钟到最早可能开始的时间点
            
            # 计算每个节点的最早可开始时间（EST）
            earliest_start_times = {}
            for node in cluster:
                # 任务的数据依赖何时准备好
                data_ready_time = 0
                for dep_id in task_to_schedule['dependencies']:
                    dep_finish_time = task_finish_time.get(dep_id, 0)
                    # 简化：如果依赖项在不同节点，增加0.1s的传输时间
                    transfer_time = 0.1 if task_placements.get(dep_id) != node else 0
                    data_ready_time = max(data_ready_time, dep_finish_time + transfer_time)
                
                # 节点本身何时空闲
                node_free_time = node_available_time[node]
                earliest_start_times[node] = max(node_free_time, data_ready_time)

            state = SchedulingState(
                workflow_graph=workflow,
                cluster_state={
                    "nodes": cluster, # 注意：这里的负载是初始负载，动态负载通过 node_available_time 体现
                    "earliest_start_times": earliest_start_times
                },
                pending_tasks=list(pending_tasks_set),
                current_task=current_task_id,
                available_nodes=list(cluster.keys()),
                timestamp=current_sim_time
            )
            
            # 4. 做出调度决策
            decision = scheduler.make_decision(state)
            chosen_node = decision.target_node
            
            # 5. 基于决策，立即更新系统状态 (这是关键!)
            task_flops = task_to_schedule['flops']
            node_cpu_gflops = cluster[chosen_node]['cpu_capacity']
            exec_time = task_flops / (node_cpu_gflops * 1e9) # 转换为 GigaFlops
            
            start_time = earliest_start_times[chosen_node]
            finish_time = start_time + exec_time
            
            task_finish_time[current_task_id] = finish_time
            task_placements[current_task_id] = chosen_node
            node_available_time[chosen_node] = finish_time # 关键：立即更新节点的可用时间！
            total_cpu_work_done += task_flops
            
            pending_tasks_set.remove(current_task_id)
            
            # print(f"  - Scheduled {current_task_id} on {chosen_node}, finishes at {finish_time:.2f}s") # 用于调试

        # --- 最终指标计算 ---
        makespan = max(task_finish_time.values()) if task_finish_time else 0
        total_cluster_cpu_seconds = sum(c['cpu_capacity'] * 1e9 for c in cluster.values()) * makespan
        avg_cpu_util = total_cpu_work_done / total_cluster_cpu_seconds if total_cluster_cpu_seconds > 0 else 0

        transfers = 0
        total_deps = 0
        for task in workflow['tasks']:
            task_id = task['id']
            if task_id in task_placements and task['dependencies']:
                task_node = task_placements[task_id]
                for dep_id in task['dependencies']:
                    total_deps += 1
                    if dep_id in task_placements and task_placements[dep_id] != task_node:
                        transfers += 1

        data_locality = (1.0 - (transfers / total_deps)) if total_deps > 0 else 1.0

        return ExperimentResult(
            method=method,
            task_count=len(workflow['tasks']),
            cluster_size=len(cluster),
            repetition=repetition,
            makespan=round(makespan, 4),
            avg_cpu_util=round(avg_cpu_util, 4),
            data_locality=round(data_locality, 4)
        )

    # --- 调度器辅助方法 (保持不变) ---
    def _get_fifo_scheduler(self):
        class FifoScheduler:
            def make_decision(self, state):
                earliest_finish_time, best_node = float('inf'), None
                task = next(t for t in state.workflow_graph['tasks'] if t['id'] == state.current_task)
                for node, est in state.cluster_state['earliest_start_times'].items():
                    exec_time = task['flops'] / (state.cluster_state['nodes'][node]['cpu_capacity'] * 1e9)
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
                avg_exec = {tid: np.mean([t['flops'] / (n['cpu_capacity'] * 1e9) for n in cluster.values()]) for tid, t in tasks.items()}
                ranks = {}
                def get_rank(task_id):
                    if task_id in ranks: return ranks[task_id]
                    successors = [t['id'] for t in workflow['tasks'] if task_id in t.get('dependencies', [])]
                    max_succ_rank = max((get_rank(s) for s in successors), default=0)
                    ranks[task_id] = avg_exec[task_id] + max_succ_rank
                    return ranks[task_id]
                for task_id in tasks: get_rank(task_id)
                return ranks
            
            def get_next_task(self, ready_tasks):
                ready_task_ids = {t['id'] for t in ready_tasks}
                for task_id in self.task_order:
                    if task_id in ready_task_ids and task_id not in self.scheduled_tasks:
                        return next(t for t in ready_tasks if t['id'] == task_id)
                return None

            def make_decision(self, state):
                self.scheduled_tasks.add(state.current_task)
                earliest_finish_time, best_node = float('inf'), None
                task = next(t for t in state.workflow_graph['tasks'] if t['id'] == state.current_task)
                for node, est in state.cluster_state['earliest_start_times'].items():
                    exec_time = task['flops'] / (state.cluster_state['nodes'][node]['cpu_capacity'] * 1e9)
                    if est + exec_time < earliest_finish_time:
                        earliest_finish_time, best_node = est + exec_time, node
                return SchedulingAction(state.current_task, best_node or state.available_nodes[0], 1.0)
        return HeftScheduler(workflow, cluster)

    def _save_and_analyze_results(self):
        # --- 将结果保存为直接的列表，以简化图表脚本的加载逻辑 ---
        results_list = [asdict(r) for r in self.results]
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            # 兼容新旧格式，新格式直接是列表
            json.dump(results_list, f, indent=2)
        print(f"Results saved to: {results_file}")

        # --- 分析逻辑保持不变 ---
        table_data = {}
        # 修复：基准应该是性能最好的传统算法之一，这里我们选择 HEFT
        heft_results = [r for r in self.results if r.method == "HEFT"]
        baseline_makespan = np.mean([r.makespan for r in heft_results]) if heft_results else 0
        print(f"\n--- Analysis complete. Baseline (HEFT average makespan): {baseline_makespan:.2f}s ---")

        for method in self.config.scheduling_methods:
            method_results = [r for r in self.results if r.method == method]
            if not method_results: continue
            
            avg_makespan = np.mean([r.makespan for r in method_results])
            improvement = ((baseline_makespan - avg_makespan) / baseline_makespan) * 100 if baseline_makespan > 0 else 0
            
            table_data[method] = {
                "makespan": f"{avg_makespan:.2f}",
                "improvement_vs_heft": f"{improvement:.1f}%",
                "cpu_util": f"{np.mean([r.avg_cpu_util for r in method_results]) * 100:.1f}%",
                "data_locality": f"{np.mean([r.data_locality for r in method_results]) * 100:.1f}%"
            }
        
        print("\n--- Final Performance Comparison (vs HEFT) ---")
        print(f"{'Method':<20} {'Makespan (s)':<15} {'Improvement':<15} {'CPU Util':<12} {'Data Locality':<15}")
        print("-" * 80)
        for method, data in table_data.items():
            print(f"{method:<20} {data['makespan']:<15} {data['improvement_vs_heft']:<15} {data['cpu_util']:<12} {data['data_locality']:<15}")

def main():
    """主函数"""
    config = ExperimentConfig(
        name="WASS-RAG Performance Evaluation (Discrete-Event Simulation)",
        workflow_sizes=[10, 50, 100], # 简化配置以加快测试
        scheduling_methods=[
            "FIFO", 
            "HEFT",
            "WASS (Heuristic)",
            "WASS-DRL (w/o RAG)",
            "WASS-RAG"
        ],
        cluster_sizes=[4, 8, 16],
        repetitions=3, # 增加重复次数以获得更可靠的统计数据
        output_dir="results/final_experiments_discrete_event" # 使用新的输出目录
    )
    
    runner = WassExperimentRunner(config)
    runner.run_all_experiments()

if __name__ == "__main__":
    main()