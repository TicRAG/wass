#!/usr/bin/env python3
"""
WASS-RAG 阶段一：知识库播种脚本

该脚本利用修复后的离散事件仿真器，运行大量高质量的基准调度算法（如 HEFT），
并将详细的仿真过程数据记录下来，为后续的 Performance Predictor 和 DRL Agent 训练
提供高质量的、源于“真实”环境的数据集。
"""

import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
import copy

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'src'))

# 导入仿真器和调度器
from experiments.real_experiment_framework import WassExperimentRunner, ExperimentConfig
from src.ai_schedulers import create_scheduler, SchedulingState

@dataclass
class TrainingSample:
    """定义一条用于训练的样本"""
    state_features: List[float]
    action_features: List[float]
    context_features: List[float]
    # 目标值
    final_makespan: float # 整个工作流的最终完工时间
    achieved_finish_time: float # 这个特定任务的完成时间

class KnowledgeSeedingFramework(WassExperimentRunner):
    """
    一个专门用于生成训练数据集的仿真框架子类。
    它重写了仿真循环，以捕获每一步的详细状态和决策信息。
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.training_dataset: List[TrainingSample] = []
        # 我们需要一个临时的WASS-RAG调度器实例，来借用它的特征编码方法
        self.feature_encoder = create_scheduler("WASS-RAG")

    def run_and_collect_data(self):
        """运行仿真并收集所有决策点的数据"""
        total_simulations = len(self.config.workflow_sizes) * len(self.config.cluster_sizes) * self.config.repetitions
        print(f"🚀 Starting Knowledge Seeding process for {total_simulations} simulations...")
        
        base_seed = int(time.time())
        sim_count = 0

        for task_count in self.config.workflow_sizes:
            for cluster_size in self.config.cluster_sizes:
                for rep in range(self.config.repetitions):
                    sim_count += 1
                    print(f"\n--- Running simulation {sim_count}/{total_simulations} (Tasks: {task_count}, Nodes: {cluster_size}, Rep: {rep}) ---")
                    
                    scenario_seed = base_seed + sim_count
                    workflow, cluster = self._generate_scenario(task_count, cluster_size, scenario_seed)
                    
                    # 我们只使用 HEFT 来生成高质量的初始决策数据
                    self._run_simulation_and_capture(workflow, cluster, "HEFT")
        
        self._save_dataset()

    def _run_simulation_and_capture(self, workflow: Dict, cluster: Dict, method: str):
        """
        重写的仿真循环，核心目标是捕获每一个决策点的 (state, action, outcome) 数据。
        """
        # 仿真状态变量
        node_available_time = {node: 0.0 for node in cluster}
        task_finish_time = {}
        task_placements = {}
        
        # 数据捕获变量
        decision_records = []

        # 初始化调度器
        scheduler = self._get_heft_scheduler(workflow, cluster)
        pending_tasks_set = {task['id'] for task in workflow['tasks']}
        
        while pending_tasks_set:
            ready_tasks = [
                task for task_id in sorted(list(pending_tasks_set))
                if all(dep in task_finish_time for dep in (task := next(t for t in workflow['tasks'] if t['id'] == task_id))['dependencies'])
            ]

            if not ready_tasks:
                if not pending_tasks_set: break
                raise RuntimeError("Simulation stuck: No ready tasks.")
            
            task_to_schedule = scheduler.get_next_task(ready_tasks)
            if not task_to_schedule: continue

            current_task_id = task_to_schedule['id']
            current_sim_time = min(node_available_time.values())

            earliest_start_times = {}
            for node in cluster:
                data_ready_time = max([task_finish_time.get(dep, 0) + (0.1 if task_placements.get(dep) != node else 0) for dep in task_to_schedule['dependencies']], default=0)
                earliest_start_times[node] = max(node_available_time[node], data_ready_time)

            state = SchedulingState(
                workflow_graph=workflow,
                cluster_state={"nodes": cluster, "earliest_start_times": earliest_start_times},
                pending_tasks=list(pending_tasks_set),
                current_task=current_task_id,
                available_nodes=list(cluster.keys()),
                timestamp=current_sim_time
            )
            
            # HEFT 做出决策
            decision = scheduler.make_decision(state)
            chosen_node = decision.target_node
            
            # --- 关键：捕获决策瞬间的状态和动作特征 ---
            state_features = self.feature_encoder._extract_simple_features_fallback(state).cpu().numpy().tolist()
            action_features = self.feature_encoder._encode_action(chosen_node, state).cpu().numpy().tolist()
            # 在这个阶段，我们还没有RAG上下文
            context_features = np.zeros(32).tolist()

            # 更新仿真状态
            task_flops = task_to_schedule['flops']
            node_cpu_gflops = cluster[chosen_node]['cpu_capacity']
            exec_time = task_flops / (node_cpu_gflops * 1e9)
            start_time = earliest_start_times[chosen_node]
            finish_time = start_time + exec_time
            
            task_finish_time[current_task_id] = finish_time
            task_placements[current_task_id] = chosen_node
            node_available_time[chosen_node] = finish_time
            pending_tasks_set.remove(current_task_id)

            # 记录这次决策
            decision_records.append({
                "state_features": state_features,
                "action_features": action_features,
                "context_features": context_features,
                "achieved_finish_time": finish_time
            })

        # 整个工作流完成后，计算最终 makespan
        final_makespan = max(task_finish_time.values()) if task_finish_time else 0
        print(f"  Simulation complete. Final Makespan: {final_makespan:.2f}s")

        # 将最终 makespan 回填到每一条记录中，并存入主数据集
        for record in decision_records:
            self.training_dataset.append(TrainingSample(
                state_features=record['state_features'],
                action_features=record['action_features'],
                context_features=record['context_features'],
                final_makespan=final_makespan,
                achieved_finish_time=record['achieved_finish_time']
            ))

    def _save_dataset(self):
        """将收集到的数据集保存到文件"""
        output_file = Path("data/kb_training_dataset.json")
        output_file.parent.mkdir(exist_ok=True)
        
        dataset_as_dicts = [asdict(sample) for sample in self.training_dataset]
        
        with open(output_file, 'w') as f:
            json.dump(dataset_as_dicts, f, indent=2)
            
        print(f"\n✅ Successfully generated and saved {len(self.training_dataset)} training samples to {output_file}")


def main():
    """主函数"""
    # 配置生成的数据量
    config = ExperimentConfig(
        name="Knowledge Base Seeding",
        workflow_sizes=[50, 100, 150], # 使用更大、更复杂的工作流
        scheduling_methods=["HEFT"],  # 只使用HEFT
        cluster_sizes=[8, 16, 32],
        repetitions=10, # 更多的重复次数以产生丰富的数据
        output_dir="temp_kb_seeding_results" # 临时目录，我们不关心这里的最终性能
    )
    
    seeder = KnowledgeSeedingFramework(config)
    seeder.run_and_collect_data()

if __name__ == "__main__":
    main()