#!/usr/bin/env python3
"""
增强的RAG知识库生成器
生成更多样化的工作流案例，提高RAG系统的覆盖范围和质量
"""

import sys
import os
import numpy as np
import random
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.knowledge_base.wrench_full_kb import WRENCHKnowledgeCase, WRENCHRAGKnowledgeBase
from scripts.workflow_generator import WorkflowGenerator
from scripts.platform_generator import PlatformGenerator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedWorkflowConfig:
    """增强的工作流配置"""
    workflow_size: int  # 工作流大小（任务数量）
    ccr: float  # 通信计算比
    dependency_probability: float  # 依赖概率
    workflow_type: str  # 工作流类型
    computation_size_range: Tuple[float, float]  # 计算大小范围
    data_size_range: Tuple[float, float]  # 数据大小范围
    
class EnhancedRAGKnowledgeBaseGenerator:
    """增强的RAG知识库生成器"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化工作流和平台生成器
        self.workflow_generator = WorkflowGenerator()
        self.platform_generator = PlatformGenerator()
        
        # 调度器类型
        self.scheduler_types = ["HEFT", "CPOP", "PEFT", "Lookahead", "Random", "MinMin", "MaxMin", "RoundRobin"]
        
        # 工作流类型
        self.workflow_types = ["chain", "tree", "dag", "fork_join", "pipeline", "hybrid"]
        
        # 增强的工作流配置
        self.workflow_configs = self._generate_workflow_configs()
        
        logger.info(f"Initialized EnhancedRAGKnowledgeBaseGenerator with {len(self.workflow_configs)} configurations")
    
    def _generate_workflow_configs(self) -> List[EnhancedWorkflowConfig]:
        """生成多样化的工作流配置"""
        configs = []
        
        # 基础配置：不同规模的工作流
        for size in [5, 10, 15, 20, 25, 30, 40, 50]:
            for ccr in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                for workflow_type in self.workflow_types:
                    # 根据工作流类型调整依赖概率
                    if workflow_type == "chain":
                        dep_prob = 0.9
                    elif workflow_type == "tree":
                        dep_prob = 0.7
                    elif workflow_type == "dag":
                        dep_prob = 0.5
                    elif workflow_type == "fork_join":
                        dep_prob = 0.6
                    elif workflow_type == "pipeline":
                        dep_prob = 0.8
                    else:  # hybrid
                        dep_prob = 0.5
                    
                    configs.append(EnhancedWorkflowConfig(
                        workflow_size=size,
                        ccr=ccr,
                        dependency_probability=dep_prob,
                        workflow_type=workflow_type,
                        computation_size_range=(1e9, 1e11),  # 1-100 Gflops
                        data_size_range=(1e6, 1e9)  # 1MB-1GB
                    ))
        
        # 极端配置：测试边界情况
        extreme_configs = [
            # 极小工作流
            EnhancedWorkflowConfig(
                workflow_size=3,
                ccr=0.1,
                dependency_probability=0.9,
                workflow_type="chain",
                computation_size_range=(1e8, 1e9),
                data_size_range=(1e5, 1e6)
            ),
            # 极大工作流
            EnhancedWorkflowConfig(
                workflow_size=100,
                ccr=10.0,
                dependency_probability=0.3,
                workflow_type="dag",
                computation_size_range=(1e10, 1e12),
                data_size_range=(1e8, 1e10)
            ),
            # 计算密集型
            EnhancedWorkflowConfig(
                workflow_size=20,
                ccr=0.01,
                dependency_probability=0.5,
                workflow_type="dag",
                computation_size_range=(1e11, 1e12),
                data_size_range=(1e5, 1e6)
            ),
            # 通信密集型
            EnhancedWorkflowConfig(
                workflow_size=20,
                ccr=100.0,
                dependency_probability=0.5,
                workflow_type="dag",
                computation_size_range=(1e8, 1e9),
                data_size_range=(1e9, 1e10)
            ),
            # 高度并行
            EnhancedWorkflowConfig(
                workflow_size=30,
                ccr=1.0,
                dependency_probability=0.1,
                workflow_type="fork_join",
                computation_size_range=(1e9, 1e11),
                data_size_range=(1e6, 1e9)
            ),
            # 高度串行
            EnhancedWorkflowConfig(
                workflow_size=30,
                ccr=1.0,
                dependency_probability=0.95,
                workflow_type="chain",
                computation_size_range=(1e9, 1e11),
                data_size_range=(1e6, 1e9)
            )
        ]
        
        configs.extend(extreme_configs)
        
        logger.info(f"Generated {len(configs)} workflow configurations")
        return configs
    
    def generate_enhanced_knowledge_base(self, num_cases: int = 5000) -> WRENCHRAGKnowledgeBase:
        """生成增强的知识库"""
        logger.info(f"Generating enhanced knowledge base with {num_cases} cases...")
        
        # 创建知识库
        kb = WRENCHRAGKnowledgeBase(embedding_dim=64)
        
        # 生成平台配置
        platform_configs = self._generate_platform_configs()
        
        # 生成案例
        for i in range(num_cases):
            if i % 500 == 0:
                logger.info(f"Generated {i}/{num_cases} cases...")
            
            # 随机选择工作流配置
            config = random.choice(self.workflow_configs)
            
            # 生成工作流
            workflow = self._generate_workflow(config)
            
            # 随机选择平台配置
            platform_config = random.choice(platform_configs)
            
            # 随机选择调度器类型
            scheduler_type = random.choice(self.scheduler_types)
            
            # 生成调度案例
            case = self._generate_scheduling_case(
                workflow, platform_config, scheduler_type, config
            )
            
            # 添加到知识库
            kb.add_case(case)
        
        logger.info(f"Successfully generated knowledge base with {len(kb.cases)} cases")
        return kb
    
    def _generate_workflow(self, config):
        """根据配置生成工作流"""
        # 根据工作流类型选择生成模式
        workflow_type_to_pattern = {
            "chain": "montage",
            "tree": "ligo",
            "dag": "cybershake",
            "fork_join": "ligo",
            "pipeline": "montage",
            "hybrid": "cybershake"
        }
        
        pattern = workflow_type_to_pattern.get(config.workflow_type, "montage")
        
        # 使用WorkflowGenerator生成工作流
        # 设置随机种子以确保可重现性
        random_seed = random.randint(1000, 9999)
        original_state = random.getstate()
        random.seed(random_seed)
        
        try:
            # 创建临时WorkflowGenerator实例
            temp_generator = WorkflowGenerator(ccr=config.ccr)
            
            # 获取对应的模式函数
            pattern_func = temp_generator.patterns[pattern]
            
            # 生成工作流
            workflow = pattern_func(config.workflow_size)
            
            return workflow
        finally:
            # 恢复随机状态
            random.setstate(original_state)
    
    def _generate_task_assignments(self, workflow, platform_config):
        """生成任务分配"""
        assignments = {}
        available_nodes = list(platform_config['nodes'].keys())
        
        for task in workflow.tasks:
            # 随机分配任务到节点
            assigned_node = random.choice(available_nodes)
            assignments[task.id] = assigned_node
        
        return assignments
    
    def _estimate_communication_time(self, source_task, dest_task, platform_config):
        """估算任务间的通信时间"""
        # 计算源任务输出文件总大小
        output_size = sum(f.size for f in source_task.output_files)
        
        # 获取平台配置中节点的平均带宽
        avg_bandwidth = np.mean([
            node['bandwidth'] for node in platform_config['nodes'].values()
        ])
        
        # 通信时间 = 数据量 / 带宽
        return output_size / avg_bandwidth
    
    def _topological_sort(self, workflow):
        """对工作流任务进行拓扑排序"""
        # 构建任务到索引的映射
        task_to_idx = {task: i for i, task in enumerate(workflow.tasks)}
        
        # 计算入度
        in_degree = [len(task.dependencies) for task in workflow.tasks]
        
        # 初始化队列
        queue = [i for i, degree in enumerate(in_degree) if degree == 0]
        sorted_tasks = []
        
        while queue:
            current_idx = queue.pop(0)
            current_task = workflow.tasks[current_idx]
            sorted_tasks.append(current_task)
            
            # 更新后继任务的入度
            for task in workflow.tasks:
                if current_task.id in task.dependencies:
                    successor_idx = task_to_idx[task]
                    in_degree[successor_idx] -= 1
                    if in_degree[successor_idx] == 0:
                        queue.append(successor_idx)
        
        return sorted_tasks
    
    def _estimate_communication_time(self, source_task, dest_task, platform_config):
        """估计通信时间"""
        # 简化实现：基于平均文件大小和带宽
        if not source_task.output_files or not dest_task.input_files:
            return 0.0
        
        # 计算需要传输的数据量
        output_size = sum(f.size for f in source_task.output_files)
        
        # 获取平均带宽
        avg_bandwidth = np.mean([
            config['bandwidth'] for config in platform_config['nodes'].values()
        ])
        
        return output_size / avg_bandwidth
    
    def _calculate_actual_makespan(self, workflow, platform_config, task_assignments):
        """计算实际执行时间"""
        # 简化实现：基于任务分配和节点速度
        node_completion_times = {node: 0 for node in platform_config['nodes']}
        
        # 按照依赖关系排序任务
        sorted_tasks = self._topological_sort(workflow)
        
        for task in sorted_tasks:
            assigned_node = task_assignments[task.id]
            
            # 计算任务开始时间（考虑依赖任务的完成时间）
            start_time = node_completion_times[assigned_node]
            
            # 考虑依赖任务
            for dep_task_id in task.dependencies:
                dep_task = next((t for t in workflow.tasks if t.id == dep_task_id), None)
                if dep_task:
                    dep_node = task_assignments[dep_task_id]
                    dep_completion = node_completion_times[dep_node]
                    
                    # 如果依赖任务在不同节点，考虑通信时间
                    if dep_node != assigned_node:
                        comm_time = self._estimate_communication_time(dep_task, task, platform_config)
                        start_time = max(start_time, dep_completion + comm_time)
                    else:
                        start_time = max(start_time, dep_completion)
            
            # 计算任务完成时间
            execution_time = task.flops / platform_config['nodes'][assigned_node]['speed']
            node_completion_times[assigned_node] = start_time + execution_time
        
        return max(node_completion_times.values())
    
    def _topological_sort(self, workflow):
        """对工作流任务进行拓扑排序"""
        # 构建任务到索引的映射
        task_to_idx = {task: i for i, task in enumerate(workflow.tasks)}
        
        # 计算入度
        in_degree = [len(task.dependencies) for task in workflow.tasks]
        
        # 初始化队列
        queue = [i for i, degree in enumerate(in_degree) if degree == 0]
        sorted_tasks = []
        
        while queue:
            current_idx = queue.pop(0)
            current_task = workflow.tasks[current_idx]
            sorted_tasks.append(current_task)
            
            # 更新后继任务的入度
            for task in workflow.tasks:
                if current_task.id in task.dependencies:
                    successor_idx = task_to_idx[task]
                    in_degree[successor_idx] -= 1
                    if in_degree[successor_idx] == 0:
                        queue.append(successor_idx)
        
        return sorted_tasks
    
    def _estimate_communication_time(self, source_task, dest_task, platform_config):
        """估计通信时间"""
        # 简化实现：基于平均文件大小和带宽
        if not source_task.output_files or not dest_task.input_files:
            return 0.0
        
        # 计算需要传输的数据量
        output_size = sum(f.size for f in source_task.output_files)
        
        # 获取平均带宽
        avg_bandwidth = np.mean([
            config['bandwidth'] for config in platform_config['nodes'].values()
        ])
        
        return output_size / avg_bandwidth
    
    
    
    def _generate_task_assignments(self, workflow, platform_config):
        """生成任务分配"""
        assignments = {}
        available_nodes = list(platform_config['nodes'].keys())
        
        for task in workflow.tasks:
            # 随机分配任务到节点
            assigned_node = random.choice(available_nodes)
            assignments[task.id] = assigned_node
        
        return assignments
    
    def _calculate_actual_makespan(self, workflow, platform_config, task_assignments):
        """计算实际执行时间"""
        # 简化实现：基于任务分配和节点速度
        node_completion_times = {node: 0 for node in platform_config['nodes']}
        
        # 按照依赖关系排序任务
        sorted_tasks = self._topological_sort(workflow)
        
        for task in sorted_tasks:
            assigned_node = task_assignments[task.id]
            
            # 计算任务开始时间（考虑依赖任务的完成时间）
            start_time = node_completion_times[assigned_node]
            
            # 考虑依赖任务
            for dep_task_id in task.dependencies:
                dep_task = next((t for t in workflow.tasks if t.id == dep_task_id), None)
                if dep_task:
                    dep_node = task_assignments[dep_task_id]
                    dep_completion = node_completion_times[dep_node]
                    
                    # 如果依赖任务在不同节点，考虑通信时间
                    if dep_node != assigned_node:
                        comm_time = self._estimate_communication_time(dep_task, task, platform_config)
                        start_time = max(start_time, dep_completion + comm_time)
                    else:
                        start_time = max(start_time, dep_completion)
            
            # 计算任务完成时间
            execution_time = task.flops / platform_config['nodes'][assigned_node]['speed']
            node_completion_times[assigned_node] = start_time + execution_time
        
        return max(node_completion_times.values())
    
    
    
    def _calculate_scheduling_quality(self, workflow, platform_config, task_assignments, actual_makespan):
        """计算调度质量指标"""
        # 计算负载均衡度
        node_loads = {node: 0 for node in platform_config['nodes']}
        
        for task in workflow.tasks:
            assigned_node = task_assignments[task.id]
            node_loads[assigned_node] += task.flops
        
        # 计算负载均衡度（标准差与均值的比值）
        load_values = list(node_loads.values())
        load_balance = 1.0 - (np.std(load_values) / (np.mean(load_values) + 1e-6))
        
        # 计算资源利用率
        total_capacity = sum(
            config['speed'] * actual_makespan 
            for config in platform_config['nodes'].values()
        )
        total_work = sum(task.flops for task in workflow.tasks)
        resource_utilization = total_work / total_capacity
        
        # 综合质量分数
        quality = 0.5 * load_balance + 0.5 * resource_utilization
        
        return {
            'load_balance': load_balance,
            'resource_utilization': resource_utilization,
            'overall_quality': quality
        }
    
    def _compute_workflow_features(self, workflow):
        """计算工作流特征"""
        return {
            'task_count': len(workflow.tasks),
            'dependency_ratio': self._calculate_dependency_ratio(workflow),
            'critical_path_length': self._calculate_critical_path_length(workflow),
            'avg_computation': np.mean([task.flops for task in workflow.tasks]),
            'max_computation': max(task.flops for task in workflow.tasks),
            'min_computation': min(task.flops for task in workflow.tasks)
        }
    
    def _compute_platform_features(self, platform_config):
        """计算平台特征"""
        nodes = platform_config['nodes']
        return {
            'node_count': len(nodes),
            'total_speed': sum(config['speed'] for config in nodes.values()),
            'avg_speed': np.mean([config['speed'] for config in nodes.values()]),
            'total_bandwidth': sum(config['bandwidth'] for config in nodes.values()),
            'avg_bandwidth': np.mean([config['bandwidth'] for config in nodes.values()])
        }
    
    def _get_scheduler_params(self, scheduler_type):
        """获取调度器参数"""
        params = {
            'heuristic': {
                'algorithm': random.choice(['min_min', 'max_min', 'sufferage']),
                'look_ahead': random.randint(1, 5)
            },
            'list_scheduling': {
                'priority_type': random.choice(['level', 'rank', 'critical_path']),
                'dynamic_priority': random.choice([True, False])
            },
            'cluster': {
                'cluster_size': random.randint(2, 8),
                'merge_threshold': random.uniform(0.5, 0.9)
            },
            'genetic': {
                'population_size': random.randint(50, 200),
                'generations': random.randint(100, 500),
                'mutation_rate': random.uniform(0.01, 0.1)
            }
        }
        
        return params.get(scheduler_type, {})
    
    def _calculate_actual_makespan(self, workflow, platform_config, task_assignments):
        """计算实际执行时间"""
        # 简化实现：基于任务分配和节点速度计算
        node_completion_times = {node: 0.0 for node in platform_config['nodes']}
        
        # 按拓扑顺序处理任务
        sorted_tasks = self._topological_sort(workflow)
        
        for task in sorted_tasks:
            assigned_node = task_assignments[task.id]
            
            # 计算任务开始时间（考虑前置任务的完成时间和通信时间）
            start_time = 0.0
            for dep_id in task.dependencies:
                dep_task = next(t for t in workflow.tasks if t.id == dep_id)
                dep_node = task_assignments[dep_id]
                
                # 如果前置任务分配到不同节点，需要考虑通信时间
                if dep_node != assigned_node:
                    simple_platform_config = {
                        'nodes': {
                            'node_0': {'bandwidth': 10.0}  # 使用默认带宽
                        }
                    }
                    comm_time = self._estimate_communication_time(dep_task, task, simple_platform_config)
                else:
                    comm_time = 0.0
                
                # 前置任务的完成时间
                dep_completion_time = node_completion_times[dep_node]
                start_time = max(start_time, dep_completion_time + comm_time)
            
            # 计算任务完成时间
            execution_time = task.flops / platform_config['nodes'][assigned_node]['speed']
            node_completion_times[assigned_node] = start_time + execution_time
        
        return max(node_completion_times.values())
    
    def _generate_task_assignments(self, workflow, platform_config):
        """生成任务分配"""
        # 简化实现：随机分配任务到节点
        nodes = list(platform_config['nodes'].keys())
        task_assignments = {}
        
        for task in workflow.tasks:
            # 随机选择一个节点
            assigned_node = random.choice(nodes)
            task_assignments[task.id] = assigned_node
        
        return task_assignments
    
    def _topological_sort(self, workflow):
        """对工作流任务进行拓扑排序"""
        # 构建依赖图
        graph = {task.id: set(task.dependencies) for task in workflow.tasks}
        
        # 计算入度
        in_degree = {task_id: 0 for task_id in graph}
        for task_id in graph:
            for dep_id in graph[task_id]:
                in_degree[task_id] += 1
        
        # 初始化队列（入度为0的任务）
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks = []
        
        # 拓扑排序
        while queue:
            task_id = queue.pop(0)
            task = next(t for t in workflow.tasks if t.id == task_id)
            sorted_tasks.append(task)
            
            # 更新后继任务的入度
            for other_task in workflow.tasks:
                if task_id in other_task.dependencies:
                    in_degree[other_task.id] -= 1
                    if in_degree[other_task.id] == 0:
                        queue.append(other_task.id)
        
        return sorted_tasks
    
    def _calculate_task_criticality(self, task, workflow) -> float:
        """计算任务关键性分数"""
        if not workflow.tasks:
            return 0.0
        
        # 构建任务到索引的映射
        task_to_idx = {task.id: idx for idx, task in enumerate(workflow.tasks)}
        
        # 计算任务的关键性（基于计算量和依赖关系）
        task_idx = task_to_idx[task.id]
        
        # 计算任务的入度和出度
        in_degree = len(task.dependencies)
        out_degree = sum(1 for t in workflow.tasks if task.id in t.dependencies)
        
        # 计算任务的关键性（入度+出度+相对计算量）
        max_flops = max(t.flops for t in workflow.tasks)
        relative_computation = task.flops / max_flops if max_flops > 0 else 0.0
        
        criticality = (in_degree + out_degree) / (2 * len(workflow.tasks)) + 0.5 * relative_computation
        
        return min(1.0, criticality)  # 限制在0-1范围内
    
    def _calculate_data_locality_score(self, task, workflow) -> float:
        """计算任务的数据局部性分数"""
        if not workflow.tasks or not task.input_files:
            return 0.0
        
        # 计算输入文件的总大小
        total_input_size = sum(task.input_files.values()) if isinstance(task.input_files, dict) else len(task.input_files)
        
        # 计算与前置任务的数据局部性
        locality_score = 0.0
        for dep_id in task.dependencies:
            dep_task = next((t for t in workflow.tasks if t.id == dep_id), None)
            if dep_task and hasattr(dep_task, 'output_files'):
                # 计算重叠文件比例
                if isinstance(dep_task.output_files, dict) and isinstance(task.input_files, dict):
                    common_files = set(dep_task.output_files.keys()) & set(task.input_files.keys())
                    if common_files:
                        locality_score += len(common_files) / len(task.input_files)
        
        return min(1.0, locality_score)  # 限制在0-1范围内
    
    def _is_critical_task(self, task, workflow) -> bool:
        """判断任务是否在关键路径上"""
        if not workflow.tasks:
            return False
        
        # 构建任务到索引的映射
        task_to_idx = {task.id: idx for idx, task in enumerate(workflow.tasks)}
        
        # 计算每个任务的最早开始时间和最晚开始时间
        est = [0.0] * len(workflow.tasks)
        lst = [0.0] * len(workflow.tasks)
        
        # 计算最早开始时间
        sorted_tasks = self._topological_sort(workflow)
        for t in sorted_tasks:
            t_idx = task_to_idx[t.id]
            max_pred_finish = 0.0
            
            for dep_id in t.dependencies:
                dep_idx = task_to_idx[dep_id]
                max_pred_finish = max(max_pred_finish, est[dep_idx] + workflow.tasks[dep_idx].flops)
            
            est[t_idx] = max_pred_finish
        
        # 计算最晚开始时间（反向拓扑排序）
        # 首先计算整个工作流的总执行时间
        total_time = max(est[i] + workflow.tasks[i].flops for i in range(len(workflow.tasks)))
        
        # 初始化最晚开始时间
        for i in range(len(workflow.tasks)):
            lst[i] = total_time - workflow.tasks[i].flops
        
        # 反向处理任务
        for t in reversed(sorted_tasks):
            t_idx = task_to_idx[t.id]
            
            # 更新所有前置任务的最晚开始时间
            for dep_id in t.dependencies:
                dep_idx = task_to_idx[dep_id]
                lst[dep_idx] = min(lst[dep_idx], lst[t_idx] - workflow.tasks[dep_idx].flops)
        
        # 如果任务的最早开始时间等于最晚开始时间，则在关键路径上
        task_idx = task_to_idx[task.id]
        return abs(est[task_idx] - lst[task_idx]) < 1e-6
    
    def _topological_sort(self, workflow) -> List:
        """工作流拓扑排序"""
        if not workflow.tasks:
            return []
        
        # 构建任务到索引的映射
        task_to_idx = {task.id: idx for idx, task in enumerate(workflow.tasks)}
        
        # 计算入度
        in_degree = [0] * len(workflow.tasks)
        for task in workflow.tasks:
            for dep_id in task.dependencies:
                if dep_id in task_to_idx:
                    in_degree[task_to_idx[dep_id]] += 1
        
        # 初始化队列
        queue = []
        for i, degree in enumerate(in_degree):
            if degree == 0:
                queue.append(workflow.tasks[i])
        
        # 拓扑排序
        result = []
        while queue:
            task = queue.pop(0)
            result.append(task)
            
            # 减少所有依赖当前任务的任务的入度
            for other_task in workflow.tasks:
                if task.id in other_task.dependencies:
                    other_idx = task_to_idx[other_task.id]
                    in_degree[other_idx] -= 1
                    if in_degree[other_idx] == 0:
                        queue.append(other_task)
        
        return result
    
    def _estimate_workflow_makespan(self, workflow, platform_config) -> float:
        """估算工作流makespan（简化实现）"""
        # 简化实现：使用总计算量除以总处理能力
        total_computation = sum(task.flops for task in workflow.tasks)
        total_speed = sum(node['speed'] for node in platform_config['nodes'].values())
        return total_computation / total_speed
    
    def _generate_scheduling_case(self, workflow, platform_config, scheduler_type, config):
        """生成调度案例"""
        # 生成任务分配
        task_assignments = self._generate_task_assignments(workflow, platform_config)
        
        # 计算实际执行时间
        actual_makespan = self._calculate_actual_makespan(workflow, platform_config, task_assignments)
        
        # 估计执行时间
        estimated_makespan = self._estimate_workflow_makespan(workflow, platform_config)
        
        # 计算调度质量指标
        quality_metrics = self._calculate_scheduling_quality(
            workflow, platform_config, task_assignments, actual_makespan
        )
        
        # 计算工作流特征
        workflow_features = self._compute_workflow_features(workflow)
        
        # 计算平台特征
        platform_features = self._compute_platform_features(platform_config)
        
        # 获取调度器参数
        scheduler_params = self._get_scheduler_params(scheduler_type)
        
        # 创建调度案例
        # 从工作流中选择一个任务作为示例任务
        example_task = workflow.tasks[0] if workflow.tasks else None
        
        if example_task:
            # 计算任务特征
            task_flops = example_task.flops
            task_input_files = len(example_task.input_files) if hasattr(example_task, 'input_files') else 0
            task_output_files = len(example_task.output_files) if hasattr(example_task, 'output_files') else 0
            task_dependencies = len(example_task.dependencies)
            
            # 计算任务子节点数量
            task_children = sum(1 for task in workflow.tasks if example_task.id in task.dependencies)
            
            # 计算工作流特征
            task_count = len(workflow.tasks)
            dependency_ratio = sum(len(task.dependencies) for task in workflow.tasks) / max(task_count, 1)
            critical_path_length = int(self._calculate_critical_path_length(workflow))
            
            # 计算节点特征
            available_nodes = list(platform_config['nodes'].keys())
            node_capacities = {node: node_config['speed'] * node_config['cores'] 
                              for node, node_config in platform_config['nodes'].items()}
            node_loads = {node: 0.0 for node in available_nodes}  # 初始负载为0
            
            # 随机选择一个节点作为示例
            chosen_node = random.choice(available_nodes)
            
            # 计算任务执行时间（简化计算）
            task_execution_time = task_flops / node_capacities[chosen_node]
            
            # 计算工作流makespan
            workflow_makespan = actual_makespan
            
            # 计算节点利用率（简化计算）
            node_utilization = {node: random.uniform(0.1, 0.9) for node in available_nodes}
            
            # 计算仿真时间
            simulation_time = workflow_makespan * random.uniform(1.0, 1.2)
            
            # 随机选择一个动作
            action_taken = available_nodes.index(chosen_node)
            
            # 计算任务等待时间（简化计算）
            task_wait_time = random.uniform(0.1, 1.0)
            
            case = WRENCHKnowledgeCase(
                workflow_id=workflow.name,
                task_count=task_count,
                dependency_ratio=dependency_ratio,
                critical_path_length=critical_path_length,
                workflow_embedding=self._compute_workflow_embedding(workflow),
                task_id=example_task.id,
                task_flops=task_flops,
                task_input_files=task_input_files,
                task_output_files=task_output_files,
                task_dependencies=task_dependencies,
                task_children=task_children,
                task_features=np.array(self._compute_task_features(example_task, workflow)),
                available_nodes=available_nodes,
                node_capacities=node_capacities,
                node_loads=node_loads,
                node_features=self._compute_node_features(
                    platform_config, 
                    {node: 0 for node in platform_config['nodes']}
                ),
                scheduler_type=scheduler_type,
                chosen_node=chosen_node,
                action_taken=action_taken,
                task_execution_time=task_execution_time,
                task_wait_time=task_wait_time,
                workflow_makespan=workflow_makespan,
                node_utilization=node_utilization,
                simulation_time=simulation_time,
                platform_config=platform_config['name'],
                metadata={
                    'workflow_config': asdict(config),
                    'platform_config': platform_config,
                    'task_assignments': task_assignments,
                    'quality_metrics': quality_metrics,
                    'workflow_features': workflow_features,
                    'platform_features': platform_features,
                    'estimated_makespan': estimated_makespan,
                    'scheduler_params': scheduler_params
                }
            )
        else:
            # 如果没有任务，创建一个空案例
            case = WRENCHKnowledgeCase(
                workflow_id=workflow.name,
                task_count=0,
                dependency_ratio=0.0,
                critical_path_length=0,
                workflow_embedding=np.zeros(64),  # 默认嵌入向量
                task_id="",
                task_flops=0.0,
                task_input_files=0,
                task_output_files=0,
                task_dependencies=0,
                task_children=0,
                task_features=np.zeros(10),  # 默认任务特征
                available_nodes=[],
                node_capacities={},
                node_loads={},
                node_features=np.zeros(5),  # 默认节点特征
                scheduler_type=scheduler_type,
                chosen_node="",
                action_taken=0,
                task_execution_time=0.0,
                task_wait_time=0.0,
                workflow_makespan=0.0,
                node_utilization={},
                simulation_time=0.0,
                platform_config=platform_config['name'],
                metadata={
                    'workflow_config': asdict(config),
                    'platform_config': platform_config,
                    'task_assignments': {},
                    'quality_metrics': quality_metrics,
                    'workflow_features': workflow_features,
                    'platform_features': platform_features,
                    'estimated_makespan': estimated_makespan,
                    'scheduler_params': scheduler_params
                }
            )
        
        return case
    
    def _generate_platform_configs(self) -> List[Dict]:
        """生成平台配置列表"""
        logger.info("Generating platform configurations...")
        
        # 使用PlatformGenerator生成标准配置
        configs = []
        
        # 小型平台配置
        small_config = {
            'name': 'small',
            'nodes': {
                f'node_{i}': {
                    'speed': random.uniform(1.0, 2.0),
                    'cores': random.randint(2, 4),
                    'memory': random.uniform(4, 8),
                    'disk_speed': random.uniform(100, 200),  # 添加磁盘速度
                    'bandwidth': random.uniform(10, 20),
                    'latency': random.uniform(0.1, 0.5)
                }
                for i in range(16)
            }
        }
        configs.append(small_config)
        
        # 中型平台配置
        medium_config = {
            'name': 'medium',
            'nodes': {
                f'node_{i}': {
                    'speed': random.uniform(2.0, 4.0),
                    'cores': random.randint(4, 8),
                    'memory': random.uniform(8, 16),
                    'disk_speed': random.uniform(200, 400),  # 添加磁盘速度
                    'bandwidth': random.uniform(20, 40),
                    'latency': random.uniform(0.05, 0.2)
                }
                for i in range(64)
            }
        }
        configs.append(medium_config)
        
        # 大型平台配置
        large_config = {
            'name': 'large',
            'nodes': {
                f'node_{i}': {
                    'speed': random.uniform(4.0, 8.0),
                    'cores': random.randint(8, 16),
                    'memory': random.uniform(16, 32),
                    'disk_speed': random.uniform(400, 800),  # 添加磁盘速度
                    'bandwidth': random.uniform(40, 80),
                    'latency': random.uniform(0.01, 0.1)
                }
                for i in range(128)
            }
        }
        configs.append(large_config)
        
        # 超大型平台配置
        xlarge_config = {
            'name': 'xlarge',
            'nodes': {
                f'node_{i}': {
                    'speed': random.uniform(8.0, 16.0),
                    'cores': random.randint(16, 32),
                    'memory': random.uniform(32, 64),
                    'disk_speed': random.uniform(800, 1600),  # 添加磁盘速度
                    'bandwidth': random.uniform(80, 160),
                    'latency': random.uniform(0.005, 0.05)
                }
                for i in range(256)
            }
        }
        configs.append(xlarge_config)
        
        # 为每个配置添加总计算能力
        for config in configs:
            config['total_power'] = sum(
                node['speed'] * node['cores'] 
                for node in config['nodes'].values()
            )
            config['efficiency'] = random.uniform(0.7, 0.95)
        
        logger.info(f"Generated {len(configs)} platform configurations")
        return configs
        
    
    def _calculate_critical_path_length(self, workflow) -> float:
        """计算工作流关键路径长度"""
        if not workflow.tasks:
            return 0.0
        
        # 构建任务到索引的映射
        task_to_idx = {task.id: idx for idx, task in enumerate(workflow.tasks)}
        
        # 初始化每个任务的最早完成时间
        eft = [0.0] * len(workflow.tasks)
        
        # 按拓扑顺序处理任务
        sorted_tasks = self._topological_sort(workflow)
        
        for task in sorted_tasks:
            task_idx = task_to_idx[task.id]
            max_pred_finish = 0.0
            
            # 找到所有前置任务的最大完成时间
            for dep_id in task.dependencies:
                dep_idx = task_to_idx[dep_id]
                # 创建一个简化的平台配置用于计算通信时间
                simple_platform_config = {
                    'nodes': {
                        'node_0': {'bandwidth': 10.0}  # 使用默认带宽
                    }
                }
                comm_time = self._estimate_communication_time(workflow.tasks[dep_idx], task, simple_platform_config)
                max_pred_finish = max(max_pred_finish, eft[dep_idx] + comm_time)
            
            # 计算当前任务的最早完成时间
            # 假设使用平均计算能力
            avg_speed = 1.0  # 默认值
            eft[task_idx] = max_pred_finish + (task.flops / avg_speed)
        
        # 返回所有任务中最大的最早完成时间，即关键路径长度
        return max(eft) if eft else 0.0
    
    def _calculate_task_parallelism(self, workflow) -> float:
        """计算工作流任务并行度"""
        if not workflow.tasks:
            return 0.0
        
        # 构建任务到索引的映射
        task_to_idx = {task.id: idx for idx, task in enumerate(workflow.tasks)}
        
        # 初始化每个任务的最早开始时间和最晚开始时间
        est = [0.0] * len(workflow.tasks)
        lst = [0.0] * len(workflow.tasks)
        
        # 计算最早开始时间
        sorted_tasks = self._topological_sort(workflow)
        for task in sorted_tasks:
            task_idx = task_to_idx[task.id]
            max_pred_finish = 0.0
            
            for dep_id in task.dependencies:
                dep_idx = task_to_idx[dep_id]
                # 简化计算，不考虑通信时间
                max_pred_finish = max(max_pred_finish, est[dep_idx] + workflow.tasks[dep_idx].flops)
            
            est[task_idx] = max_pred_finish
        
        # 计算最晚开始时间（反向拓扑排序）
        # 首先计算整个工作流的总执行时间
        total_time = max(est[i] + task.flops for i, task in enumerate(workflow.tasks))
        
        # 初始化最晚开始时间
        for i in range(len(workflow.tasks)):
            lst[i] = total_time - workflow.tasks[i].flops
        
        # 反向处理任务
        for task in reversed(sorted_tasks):
            task_idx = task_to_idx[task.id]
            
            # 更新所有前置任务的最晚开始时间
            for dep_id in task.dependencies:
                dep_idx = task_to_idx[dep_id]
                lst[dep_idx] = min(lst[dep_idx], lst[task_idx] - workflow.tasks[dep_idx].flops)
        
        # 计算每个任务的时间窗口（最晚开始时间 - 最早开始时间）
        time_windows = [lst[i] - est[i] for i in range(len(workflow.tasks))]
        
        # 计算平均并行度（时间窗口与总执行时间的比值）
        avg_parallelism = sum(time_windows) / total_time if total_time > 0 else 0.0
        
        return avg_parallelism
    
    def _calculate_workflow_complexity(self, workflow) -> float:
        """计算工作流复杂度"""
        if not workflow.tasks:
            return 0.0
        
        # 计算任务数量复杂度因子
        task_count_factor = min(len(workflow.tasks) / 10.0, 1.0)  # 归一化到0-1
        
        # 计算依赖关系复杂度因子
        total_dependencies = sum(len(task.dependencies) for task in workflow.tasks)
        max_possible_dependencies = len(workflow.tasks) * (len(workflow.tasks) - 1)
        dependency_factor = total_dependencies / max_possible_dependencies if max_possible_dependencies > 0 else 0.0
        
        # 计算计算量复杂度因子
        total_flops = sum(task.flops for task in workflow.tasks)
        # 假设最大计算量为1e12，可根据实际情况调整
        flops_factor = min(total_flops / 1e12, 1.0)
        
        # 计算数据传输复杂度因子
        total_data_size = sum(sum(task.input_files.values()) for task in workflow.tasks)
        # 假设最大数据量为1e9，可根据实际情况调整
        data_factor = min(total_data_size / 1e9, 1.0)
        
        # 综合复杂度（加权平均）
        complexity = 0.3 * task_count_factor + 0.3 * dependency_factor + 0.2 * flops_factor + 0.2 * data_factor
        
        return complexity
    
    def _calculate_platform_utilization(self, workflow, platform_config) -> float:
        """计算平台利用率"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算能力
        total_speed = sum(node.get('speed', 1.0) for node in platform_config['nodes'].values())
        if total_speed <= 0:
            return 0.0
        
        # 计算总计算量
        total_flops = sum(task.flops for task in workflow.tasks)
        
        # 计算总执行时间（简化计算，不考虑通信开销）
        total_time = total_flops / total_speed
        
        # 计算每个节点的总计算时间
        node_compute_times = {}
        for node_id, node_config in platform_config['nodes'].items():
            node_speed = node_config.get('speed', 1.0)
            # 简化假设：任务均匀分配到各个节点
            node_flops = total_flops / len(platform_config['nodes'])
            node_compute_times[node_id] = node_flops / node_speed
        
        # 计算平均节点利用率（计算时间/总时间）
        avg_utilization = sum(min(compute_time / total_time, 1.0) for compute_time in node_compute_times.values()) / len(node_compute_times)
        
        return avg_utilization
    
    def _calculate_load_balance(self, workflow, platform_config) -> float:
        """计算平台负载均衡度"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算量
        total_flops = sum(task.flops for task in workflow.tasks)
        
        # 计算每个节点的计算能力
        node_speeds = []
        for node_id, node_config in platform_config['nodes'].items():
            node_speed = node_config.get('speed', 1.0)
            node_speeds.append(node_speed)
        
        # 计算理想负载分配（按计算能力比例）
        total_speed = sum(node_speeds)
        if total_speed <= 0:
            return 0.0
        
        ideal_loads = [speed / total_speed * total_flops for speed in node_speeds]
        
        # 简化假设：实际负载分配（可以根据实际调度策略调整）
        # 这里假设任务均匀分配到各个节点
        actual_loads = [total_flops / len(node_speeds)] * len(node_speeds)
        
        # 计算负载均衡度（1 - 标准差/平均值）
        if len(actual_loads) > 0:
            avg_load = sum(actual_loads) / len(actual_loads)
            if avg_load > 0:
                variance = sum((load - avg_load) ** 2 for load in actual_loads) / len(actual_loads)
                std_dev = variance ** 0.5
                load_balance = 1.0 - (std_dev / avg_load)
                return max(0.0, min(1.0, load_balance))  # 限制在0-1范围内
        
        return 0.0
    
    def _calculate_communication_overhead(self, workflow, platform_config) -> float:
        """计算通信开销比例"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算时间
        total_flops = sum(task.flops for task in workflow.tasks)
        total_speed = sum(node.get('speed', 1.0) for node in platform_config['nodes'].values())
        if total_speed <= 0:
            return 0.0
        compute_time = total_flops / total_speed
        
        # 计算总通信时间
        total_comm_time = 0.0
        task_to_idx = {task.id: idx for idx, task in enumerate(workflow.tasks)}
        
        for task in workflow.tasks:
            for dep_id in task.dependencies:
                dep_idx = task_to_idx[dep_id]
                comm_time = self._estimate_communication_time(workflow.tasks[dep_idx], task, platform_config)
                total_comm_time += comm_time
        
        # 计算通信开销比例
        total_time = compute_time + total_comm_time
        if total_time > 0:
            return total_comm_time / total_time
        
        return 0.0
    
    def _calculate_energy_efficiency(self, workflow, platform_config) -> float:
        """计算能效比（计算量/能耗）"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算量
        total_flops = sum(task.flops for task in workflow.tasks)
        
        # 计算总能耗
        total_energy = 0.0
        for node_id, node_config in platform_config['nodes'].items():
            # 获取节点功率（瓦特）
            power = node_config.get('power', 100.0)  # 默认100W
            
            # 计算节点计算时间（简化假设：任务均匀分配）
            node_speed = node_config.get('speed', 1.0)
            if node_speed > 0:
                node_flops = total_flops / len(platform_config['nodes'])
                compute_time = node_flops / node_speed
                node_energy = power * compute_time  # 能量 = 功率 × 时间
                total_energy += node_energy
        
        # 计算能效比（计算量/能耗）
        if total_energy > 0:
            return total_flops / total_energy
        
        return 0.0
    
    def _calculate_resource_contention(self, workflow, platform_config) -> float:
        """计算资源争用度"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算每个节点的资源需求
        node_demands = {}
        for node_id in platform_config['nodes'].keys():
            node_demands[node_id] = 0.0
        
        # 计算总计算需求
        total_flops = sum(task.flops for task in workflow.tasks)
        
        # 简化假设：任务均匀分配到各个节点
        if len(platform_config['nodes']) > 0:
            flops_per_node = total_flops / len(platform_config['nodes'])
            
            # 计算每个节点的资源利用率
            node_utilizations = []
            for node_id, node_config in platform_config['nodes'].items():
                node_speed = node_config.get('speed', 1.0)
                if node_speed > 0:
                    # 计算节点利用率（需求/能力）
                    utilization = flops_per_node / node_speed
                    node_utilizations.append(utilization)
            
            # 计算资源争用度（基于利用率的标准差）
            if len(node_utilizations) > 0:
                avg_utilization = sum(node_utilizations) / len(node_utilizations)
                if avg_utilization > 0:
                    variance = sum((util - avg_utilization) ** 2 for util in node_utilizations) / len(node_utilizations)
                    std_dev = variance ** 0.5
                    # 资源争用度 = 标准差/平均值
                    contention = std_dev / avg_utilization
                    return min(1.0, contention)  # 限制在0-1范围内
        
        return 0.0
    
    def _calculate_qos_satisfaction(self, workflow, platform_config) -> float:
        """计算服务质量满意度"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算时间
        total_flops = sum(task.flops for task in workflow.tasks)
        total_speed = sum(node.get('speed', 1.0) for node in platform_config['nodes'].values())
        if total_speed <= 0:
            return 0.0
        compute_time = total_flops / total_speed
        
        # 计算总通信时间
        total_comm_time = 0.0
        task_to_idx = {task.id: idx for idx, task in enumerate(workflow.tasks)}
        
        for task in workflow.tasks:
            for dep_id in task.dependencies:
                dep_idx = task_to_idx[dep_id]
                comm_time = self._estimate_communication_time(workflow.tasks[dep_idx], task, platform_config)
                total_comm_time += comm_time
        
        # 总执行时间
        total_time = compute_time + total_comm_time
        
        # 假设QoS要求（可根据实际情况调整）
        qos_deadline = 1000.0  # 默认时间限制
        qos_energy_limit = 10000.0  # 默认能耗限制
        
        # 计算时间满意度
        time_satisfaction = 1.0
        if total_time > qos_deadline:
            time_satisfaction = qos_deadline / total_time
        
        # 计算能耗满意度
        energy_satisfaction = 1.0
        total_energy = 0.0
        for node_id, node_config in platform_config['nodes'].items():
            power = node_config.get('power', 100.0)
            node_speed = node_config.get('speed', 1.0)
            if node_speed > 0:
                node_flops = total_flops / len(platform_config['nodes'])
                compute_time = node_flops / node_speed
                node_energy = power * compute_time
                total_energy += node_energy
        
        if total_energy > qos_energy_limit:
            energy_satisfaction = qos_energy_limit / total_energy
        
        # 综合QoS满意度（加权平均）
        qos_satisfaction = 0.6 * time_satisfaction + 0.4 * energy_satisfaction
        
        return qos_satisfaction
    
    def _calculate_system_throughput(self, workflow, platform_config) -> float:
        """计算系统吞吐量（任务数/总时间）"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算时间
        total_flops = sum(task.flops for task in workflow.tasks)
        total_speed = sum(node.get('speed', 1.0) for node in platform_config['nodes'].values())
        if total_speed <= 0:
            return 0.0
        compute_time = total_flops / total_speed
        
        # 计算总通信时间
        total_comm_time = 0.0
        task_to_idx = {task.id: idx for idx, task in enumerate(workflow.tasks)}
        
        for task in workflow.tasks:
            for dep_id in task.dependencies:
                dep_idx = task_to_idx[dep_id]
                comm_time = self._estimate_communication_time(workflow.tasks[dep_idx], task, platform_config)
                total_comm_time += comm_time
        
        # 总执行时间
        total_time = compute_time + total_comm_time
        
        # 计算吞吐量（任务数/总时间）
        if total_time > 0:
            throughput = len(workflow.tasks) / total_time
            return throughput
        
        return 0.0
    
    def _calculate_fault_tolerance(self, workflow, platform_config) -> float:
        """计算系统容错能力"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算节点冗余度
        node_count = len(platform_config['nodes'])
        redundancy_factor = min(node_count / 3.0, 1.0)  # 3个节点以上认为有冗余
        
        # 计算任务关键性
        critical_tasks = 0
        for task in workflow.tasks:
            # 检查任务是否为关键任务（有多个依赖或被多个任务依赖）
            dependency_count = len(task.dependencies)
            successor_count = sum(1 for t in workflow.tasks if task.id in t.dependencies)
            
            if dependency_count > 2 or successor_count > 2:
                critical_tasks += 1
        
        criticality_ratio = critical_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算容错能力（冗余度和关键性的综合）
        fault_tolerance = (1.0 - criticality_ratio) * redundancy_factor
        
        return fault_tolerance
    
    def _calculate_scalability(self, workflow, platform_config) -> float:
        """计算系统可扩展性"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算当前节点数
        current_nodes = len(platform_config['nodes'])
        
        # 计算任务并行度
        parallelism = self._calculate_task_parallelism(workflow)
        
        # 计算通信开销比例
        comm_overhead = self._calculate_communication_overhead(workflow, platform_config)
        
        # 计算资源争用度
        resource_contention = self._calculate_resource_contention(workflow, platform_config)
        
        # 计算可扩展性因子
        # 1. 并行度越高，可扩展性越好
        parallelism_factor = min(parallelism / current_nodes, 1.0) if current_nodes > 0 else 0.0
        
        # 2. 通信开销越低，可扩展性越好
        comm_factor = 1.0 - comm_overhead
        
        # 3. 资源争用度越低，可扩展性越好
        contention_factor = 1.0 - resource_contention
        
        # 综合可扩展性（加权平均）
        scalability = 0.5 * parallelism_factor + 0.3 * comm_factor + 0.2 * contention_factor
        
        return max(0.0, min(1.0, scalability))  # 限制在0-1范围内
    
    def _calculate_cost_efficiency(self, workflow, platform_config) -> float:
        """计算成本效益比（计算量/成本）"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算量
        total_flops = sum(task.flops for task in workflow.tasks)
        
        # 计算总成本
        total_cost = 0.0
        for node_id, node_config in platform_config['nodes'].items():
            # 获取节点成本（每小时）
            cost_per_hour = node_config.get('cost', 1.0)  # 默认成本
            
            # 计算节点使用时间（简化假设：任务均匀分配）
            node_speed = node_config.get('speed', 1.0)
            if node_speed > 0:
                node_flops = total_flops / len(platform_config['nodes'])
                compute_time = node_flops / node_speed
                # 转换为小时（假设时间单位为秒）
                compute_hours = compute_time / 3600.0
                node_cost = cost_per_hour * compute_hours
                total_cost += node_cost
        
        # 计算成本效益比（计算量/成本）
        if total_cost > 0:
            return total_flops / total_cost
        
        return 0.0
    
    def _calculate_adaptability(self, workflow, platform_config) -> float:
        """计算系统适应性（处理动态变化的能力）"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算任务异构性（任务计算量的差异程度）
        task_flops = [task.flops for task in workflow.tasks]
        if task_flops:
            avg_flops = sum(task_flops) / len(task_flops)
            if avg_flops > 0:
                flops_variance = sum((flops - avg_flops) ** 2 for flops in task_flops) / len(task_flops)
                task_heterogeneity = (flops_variance ** 0.5) / avg_flops
            else:
                task_heterogeneity = 0.0
        else:
            task_heterogeneity = 0.0
        
        # 计算节点异构性（节点计算能力的差异程度）
        node_speeds = [node.get('speed', 1.0) for node in platform_config['nodes'].values()]
        if node_speeds:
            avg_speed = sum(node_speeds) / len(node_speeds)
            if avg_speed > 0:
                speed_variance = sum((speed - avg_speed) ** 2 for speed in node_speeds) / len(node_speeds)
                node_heterogeneity = (speed_variance ** 0.5) / avg_speed
            else:
                node_heterogeneity = 0.0
        else:
            node_heterogeneity = 0.0
        
        # 计算依赖复杂度
        dependency_ratio = self._calculate_dependency_ratio(workflow)
        
        # 计算适应性（基于异构性和复杂度的综合）
        # 1. 任务和节点异构性越高，适应性需求越大
        heterogeneity_factor = (task_heterogeneity + node_heterogeneity) / 2.0
        
        # 2. 依赖关系越复杂，适应性需求越大
        complexity_factor = dependency_ratio
        
        # 3. 节点数量越多，适应性越强
        node_count_factor = min(len(platform_config['nodes']) / 5.0, 1.0)  # 5个节点以上认为适应性强
        
        # 综合适应性
        adaptability = 0.4 * heterogeneity_factor + 0.3 * complexity_factor + 0.3 * node_count_factor
        
        return max(0.0, min(1.0, adaptability))  # 限制在0-1范围内
    
    def _calculate_reliability(self, workflow, platform_config) -> float:
        """计算系统可靠性"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算节点可靠性（基于节点数量和冗余度）
        node_count = len(platform_config['nodes'])
        node_reliability = min(node_count / 3.0, 1.0)  # 3个节点以上认为可靠性高
        
        # 计算任务关键路径可靠性
        critical_path_length = self._calculate_critical_path_length(workflow)
        max_path_length = sum(task.flops for task in workflow.tasks)  # 理论最大路径长度
        if max_path_length > 0:
            path_reliability = 1.0 - (critical_path_length / max_path_length)
        else:
            path_reliability = 1.0
        
        # 计算容错能力
        fault_tolerance = self._calculate_fault_tolerance(workflow, platform_config)
        
        # 计算负载均衡度
        load_balance = self._calculate_load_balance(workflow, platform_config)
        
        # 综合可靠性（加权平均）
        reliability = 0.3 * node_reliability + 0.3 * path_reliability + 0.2 * fault_tolerance + 0.2 * load_balance
        
        return max(0.0, min(1.0, reliability))  # 限制在0-1范围内
    
    def _calculate_security(self, workflow, platform_config) -> float:
        """计算系统安全性"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算数据传输安全性（基于通信量）
        total_data_size = 0
        for task in workflow.tasks:
            total_data_size += sum(task.input_files.values())
        
        # 数据量越大，安全风险越高
        data_risk = min(total_data_size / 1e9, 1.0)  # 假设1GB为基准
        
        # 计算节点安全性（基于节点数量和分布）
        node_count = len(platform_config['nodes'])
        # 节点越多，分布式安全性越高
        distribution_security = min(node_count / 5.0, 1.0)  # 5个节点以上认为安全性高
        
        # 计算任务隔离性（基于任务依赖关系）
        isolated_tasks = 0
        for task in workflow.tasks:
            if len(task.dependencies) == 0:  # 无依赖的任务
                isolated_tasks += 1
        
        isolation_factor = isolated_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算容错能力（与安全性相关）
        fault_tolerance = self._calculate_fault_tolerance(workflow, platform_config)
        
        # 综合安全性（加权平均）
        security = 0.3 * (1.0 - data_risk) + 0.3 * distribution_security + 0.2 * isolation_factor + 0.2 * fault_tolerance
        
        return max(0.0, min(1.0, security))  # 限制在0-1范围内
    
    def _calculate_fault_tolerance(self, workflow, platform_config) -> float:
        """计算系统容错能力"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算节点冗余度（节点数量越多，冗余度越高）
        node_count = len(platform_config['nodes'])
        node_redundancy = min(node_count / 5.0, 1.0)  # 5个节点以上认为冗余度高
        
        # 计算任务冗余度（有备份的任务比例）
        redundant_tasks = 0
        for task in workflow.tasks:
            # 检查任务是否有备份（简化版，假设有特定属性表示备份）
            if hasattr(task, 'has_backup') and task.has_backup:
                redundant_tasks += 1
        
        task_redundancy = redundant_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算依赖复杂度（依赖关系越简单，容错能力越强）
        total_dependencies = sum(len(task.dependencies) for task in workflow.tasks)
        max_possible_dependencies = len(workflow.tasks) * (len(workflow.tasks) - 1) / 2  # 完全图
        
        if max_possible_dependencies > 0:
            dependency_complexity = 1.0 - (total_dependencies / max_possible_dependencies)
        else:
            dependency_complexity = 1.0
        
        # 计算节点异构性（异构性越高，容错能力越强）
        node_speeds = [node.get('speed', 1.0) for node in platform_config['nodes']]
        if node_speeds:
            avg_speed = sum(node_speeds) / len(node_speeds)
            speed_variance = sum((speed - avg_speed) ** 2 for speed in node_speeds) / len(node_speeds)
            # 速度方差越大，异构性越高
            heterogeneity = min(speed_variance / (avg_speed ** 2), 1.0) if avg_speed > 0 else 0.0
        else:
            heterogeneity = 0.0
        
        # 综合容错能力（加权平均）
        fault_tolerance = 0.4 * node_redundancy + 0.3 * task_redundancy + 0.2 * dependency_complexity + 0.1 * heterogeneity
        
        return max(0.0, min(1.0, fault_tolerance))  # 限制在0-1范围内
    
    def _calculate_critical_path_length(self, workflow) -> float:
        """计算工作流关键路径长度"""
        if not workflow.tasks:
            return 0.0
        
        # 构建任务图
        task_graph = {}
        for task in workflow.tasks:
            task_graph[task.id] = {
                'task': task,
                'dependencies': task.dependencies,
                'flops': task.flops,
                'earliest_start': 0.0,
                'earliest_finish': 0.0
            }
        
        # 计算每个任务的最早开始和完成时间
        for task_id, task_info in task_graph.items():
            # 如果有依赖任务，则最早开始时间为所有依赖任务的最大完成时间
            if task_info['dependencies']:
                max_dep_finish = 0.0
                for dep_id in task_info['dependencies']:
                    if dep_id in task_graph:
                        max_dep_finish = max(max_dep_finish, task_graph[dep_id]['earliest_finish'])
                task_info['earliest_start'] = max_dep_finish
            
            # 最早完成时间 = 最早开始时间 + 任务计算量
            task_info['earliest_finish'] = task_info['earliest_start'] + task_info['flops']
        
        # 找出关键路径长度（所有任务最早完成时间的最大值）
        critical_path_length = max(task_info['earliest_finish'] for task_info in task_graph.values())
        
        return critical_path_length
    
    def _calculate_load_balance(self, workflow, platform_config) -> float:
        """计算系统负载均衡度"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算量
        total_flops = sum(task.flops for task in workflow.tasks)
        
        # 计算每个节点的理想负载
        node_count = len(platform_config['nodes'])
        if node_count == 0:
            return 0.0
        
        ideal_load_per_node = total_flops / node_count
        
        # 计算每个节点的实际负载（简化版，假设任务均匀分布）
        # 在实际应用中，这里应该考虑任务调度结果
        node_loads = []
        for i in range(node_count):
            # 简化计算：假设任务按顺序分配到节点
            node_load = 0.0
            for j, task in enumerate(workflow.tasks):
                if j % node_count == i:  # 简单的轮询分配
                    node_load += task.flops
            node_loads.append(node_load)
        
        # 计算负载均衡度（1 - 负载标准差/平均负载）
        if ideal_load_per_node > 0:
            # 计算负载标准差
            variance = sum((load - ideal_load_per_node) ** 2 for load in node_loads) / node_count
            std_dev = variance ** 0.5
            
            # 计算负载均衡度
            load_balance = 1.0 - (std_dev / ideal_load_per_node)
        else:
            load_balance = 1.0
        
        return max(0.0, min(1.0, load_balance))  # 限制在0-1范围内
    
    def _estimate_communication_time(self, task1, task2, platform_config) -> float:
        """估算两个任务之间的通信时间"""
        if not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 获取平台配置中的网络带宽
        network_bandwidth = platform_config.get('network_bandwidth', 1e9)  # 默认1Gbps
        
        # 计算需要传输的数据量（简化版，假设所有输出文件都需要传输）
        data_size = 0
        if hasattr(task1, 'output_files'):
            if isinstance(task1.output_files, dict):
                data_size += sum(task1.output_files.values())
            elif isinstance(task1.output_files, list):
                # 如果是列表，假设每个元素都有size属性
                data_size += sum(getattr(f, 'size', 0) for f in task1.output_files)
        
        # 如果任务2有输入文件需求，且与任务1的输出文件匹配，则计算通信时间
        if hasattr(task2, 'input_files'):
            if isinstance(task2.input_files, dict):
                for file_name, file_size in task2.input_files.items():
                    if hasattr(task1, 'output_files'):
                        if isinstance(task1.output_files, dict) and file_name in task1.output_files:
                            data_size += file_size
                        elif isinstance(task1.output_files, list):
                            # 如果是列表，查找匹配的文件
                            for f in task1.output_files:
                                if hasattr(f, 'name') and f.name == file_name:
                                    data_size += getattr(f, 'size', 0)
                                    break
            elif isinstance(task2.input_files, list):
                # 如果是列表，假设每个元素都有size属性
                for f in task2.input_files:
                    data_size += getattr(f, 'size', 0)
        
        # 计算通信时间（数据量/带宽）
        if network_bandwidth > 0:
            communication_time = data_size / network_bandwidth
        else:
            communication_time = 0.0
        
        return communication_time
    
    def _calculate_resource_utilization(self, workflow, platform_config) -> float:
        """计算系统资源利用率"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算资源
        total_compute_capacity = sum(node.get('speed', 1.0) for node in platform_config['nodes'])
        
        # 计算总计算需求
        total_compute_demand = sum(task.flops for task in workflow.tasks)
        
        # 计算计算资源利用率
        if total_compute_capacity > 0:
            compute_utilization = min(total_compute_demand / total_compute_capacity, 1.0)
        else:
            compute_utilization = 0.0
        
        # 计算内存资源利用率（简化版）
        total_memory_capacity = sum(node.get('memory', 1.0) for node in platform_config['nodes'])
        total_memory_demand = sum(getattr(task, 'memory', 1.0) for task in workflow.tasks)
        
        if total_memory_capacity > 0:
            memory_utilization = min(total_memory_demand / total_memory_capacity, 1.0)
        else:
            memory_utilization = 0.0
        
        # 计算存储资源利用率（简化版）
        total_storage_capacity = sum(node.get('storage', 1.0) for node in platform_config['nodes'])
        total_storage_demand = sum(sum(task.input_files.values()) + sum(getattr(task, 'output_files', {}).values()) for task in workflow.tasks)
        
        if total_storage_capacity > 0:
            storage_utilization = min(total_storage_demand / total_storage_capacity, 1.0)
        else:
            storage_utilization = 0.0
        
        # 综合资源利用率（加权平均）
        resource_utilization = 0.5 * compute_utilization + 0.3 * memory_utilization + 0.2 * storage_utilization
        
        return max(0.0, min(1.0, resource_utilization))  # 限制在0-1范围内
    
    def _calculate_energy_efficiency(self, workflow, platform_config) -> float:
        """计算系统能效比"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算总计算量
        total_flops = sum(task.flops for task in workflow.tasks)
        
        # 计算总能耗
        total_energy = 0.0
        for node in platform_config['nodes']:
            # 获取节点功耗（W）
            power = node.get('power', 100.0)  # 默认100W
            
            # 估算节点使用时间（简化版，假设均匀分配任务）
            node_flops = total_flops / len(platform_config['nodes'])
            node_speed = node.get('speed', 1.0)
            
            if node_speed > 0:
                usage_time = node_flops / node_speed
            else:
                usage_time = 0.0
            
            # 计算节点能耗（功耗 × 时间）
            node_energy = power * usage_time
            total_energy += node_energy
        
        # 计算能效比（计算量/能耗）
        if total_energy > 0:
            energy_efficiency = total_flops / total_energy
        else:
            energy_efficiency = 0.0
        
        # 归一化能效比（假设最大能效比为1e9 flops/W）
        max_efficiency = 1e9
        normalized_efficiency = min(energy_efficiency / max_efficiency, 1.0)
        
        return max(0.0, normalized_efficiency)  # 限制在0-1范围内
    
    def _calculate_response_time(self, workflow, platform_config) -> float:
        """计算系统响应时间"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算关键路径长度
        critical_path_length = self._calculate_critical_path_length(workflow)
        
        # 计算总计算时间
        total_compute_time = 0.0
        for task in workflow.tasks:
            # 找到最快的节点
            fastest_node_speed = max(node.get('speed', 1.0) for node in platform_config['nodes'])
            if fastest_node_speed > 0:
                task_compute_time = task.flops / fastest_node_speed
            else:
                task_compute_time = 0.0
            total_compute_time += task_compute_time
        
        # 计算总通信时间
        total_communication_time = 0.0
        for task in workflow.tasks:
            for dep_id in task.dependencies:
                # 找到依赖任务
                dep_task = None
                for t in workflow.tasks:
                    if t.id == dep_id:
                        dep_task = t
                        break
                
                if dep_task:
                    # 估算通信时间
                    comm_time = self._estimate_communication_time(dep_task, task, platform_config)
                    total_communication_time += comm_time
        
        # 计算平均响应时间（关键路径长度 + 平均计算时间 + 平均通信时间）
        avg_compute_time = total_compute_time / len(workflow.tasks) if workflow.tasks else 0.0
        avg_comm_time = total_communication_time / len(workflow.tasks) if workflow.tasks else 0.0
        
        response_time = critical_path_length + avg_compute_time + avg_comm_time
        
        # 归一化响应时间（假设最大响应时间为1000秒）
        max_response_time = 1000.0
        normalized_response_time = min(response_time / max_response_time, 1.0)
        
        # 返回1-归一化响应时间（值越大表示响应时间越好）
        return max(0.0, 1.0 - normalized_response_time)
    
    def _calculate_availability(self, workflow, platform_config) -> float:
        """计算系统可用性"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算节点可用性（基于节点数量和冗余度）
        node_count = len(platform_config['nodes'])
        node_availability = min(node_count / 3.0, 1.0)  # 3个节点以上认为可用性高
        
        # 计算任务关键性（关键任务越多，可用性要求越高）
        critical_tasks = 0
        for task in workflow.tasks:
            # 被多个任务依赖的任务认为是关键任务
            dependency_count = 0
            for other_task in workflow.tasks:
                if task.id in other_task.dependencies:
                    dependency_count += 1
            
            if dependency_count > 1:  # 被多个任务依赖
                critical_tasks += 1
        
        critical_ratio = critical_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算容错能力（与可用性相关）
        fault_tolerance = self._calculate_fault_tolerance(workflow, platform_config)
        
        # 计算系统可靠性（与可用性相关）
        reliability = self._calculate_reliability(workflow, platform_config)
        
        # 综合可用性（加权平均）
        availability = 0.3 * node_availability + 0.2 * (1.0 - critical_ratio) + 0.3 * fault_tolerance + 0.2 * reliability
        
        return max(0.0, min(1.0, availability))  # 限制在0-1范围内
    
    def _calculate_maintainability(self, workflow, platform_config) -> float:
        """计算系统可维护性"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算任务模块化程度（依赖关系越简单，可维护性越高）
        total_dependencies = sum(len(task.dependencies) for task in workflow.tasks)
        max_possible_dependencies = len(workflow.tasks) * (len(workflow.tasks) - 1)
        
        if max_possible_dependencies > 0:
            modularity = 1.0 - (total_dependencies / max_possible_dependencies)
        else:
            modularity = 1.0
        
        # 计算节点同构性（节点越相似，可维护性越高）
        node_speeds = [node.get('speed', 1.0) for node in platform_config['nodes']]
        if node_speeds:
            avg_speed = sum(node_speeds) / len(node_speeds)
            speed_variance = sum((speed - avg_speed) ** 2 for speed in node_speeds) / len(node_speeds)
            # 方差越小，同构性越高
            node_homogeneity = 1.0 - min(speed_variance / (avg_speed ** 2), 1.0) if avg_speed > 0 else 1.0
        else:
            node_homogeneity = 0.0
        
        # 计算系统复杂度（任务和节点数量越少，可维护性越高）
        task_complexity = 1.0 - min(len(workflow.tasks) / 100.0, 1.0)  # 假设100个任务为复杂度上限
        node_complexity = 1.0 - min(len(platform_config['nodes']) / 20.0, 1.0)  # 假设20个节点为复杂度上限
        
        # 计算文档化程度（简化版，假设有配置文件表示有文档）
        has_documentation = 1.0 if platform_config.get('documentation', False) else 0.0
        
        # 综合可维护性（加权平均）
        maintainability = 0.3 * modularity + 0.2 * node_homogeneity + 0.2 * task_complexity + 0.2 * node_complexity + 0.1 * has_documentation
        
        return max(0.0, min(1.0, maintainability))  # 限制在0-1范围内
    
    def _calculate_usability(self, workflow, platform_config) -> float:
        """计算系统易用性"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算任务自动化程度（任务配置越简单，易用性越高）
        # 简化版：假设任务有预定义配置表示自动化程度高
        automated_tasks = 0
        for task in workflow.tasks:
            # 假设有预定义参数的任务是自动化的
            if hasattr(task, 'preset_parameters') and task.preset_parameters:
                automated_tasks += 1
        
        automation_ratio = automated_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算配置复杂度（配置参数越少，易用性越高）
        config_complexity = 0.0
        for node in platform_config['nodes']:
            # 计算每个节点的配置参数数量
            param_count = len([key for key in node.keys() if key not in ['id', 'name']])
            config_complexity += param_count
        
        avg_config_complexity = config_complexity / len(platform_config['nodes']) if platform_config['nodes'] else 0.0
        config_simplicity = 1.0 - min(avg_config_complexity / 10.0, 1.0)  # 假设10个参数为复杂度上限
        
        # 计算监控友好度（有监控配置表示易用性高）
        has_monitoring = 1.0 if platform_config.get('monitoring', False) else 0.0
        
        # 计算用户界面友好度（简化版，假设有UI配置表示友好）
        has_ui = 1.0 if platform_config.get('user_interface', False) else 0.0
        
        # 计算文档完整性（有文档表示易用性高）
        has_documentation = 1.0 if platform_config.get('documentation', False) else 0.0
        
        # 综合易用性（加权平均）
        usability = 0.3 * automation_ratio + 0.2 * config_simplicity + 0.2 * has_monitoring + 0.15 * has_ui + 0.15 * has_documentation
        
        return max(0.0, min(1.0, usability))  # 限制在0-1范围内
    
    def _calculate_portability(self, workflow, platform_config) -> float:
        """计算系统可移植性"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算平台独立性（任务不依赖特定平台特性，可移植性高）
        platform_specific_tasks = 0
        for task in workflow.tasks:
            # 假设有平台特定属性的任务是平台特定的
            if hasattr(task, 'platform_specific') and task.platform_specific:
                platform_specific_tasks += 1
        
        platform_independence = 1.0 - (platform_specific_tasks / len(workflow.tasks)) if workflow.tasks else 1.0
        
        # 计算标准化程度（使用标准接口和协议的任务越多，可移植性越高）
        standard_tasks = 0
        for task in workflow.tasks:
            # 假设有标准接口的任务是标准化的
            if hasattr(task, 'standard_interface') and task.standard_interface:
                standard_tasks += 1
        
        standardization_ratio = standard_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算配置灵活性（配置越灵活，可移植性越高）
        flexible_configs = 0
        for node in platform_config['nodes']:
            # 假设有可配置参数的节点是灵活的
            if len([key for key in node.keys() if key not in ['id', 'name']]) > 0:
                flexible_configs += 1
        
        config_flexibility = flexible_configs / len(platform_config['nodes']) if platform_config['nodes'] else 0.0
        
        # 计算依赖隔离度（任务依赖越少，可移植性越高）
        total_dependencies = sum(len(task.dependencies) for task in workflow.tasks)
        max_possible_dependencies = len(workflow.tasks) * (len(workflow.tasks) - 1)
        
        if max_possible_dependencies > 0:
            dependency_isolation = 1.0 - (total_dependencies / max_possible_dependencies)
        else:
            dependency_isolation = 1.0
        
        # 计算文档完整性（有完整文档的系统更易移植）
        has_documentation = 1.0 if platform_config.get('documentation', False) else 0.0
        
        # 综合可移植性（加权平均）
        portability = 0.3 * platform_independence + 0.2 * standardization_ratio + 0.2 * config_flexibility + 0.2 * dependency_isolation + 0.1 * has_documentation
        
        return max(0.0, min(1.0, portability))  # 限制在0-1范围内
    
    def _calculate_interoperability(self, workflow, platform_config) -> float:
        """计算系统互操作性"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算接口兼容性（任务使用标准接口，互操作性高）
        compatible_tasks = 0
        for task in workflow.tasks:
            # 假设有兼容接口的任务是互操作的
            if hasattr(task, 'compatible_interface') and task.compatible_interface:
                compatible_tasks += 1
        
        interface_compatibility = compatible_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算数据格式标准化（使用标准数据格式的任务越多，互操作性越高）
        standard_format_tasks = 0
        for task in workflow.tasks:
            # 假设使用标准数据格式的任务是互操作的
            if hasattr(task, 'standard_data_format') and task.standard_data_format:
                standard_format_tasks += 1
        
        data_format_standardization = standard_format_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算协议兼容性（使用标准协议的节点越多，互操作性越高）
        protocol_compatible_nodes = 0
        for node in platform_config['nodes']:
            # 假设支持标准协议的节点是互操作的
            if node.get('standard_protocol', False):
                protocol_compatible_nodes += 1
        
        protocol_compatibility = protocol_compatible_nodes / len(platform_config['nodes']) if platform_config['nodes'] else 0.0
        
        # 计算API开放性（提供开放API的节点越多，互操作性越高）
        open_api_nodes = 0
        for node in platform_config['nodes']:
            # 假设提供开放API的节点是互操作的
            if node.get('open_api', False):
                open_api_nodes += 1
        
        api_openness = open_api_nodes / len(platform_config['nodes']) if platform_config['nodes'] else 0.0
        
        # 计算文档完整性（有完整文档的系统更易互操作）
        has_documentation = 1.0 if platform_config.get('documentation', False) else 0.0
        
        # 综合互操作性（加权平均）
        interoperability = 0.3 * interface_compatibility + 0.2 * data_format_standardization + 0.2 * protocol_compatibility + 0.2 * api_openness + 0.1 * has_documentation
        
        return max(0.0, min(1.0, interoperability))  # 限制在0-1范围内
    
    def _calculate_compliance(self, workflow, platform_config) -> float:
        """计算系统合规性"""
        if not workflow.tasks or not platform_config or 'nodes' not in platform_config:
            return 0.0
        
        # 计算安全合规性（符合安全标准的任务越多，合规性越高）
        security_compliant_tasks = 0
        for task in workflow.tasks:
            # 假设符合安全标准的任务是合规的
            if hasattr(task, 'security_compliant') and task.security_compliant:
                security_compliant_tasks += 1
        
        security_compliance = security_compliant_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算数据保护合规性（符合数据保护标准的任务越多，合规性越高）
        data_protection_compliant_tasks = 0
        for task in workflow.tasks:
            # 假设符合数据保护标准的任务是合规的
            if hasattr(task, 'data_protection_compliant') and task.data_protection_compliant:
                data_protection_compliant_tasks += 1
        
        data_protection_compliance = data_protection_compliant_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        # 计算隐私合规性（符合隐私标准的节点越多，合规性越高）
        privacy_compliant_nodes = 0
        for node in platform_config['nodes']:
            # 假设符合隐私标准的节点是合规的
            if node.get('privacy_compliant', False):
                privacy_compliant_nodes += 1
        
        privacy_compliance = privacy_compliant_nodes / len(platform_config['nodes']) if platform_config['nodes'] else 0.0
        
        # 计算审计合规性（支持审计的节点越多，合规性越高）
        audit_compliant_nodes = 0
        for node in platform_config['nodes']:
            # 假设支持审计的节点是合规的
            if node.get('audit_compliant', False):
                audit_compliant_nodes += 1
        
        audit_compliance = audit_compliant_nodes / len(platform_config['nodes']) if platform_config['nodes'] else 0.0
        
        # 计算文档完整性（有完整文档的系统更易合规）
        has_documentation = 1.0 if platform_config.get('documentation', False) else 0.0
        
        # 综合合规性（加权平均）
        compliance = 0.3 * security_compliance + 0.2 * data_protection_compliance + 0.2 * privacy_compliance + 0.2 * audit_compliance + 0.1 * has_documentation
        
        return max(0.0, min(1.0, compliance))  # 限制在0-1范围内
    
    def _calculate_dependency_ratio(self, workflow) -> float:
        """计算工作流依赖比例"""
        if not workflow.tasks:
            return 0.0
        
        total_dependencies = sum(len(task.dependencies) for task in workflow.tasks)
        max_possible_dependencies = len(workflow.tasks) * (len(workflow.tasks) - 1)
        
        return total_dependencies / max_possible_dependencies if max_possible_dependencies > 0 else 0.0
    
    def _compute_workflow_embedding(self, workflow) -> np.ndarray:
        """计算工作流嵌入向量"""
        # 简化实现：使用工作流的基本统计特征
        features = [
            len(workflow.tasks),  # 任务数量
            self._calculate_dependency_ratio(workflow),  # 依赖比例
            self._calculate_critical_path_length(workflow),  # 关键路径长度
            np.mean([task.flops for task in workflow.tasks]),  # 平均计算量
            np.std([task.flops for task in workflow.tasks]),  # 计算量标准差
            np.mean([len(task.dependencies) for task in workflow.tasks]),  # 平均入度
            np.mean([len(task.input_files) for task in workflow.tasks]),  # 平均输入文件数
        ]
        
        # 填充到64维
        embedding = np.zeros(64)
        embedding[:len(features)] = features
        
        return embedding
    
    def _compute_node_features(self, platform_config, node_loads) -> np.ndarray:
        """计算节点特征向量"""
        # 计算平台级别的统计特征，而不是每个节点的特征
        node_speeds = [node_config['speed'] for node_config in platform_config['nodes'].values()]
        node_cores = [node_config['cores'] for node_config in platform_config['nodes'].values()]
        node_disk_speeds = [node_config['disk_speed'] for node_config in platform_config['nodes'].values()]
        node_bandwidths = [node_config['bandwidth'] for node_config in platform_config['nodes'].values()]
        node_latencies = [node_config['latency'] for node_config in platform_config['nodes'].values()]
        load_values = [node_loads.get(node_name, 0.0) for node_name in platform_config['nodes'].keys()]
        
        # 计算统计特征
        features = [
            len(platform_config['nodes']),  # 节点数量
            np.mean(node_speeds),  # 平均速度
            np.std(node_speeds),  # 速度标准差
            np.max(node_speeds),  # 最大速度
            np.min(node_speeds),  # 最小速度
            np.mean(node_cores),  # 平均核心数
            np.std(node_cores),  # 核心数标准差
            np.mean(node_disk_speeds),  # 平均磁盘速度
            np.std(node_disk_speeds),  # 磁盘速度标准差
            np.mean(node_bandwidths),  # 平均带宽
            np.std(node_bandwidths),  # 带宽标准差
            np.mean(node_latencies),  # 平均延迟
            np.std(node_latencies),  # 延迟标准差
            np.mean(load_values),  # 平均负载
            np.std(load_values),  # 负载标准差
            sum(node_speeds[i] * node_cores[i] for i in range(len(node_speeds))),  # 总计算能力
            sum(node_bandwidths),  # 总带宽
        ]
        
        # 填充到64维
        embedding = np.zeros(64)
        embedding[:len(features)] = features
        
        return embedding
    
    def _compute_task_features(self, task, workflow) -> np.ndarray:
        """计算任务特征向量"""
        # 计算任务特征
        features = [
            task.flops,  # 计算量
            len(task.dependencies),  # 入度
            len(task.input_files),  # 输入文件数
            float(self._is_critical_task(task, workflow)),  # 是否在关键路径上
            self._calculate_task_criticality(task, workflow),  # 任务关键性
            self._calculate_data_locality_score(task, workflow),  # 数据局部性分数
        ]
        
        # 填充到64维
        embedding = np.zeros(64)
        embedding[:len(features)] = features
        
        return embedding
    
    def _estimate_workflow_makespan(self, workflow, platform_config) -> float:
        """估算工作流makespan（简化实现）"""
        # 简化实现：使用总计算量除以总处理能力
        total_computation = sum(task.flops for task in workflow.tasks)
        total_speed = sum(node['speed'] for node in platform_config['nodes'].values())
        return total_computation / total_speed
    
    def save_knowledge_base(self, kb: WRENCHRAGKnowledgeBase, filename: str = "enhanced_rag_kb.json"):
        """保存知识库到文件"""
        output_path = self.output_dir / filename
        
        # 转换为可序列化的格式
        serializable_cases = []
        for case in kb.cases:
            case_dict = asdict(case)
            # 转换numpy数组为列表
            case_dict['workflow_embedding'] = case_dict['workflow_embedding'].tolist()
            case_dict['task_features'] = case_dict['task_features'].tolist()
            case_dict['node_features'] = case_dict['node_features'].tolist()
            serializable_cases.append(case_dict)
        
        # 保存到文件
        with open(output_path, 'w') as f:
            json.dump({
                'cases': serializable_cases,
                'case_index': kb.case_index
            }, f, indent=2)
        
        logger.info(f"Knowledge base saved to {output_path}")
    
    def load_knowledge_base(self, filename: str = "enhanced_rag_kb.json") -> WRENCHRAGKnowledgeBase:
        """从文件加载知识库"""
        input_path = self.output_dir / filename
        
        if not input_path.exists():
            logger.warning(f"Knowledge base file {input_path} not found")
            return None
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # 创建知识库
        kb = WRENCHRAGKnowledgeBase(embedding_dim=64)
        
        # 加载案例
        for case_dict in data['cases']:
            # 转换列表为numpy数组
            case_dict['workflow_embedding'] = np.array(case_dict['workflow_embedding'])
            case_dict['task_features'] = np.array(case_dict['task_features'])
            case_dict['node_features'] = np.array(case_dict['node_features'])
            
            # 创建案例对象
            case = WRENCHKnowledgeCase(**case_dict)
            kb.add_case(case)
        
        logger.info(f"Knowledge base loaded from {input_path} with {len(kb.cases)} cases")
        return kb

def main():
    """主函数：生成增强的RAG知识库"""
    logger.info("Starting enhanced RAG knowledge base generation...")
    
    # 创建生成器
    generator = EnhancedRAGKnowledgeBaseGenerator()
    
    # 生成知识库
    kb = generator.generate_enhanced_knowledge_base(num_cases=5000)
    
    # 保存知识库
    generator.save_knowledge_base(kb)
    
    logger.info("Enhanced RAG knowledge base generation completed!")

if __name__ == "__main__":
    main()