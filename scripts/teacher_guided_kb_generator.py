#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进的RAG知识库生成器，使用优秀教师调度器的案例
"""

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.knowledge_base.wrench_full_kb import WRENCHKnowledgeCase, WRENCHRAGKnowledgeBase
from src.wrench_schedulers import HEFTScheduler, WassHeuristicScheduler

class TeacherGuidedKnowledgeGenerator:
    """教师引导的知识库生成器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.knowledge_base = WRENCHRAGKnowledgeBase()
        
        # 配置参数
        self.kb_cfg = self.config.get('knowledge_base', {})
        self.num_cases = self.kb_cfg.get('num_cases', 1000)
        self.output_path = self.kb_cfg.get('output_path', 'src/knowledge_base/wrench_teacher_guided_kb.json')
    
    def generate_wrench_workflow(self, num_tasks: int, complexity: str = 'medium'):
        """生成WRENCH工作流"""
        try:
            import wrench
            
            # 创建仿真环境
            simulation = wrench.Simulation()
            
            # 创建平台
            platform = simulation.create_platform([
                wrench.Host("ComputeHost1", "100Gf", ["100Gf", "100GB"]),
                wrench.Host("ComputeHost2", "150Gf", ["150Gf", "150GB"]),
                wrench.Host("ComputeHost3", "200Gf", ["200Gf", "200GB"]),
                wrench.Host("ComputeHost4", "250Gf", ["250Gf", "250GB"])
            ])
            
            # 创建计算服务
            compute_service = simulation.create_bare_metal_compute_service(
                "ComputeService",
                platform.get_hosts(),
                {}
            )
            
            # 创建工作流
            workflow = simulation.create_workflow()
            
            # 创建任务
            tasks = []
            for i in range(num_tasks):
                # 根据复杂度设置任务大小
                if complexity == 'simple':
                    flops = random.uniform(1e9, 5e9)
                elif complexity == 'medium':
                    flops = random.uniform(1e9, 1e10)
                else:  # complex
                    flops = random.uniform(5e9, 2e10)
                
                task = workflow.add_task(f"task_{i}", flops)
                tasks.append(task)
            
            # 生成依赖关系
            if complexity == 'simple':
                # 简单链式结构
                for i in range(1, num_tasks):
                    workflow.add_control_dependency(tasks[i-1], tasks[i])
            elif complexity == 'medium':
                # 中等复杂度，每个任务依赖1-2个前置任务
                for i in range(1, num_tasks):
                    num_deps = min(random.randint(1, 2), i)
                    for j in range(max(0, i-num_deps), i):
                        workflow.add_control_dependency(tasks[j], tasks[i])
            else:  # complex
                # 复杂结构，可能有多个依赖和分支
                for i in range(1, num_tasks):
                    num_deps = min(random.randint(1, 3), i)
                    deps = random.sample(range(i), num_deps)
                    for j in deps:
                        workflow.add_control_dependency(tasks[j], tasks[i])
            
            # 添加工作流到仿真
            simulation.add_workflow(workflow, "workflow_to_schedule")
            
            return workflow, tasks, simulation, compute_service
            
        except Exception as e:
            print(f"生成工作流失败: {e}")
            return None, None, None, None
    
    def extract_workflow_features(self, workflow, tasks):
        """提取工作流特征"""
        if not workflow or not tasks:
            return None
        
        features = {
            'num_tasks': len(tasks),
            'avg_task_size': np.mean([task.get_flops() for task in tasks]),
            'max_task_size': max([task.get_flops() for task in tasks]),
            'min_task_size': min([task.get_flops() for task in tasks]),
            'task_size_std': np.std([task.get_flops() for task in tasks]),
            'avg_dependencies': np.mean([len(task.get_parents()) for task in tasks]),
            'max_dependencies': max([len(task.get_parents()) for task in tasks]),
            'critical_path_length': self._estimate_critical_path_length(tasks),
            'parallelism_degree': self._estimate_parallelism_degree(tasks)
        }
        
        return features
    
    def _estimate_critical_path_length(self, tasks):
        """估计关键路径长度"""
        # 简化版本：计算最长路径
        task_depths = {}
        
        # 初始化入口任务
        for task in tasks:
            if len(task.get_parents()) == 0:
                task_depths[task] = 1
        
        # 动态规划计算深度
        changed = True
        while changed:
            changed = False
            for task in tasks:
                if task not in task_depths and all(parent in task_depths for parent in task.get_parents()):
                    parent_depths = [task_depths[parent] for parent in task.get_parents()]
                    task_depths[task] = max(parent_depths) + 1
                    changed = True
        
        return max(task_depths.values()) if task_depths else 1
    
    def _estimate_parallelism_degree(self, tasks):
        """估计并行度"""
        # 简化版本：计算平均每层的任务数
        task_depths = {}
        
        # 初始化入口任务
        for task in tasks:
            if len(task.get_parents()) == 0:
                task_depths[task] = 1
        
        # 动态规划计算深度
        changed = True
        while changed:
            changed = False
            for task in tasks:
                if task not in task_depths and all(parent in task_depths for parent in task.get_parents()):
                    parent_depths = [task_depths[parent] for parent in task.get_parents()]
                    task_depths[task] = max(parent_depths) + 1
                    changed = True
        
        # 统计每层任务数
        depth_counts = {}
        for depth in task_depths.values():
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        return np.mean(list(depth_counts.values())) if depth_counts else 1
    
    def simulate_teacher_scheduling(self, workflow, tasks, teacher_type='HEFT', simulation=None, compute_service=None):
        """模拟教师调度器的调度过程"""
        try:
            import wrench
            
            # 如果没有提供simulation，则创建新的仿真环境
            if simulation is None:
                simulation = wrench.Simulation()
                
                # 创建平台
                platform = simulation.create_platform([
                    wrench.Host("ComputeHost1", "100Gf", ["100Gf", "100GB"]),
                    wrench.Host("ComputeHost2", "150Gf", ["150Gf", "150GB"]),
                    wrench.Host("ComputeHost3", "200Gf", ["200Gf", "200GB"]),
                    wrench.Host("ComputeHost4", "250Gf", ["250Gf", "250GB"])
                ])
                
                # 创建计算服务
                compute_service = simulation.create_bare_metal_compute_service(
                    "ComputeService",
                    platform.get_hosts(),
                    {}
                )
                
                # 添加工作流到仿真
                simulation.add_workflow(workflow, "workflow_to_schedule")
            else:
                # 使用已有的simulation和compute_service
                platform = None  # 我们假设平台已经创建
            
            # 创建教师调度器
            hosts = {
                "ComputeHost1": [1, 100.0],
                "ComputeHost2": [1, 150.0],
                "ComputeHost3": [1, 200.0],
                "ComputeHost4": [1, 250.0]
            }
            
            if teacher_type == 'HEFT':
                scheduler = HEFTScheduler(simulation, compute_service, hosts)
            else:  # WASS-Heuristic
                scheduler = WassHeuristicScheduler(simulation, compute_service, hosts)
            
            # 执行调度
            scheduler.submit_ready_tasks(workflow)
            
            # 启动仿真
            simulation.launch()
            
            # 等待完成
            simulation.wait_for_completion()
            
            # 收集调度结果
            scheduling_cases = []
            
            for task in tasks:
                # 获取任务执行信息
                execution_info = {
                    'task_id': task.get_id(),
                    'task_flops': task.get_flops(),
                    'num_parents': len(task.get_parents()),
                    'num_children': len(task.get_children()),
                    'is_entry': len(task.get_parents()) == 0,
                    'is_exit': len(task.get_children()) == 0,
                    'assigned_host': task.get_execution_host().get_name() if task.get_execution_host() else None,
                    'start_time': task.get_start_time(),
                    'end_time': task.get_end_time(),
                    'execution_time': task.get_execution_time(),
                    'teacher_type': teacher_type
                }
                
                scheduling_cases.append(execution_info)
            
            # 计算总体makespan
            makespan = max([task.get_end_time() for task in tasks])
            
            return scheduling_cases, makespan
            
        except Exception as e:
            print(f"模拟教师调度失败: {e}")
            return [], float('inf')
    
    def create_knowledge_case(self, workflow_features, scheduling_case):
        """创建知识案例"""
        # 提取任务特征
        task_features = {
            'flops': scheduling_case['task_flops'],
            'num_parents': scheduling_case['num_parents'],
            'num_children': scheduling_case['num_children'],
            'is_entry': scheduling_case['is_entry'],
            'is_exit': scheduling_case['is_exit']
        }
        
        # 创建知识案例
        case = WRENCHKnowledgeCase(
            workflow_features=workflow_features,
            task_features=task_features,
            decision=scheduling_case['assigned_host'],
            performance_score=1.0 / (scheduling_case['execution_time'] + 1e-6),  # 执行时间越短越好
            metadata={
                'teacher_type': scheduling_case['teacher_type'],
                'start_time': scheduling_case['start_time'],
                'end_time': scheduling_case['end_time'],
                'execution_time': scheduling_case['execution_time']
            }
        )
        
        return case
    
    def generate_knowledge_base(self):
        """生成知识库"""
        print(f"🚀 开始生成教师引导的知识库: {self.num_cases} 个案例")
        
        # 生成不同规模和复杂度的工作流
        workflow_configs = [
            {'num_tasks': 5, 'complexity': 'simple'},
            {'num_tasks': 10, 'complexity': 'medium'},
            {'num_tasks': 15, 'complexity': 'medium'},
            {'num_tasks': 20, 'complexity': 'complex'}
        ]
        
        # 计算每个配置的案例数量
        cases_per_config = self.num_cases // len(workflow_configs)
        
        total_cases = 0
        
        for config in workflow_configs:
            print(f"📝 生成配置: {config['num_tasks']} 任务, {config['complexity']} 复杂度")
            
            config_cases = 0
            
            while config_cases < cases_per_config:
                # 生成工作流
                workflow, tasks, simulation, compute_service = self.generate_wrench_workflow(
                    config['num_tasks'], 
                    config['complexity']
                )
                
                if workflow is None:
                    continue
                
                # 提取工作流特征
                workflow_features = self.extract_workflow_features(workflow, tasks)
                
                # 为每个教师调度器生成案例
                for teacher_type in ['HEFT', 'WASS-Heuristic']:
                    # 模拟教师调度
                    scheduling_cases, makespan = self.simulate_teacher_scheduling(
                        workflow, tasks, teacher_type, simulation, compute_service
                    )
                    
                    if not scheduling_cases:
                        continue
                    
                    # 创建知识案例
                    for scheduling_case in scheduling_cases:
                        case = self.create_knowledge_case(workflow_features, scheduling_case)
                        self.knowledge_base.add_case(case)
                        total_cases += 1
                        config_cases += 1
                    
                    print(f"   {teacher_type}: {len(scheduling_cases)} 个案例, Makespan: {makespan:.2f}s")
                
                # 进度报告
                if total_cases % 100 == 0:
                    print(f"📊 已生成 {total_cases} 个案例")
        
        # 保存知识库
        self.knowledge_base.save(self.output_path)
        
        print(f"✅ 知识库生成完成!")
        print(f"   总案例数: {total_cases}")
        print(f"   保存路径: {self.output_path}")
        
        return self.knowledge_base

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WASS-RAG教师引导知识库生成')
    parser.add_argument('--config', type=str, default='configs/experiment.yaml', help='配置文件路径')
    parser.add_argument('--num-cases', type=int, default=1000, help='生成的案例数量')
    
    args = parser.parse_args()
    
    # 创建知识库生成器
    generator = TeacherGuidedKnowledgeGenerator(args.config)
    
    # 生成知识库
    knowledge_base = generator.generate_knowledge_base()
    
    print("🎉 知识库生成完成!")

if __name__ == '__main__':
    main()