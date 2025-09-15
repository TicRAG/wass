#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
综合实验脚本：测试教师引导的DRL和RAG调度器
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml
import argparse

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.wrench_schedulers import HEFTScheduler, WassHeuristicScheduler
from scripts.teacher_guided_drl_trainer import WRENCHBasedDRLTrainer
from scripts.teacher_guided_kb_generator import TeacherGuidedKnowledgeGenerator
from scripts.teacher_guided_rag_scheduler import TeacherGuidedRAGScheduler

class TeacherGuidedExperiment:
    """教师引导的综合实验"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.exp_cfg = self.config.get('experiment', {})
        self.results_dir = Path(self.exp_cfg.get('results_dir', 'results/teacher_guided_experiments'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.num_runs = self.exp_cfg.get('num_runs', 12)
        self.workflow_sizes = self.exp_cfg.get('workflow_sizes', [5, 10, 15, 20])
        
        # 调度器
        self.schedulers = {
            'HEFT': HEFTScheduler(),
            'WASS-Heuristic': WassHeuristicScheduler(),
            'WASS-DRL': None,  # 将在训练后初始化
            'WASS-RAG': None   # 将在初始化后设置
        }
        
        # 实验结果
        self.results = []
        self.detailed_results = []
    
    def generate_wrench_workflow(self, num_tasks: int, complexity: str = 'medium'):
        """生成WRENCH工作流"""
        try:
            import wrench
            
            # 创建工作流
            workflow = wrench.Workflow()
            
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
            
            return workflow, tasks
            
        except Exception as e:
            print(f"生成工作流失败: {e}")
            return None, None
    
    def run_scheduler_experiment(self, scheduler_name: str, workflow, tasks, run_id: int):
        """运行单个调度器实验"""
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
            
            # 添加工作流到仿真
            simulation.add_workflow(workflow)
            
            # 获取调度器
            scheduler = self.schedulers[scheduler_name]
            
            # 执行调度
            start_time = time.time()
            
            if scheduler_name == 'WASS-RAG':
                # RAG调度器需要特殊处理
                scheduling_decisions, makespan = scheduler.schedule(workflow, compute_service)
            else:
                # 其他调度器
                scheduler.schedule(workflow, compute_service)
                
                # 启动仿真
                simulation.launch()
                
                # 等待完成
                simulation.wait_for_completion()
                
                # 计算makespan
                makespan = max([task.get_end_time() for task in tasks])
            
            end_time = time.time()
            scheduling_time = end_time - start_time
            
            # 收集详细结果
            detailed_result = {
                'run_id': run_id,
                'scheduler': scheduler_name,
                'workflow_size': len(tasks),
                'makespan': makespan,
                'scheduling_time': scheduling_time,
                'task_details': []
            }
            
            # 收集任务详情
            for task in tasks:
                task_detail = {
                    'task_id': task.get_id(),
                    'flops': task.get_flops(),
                    'start_time': task.get_start_time(),
                    'end_time': task.get_end_time(),
                    'execution_time': task.get_execution_time(),
                    'assigned_host': task.get_execution_host().get_name() if task.get_execution_host() else None
                }
                detailed_result['task_details'].append(task_detail)
            
            return makespan, scheduling_time, detailed_result
            
        except Exception as e:
            print(f"运行调度器实验失败 {scheduler_name}: {e}")
            return float('inf'), float('inf'), None
    
    def train_drl_agent(self):
        """训练DRL智能体"""
        print("🚀 开始训练DRL智能体...")
        
        # 创建训练器
        trainer = WRENCHBasedDRLTrainer(self.config)
        
        # 训练
        episodes = self.config.get('drl', {}).get('episodes', 500)
        results = trainer.train(episodes)
        
        print(f"✅ DRL训练完成! 最佳Makespan: {results['best_makespan']:.2f}s")
        
        # 加载训练好的模型
        model_path = Path(self.config.get('checkpoint', {}).get('dir', 'models/checkpoints/')) / 'wass_drl_teacher_guided.pth'
        
        if model_path.exists():
            checkpoint = torch.load(model_path, weights_only=False)
            # 这里需要初始化DRL智能体并加载权重
            # 简化版本：直接使用训练器中的智能体
            self.schedulers['WASS-DRL'] = trainer.agent
            print(f"📁 DRL模型已加载: {model_path}")
        else:
            print(f"⚠️  模型文件不存在: {model_path}")
    
    def generate_knowledge_base(self):
        """生成知识库"""
        print("🚀 开始生成知识库...")
        
        # 创建知识库生成器
        generator = TeacherGuidedKnowledgeGenerator(self.config)
        
        # 生成知识库
        knowledge_base = generator.generate_knowledge_base()
        
        print(f"✅ 知识库生成完成! 总案例数: {len(knowledge_base.cases)}")
    
    def initialize_rag_scheduler(self):
        """初始化RAG调度器"""
        print("🚀 初始化RAG调度器...")
        
        # 创建RAG调度器
        rag_scheduler = TeacherGuidedRAGScheduler(self.config)
        
        # 设置调度器
        self.schedulers['WASS-RAG'] = rag_scheduler
        
        print("✅ RAG调度器初始化完成")
    
    def run_single_experiment(self, workflow_size: int, run_id: int):
        """运行单次实验"""
        print(f"📊 运行实验: 工作流大小 {workflow_size}, 运行ID {run_id}")
        
        # 生成工作流
        complexity = 'medium'  # 使用中等复杂度
        workflow, tasks = self.generate_wrench_workflow(workflow_size, complexity)
        
        if workflow is None:
            return None
        
        # 运行所有调度器
        scheduler_results = {}
        
        for scheduler_name in self.schedulers:
            if self.schedulers[scheduler_name] is None:
                continue
            
            print(f"   运行调度器: {scheduler_name}")
            makespan, scheduling_time, detailed_result = self.run_scheduler_experiment(
                scheduler_name, workflow, tasks, run_id
            )
            
            scheduler_results[scheduler_name] = {
                'makespan': makespan,
                'scheduling_time': scheduling_time
            }
            
            # 保存详细结果
            if detailed_result:
                self.detailed_results.append(detailed_result)
        
        return scheduler_results
    
    def run_all_experiments(self):
        """运行所有实验"""
        print(f"🚀 开始运行所有实验: {self.num_runs} 次运行")
        
        # 训练DRL智能体
        self.train_drl_agent()
        
        # 生成知识库
        self.generate_knowledge_base()
        
        # 初始化RAG调度器
        self.initialize_rag_scheduler()
        
        # 运行实验
        for run_id in range(self.num_runs):
            print(f"\n🎯 运行 {run_id + 1}/{self.num_runs}")
            
            # 为每个工作流大小运行实验
            for workflow_size in self.workflow_sizes:
                result = self.run_single_experiment(workflow_size, run_id)
                
                if result:
                    # 添加到结果列表
                    experiment_result = {
                        'run_id': run_id,
                        'workflow_size': workflow_size,
                        'results': result
                    }
                    self.results.append(experiment_result)
        
        # 保存结果
        self.save_results()
        
        # 分析结果
        self.analyze_results()
    
    def save_results(self):
        """保存实验结果"""
        # 保存详细结果
        detailed_path = self.results_dir / 'detailed_results.json'
        with open(detailed_path, 'w') as f:
            json.dump(self.detailed_results, f, indent=2)
        
        # 保存汇总结果
        summary_path = self.results_dir / 'summary_results.json'
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📁 结果已保存:")
        print(f"   详细结果: {detailed_path}")
        print(f"   汇总结果: {summary_path}")
    
    def analyze_results(self):
        """分析实验结果"""
        print("\n📊 分析实验结果...")
        
        # 按调度器和工作流大小分组
        scheduler_stats = {}
        
        for scheduler_name in self.schedulers:
            if self.schedulers[scheduler_name] is None:
                continue
            
            scheduler_stats[scheduler_name] = {
                'all_makespans': [],
                'all_scheduling_times': [],
                'by_workflow_size': {}
            }
            
            for size in self.workflow_sizes:
                scheduler_stats[scheduler_name]['by_workflow_size'][size] = {
                    'makespans': [],
                    'scheduling_times': []
                }
        
        # 收集数据
        for experiment in self.results:
            workflow_size = experiment['workflow_size']
            
            for scheduler_name, result in experiment['results'].items():
                makespan = result['makespan']
                scheduling_time = result['scheduling_time']
                
                # 添加到总体统计
                scheduler_stats[scheduler_name]['all_makespans'].append(makespan)
                scheduler_stats[scheduler_name]['all_scheduling_times'].append(scheduling_time)
                
                # 添加到按大小分组的统计
                scheduler_stats[scheduler_name]['by_workflow_size'][workflow_size]['makespans'].append(makespan)
                scheduler_stats[scheduler_name]['by_workflow_size'][workflow_size]['scheduling_times'].append(scheduling_time)
        
        # 计算统计指标
        analysis_results = {}
        
        for scheduler_name, stats in scheduler_stats.items():
            # 总体统计
            all_makespans = stats['all_makespans']
            all_scheduling_times = stats['all_scheduling_times']
            
            analysis_results[scheduler_name] = {
                'avg_makespan': np.mean(all_makespans),
                'std_makespan': np.std(all_makespans),
                'min_makespan': np.min(all_makespans),
                'max_makespan': np.max(all_makespans),
                'avg_scheduling_time': np.mean(all_scheduling_times),
                'std_scheduling_time': np.std(all_scheduling_times),
                'by_workflow_size': {}
            }
            
            # 按工作流大小统计
            for size, size_stats in stats['by_workflow_size'].items():
                makespans = size_stats['makespans']
                scheduling_times = size_stats['scheduling_times']
                
                analysis_results[scheduler_name]['by_workflow_size'][size] = {
                    'avg_makespan': np.mean(makespans),
                    'std_makespan': np.std(makespans),
                    'min_makespan': np.min(makespans),
                    'max_makespan': np.max(makespans),
                    'avg_scheduling_time': np.mean(scheduling_times),
                    'std_scheduling_time': np.std(scheduling_times)
                }
        
        # 保存分析结果
        analysis_path = self.results_dir / 'analysis_results.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # 打印汇总结果
        self.print_summary(analysis_results)
        
        print(f"📊 分析结果已保存: {analysis_path}")
    
    def print_summary(self, analysis_results):
        """打印汇总结果"""
        print("\n" + "="*80)
        print("🏆 教师引导调度器性能汇总")
        print("="*80)
        
        # 找出最佳调度器
        best_scheduler = None
        best_makespan = float('inf')
        
        print("\n== 全局调度器性能 ==")
        print(f"{'调度器':<15} {'平均Makespan':<15} {'标准差':<10} {'最佳':<10} {'实验次数':<10}")
        print("-" * 70)
        
        for scheduler_name, stats in analysis_results.items():
            avg_makespan = stats['avg_makespan']
            std_makespan = stats['std_makespan']
            min_makespan = stats['min_makespan']
            count = len(stats['by_workflow_size'][self.workflow_sizes[0]]['makespans']) * len(self.workflow_sizes)
            
            print(f"{scheduler_name:<15} {avg_makespan:<15.2f} {std_makespan:<10.2f} {min_makespan:<10.2f} {count:<10}")
            
            if avg_makespan < best_makespan:
                best_makespan = avg_makespan
                best_scheduler = scheduler_name
        
        print(f"\n🏆 最佳调度器: {best_scheduler} (平均Makespan: {best_makespan:.2f}s)")
        
        # 按工作流大小打印结果
        print("\n== 按工作流大小的平均Makespan ==")
        for size in self.workflow_sizes:
            print(f"\n工作流大小 {size}")
            size_results = []
            
            for scheduler_name, stats in analysis_results.items():
                size_stats = stats['by_workflow_size'][size]
                avg_makespan = size_stats['avg_makespan']
                size_results.append((scheduler_name, avg_makespan))
            
            # 按性能排序
            size_results.sort(key=lambda x: x[1])
            
            for scheduler_name, avg_makespan in size_results:
                print(f"  {scheduler_name}: {avg_makespan:.2f}")
            
            # 计算改进百分比
            if size_results:
                baseline = size_results[0][1]  # 最佳性能
                for scheduler_name, avg_makespan in size_results[1:]:
                    improvement = ((avg_makespan - baseline) / baseline) * 100
                    print(f"  -> {scheduler_name} vs {size_results[0][0]}: {improvement:.2f}%")
        
        print("\n" + "="*80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='教师引导的WASS-DRL和RAG实验')
    parser.add_argument('--config', type=str, default='configs/experiment.yaml', help='配置文件路径')
    parser.add_argument('--runs', type=int, default=12, help='实验运行次数')
    parser.add_argument('--workflow-sizes', type=str, default='5,10,15,20', help='工作流大小，逗号分隔')
    
    args = parser.parse_args()
    
    # 解析工作流大小
    workflow_sizes = [int(s.strip()) for s in args.workflow_sizes.split(',')]
    
    # 创建实验
    experiment = TeacherGuidedExperiment(args.config)
    
    # 更新配置
    experiment.num_runs = args.runs
    experiment.workflow_sizes = workflow_sizes
    
    print(f"🚀 开始教师引导实验:")
    print(f"   实验次数: {args.runs}")
    print(f"   工作流大小: {workflow_sizes}")
    
    # 运行实验
    experiment.run_all_experiments()
    
    print("\n🎉 所有实验完成!")

if __name__ == '__main__':
    main()