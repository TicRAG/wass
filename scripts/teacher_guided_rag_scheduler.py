#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进的WASS-RAG调度器，更好地融合教师知识和DRL决策
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.knowledge_base.wrench_full_kb import WRENCHRAGKnowledgeBase
from src.wrench_schedulers import HEFTScheduler, WassHeuristicScheduler
from src.drl.reward import compute_step_reward, compute_final_reward, StepContext, EpisodeStats

class TeacherGuidedRAGScheduler:
    """教师引导的RAG调度器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rag_cfg = self.config.get('rag', {})
        self.drl_cfg = self.config.get('drl', {})
        
        # 加载知识库
        kb_path = self.rag_cfg.get('knowledge_base_path', 'src/knowledge_base/wrench_teacher_guided_kb.json')
        self.knowledge_base = WRENCHRAGKnowledgeBase.load(kb_path)
        
        # 融合权重（动态调整）
        self.drl_weight = 0.3  # DRL决策权重
        self.rag_weight = 0.4  # RAG建议权重
        self.teacher_weight = 0.3  # 教师建议权重
        
        # 自适应参数
        self.confidence_threshold = 0.7  # 置信度阈值
        self.adaptation_rate = 0.01  # 自适应学习率
        
        # 性能统计
        self.performance_history = []
        self.decision_history = []
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_workflow_features(self, workflow):
        """提取工作流特征"""
        tasks = workflow.get_tasks()
        
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
    
    def extract_task_features(self, task):
        """提取任务特征"""
        return {
            'flops': task.get_flops(),
            'num_parents': len(task.get_parents()),
            'num_children': len(task.get_children()),
            'is_entry': len(task.get_parents()) == 0,
            'is_exit': len(task.get_children()) == 0
        }
    
    def _estimate_critical_path_length(self, tasks):
        """估计关键路径长度"""
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
    
    def get_rag_suggestion(self, workflow_features, task_features, available_nodes):
        """获取RAG建议"""
        # 从知识库中检索相似案例
        similar_cases = self.knowledge_base.retrieve_similar_cases(
            workflow_features, task_features, top_k=5
        )
        
        if not similar_cases:
            return None, 0.0
        
        # 统计每个节点的推荐次数和平均性能分数
        node_scores = {}
        node_counts = {}
        
        for case in similar_cases:
            decision = case.decision
            score = case.performance_score
            
            if decision in node_scores:
                node_scores[decision] += score
                node_counts[decision] += 1
            else:
                node_scores[decision] = score
                node_counts[decision] = 1
        
        # 计算每个节点的平均分数
        avg_scores = {}
        for node in node_scores:
            avg_scores[node] = node_scores[node] / node_counts[node]
        
        # 选择最佳节点
        best_node = max(avg_scores, key=avg_scores.get)
        confidence = avg_scores[best_node] / max(avg_scores.values()) if avg_scores else 0.0
        
        # 如果最佳节点不在可用节点中，返回None
        if best_node not in available_nodes:
            return None, 0.0
        
        return best_node, confidence
    
    def get_teacher_suggestion(self, task, available_nodes, node_capacities, node_loads, teacher_type='WASS-Heuristic'):
        """获取教师调度器建议"""
        if teacher_type == 'HEFT':
            # HEFT策略：选择能最早完成任务的节点
            best_node = None
            best_finish_time = float('inf')
            
            for node in available_nodes:
                capacity = node_capacities.get(node, 1.0)
                load = node_loads.get(node, 0.0)
                exec_time = task.get_flops() / (capacity * 1e9)
                finish_time = load + exec_time
                
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_node = node
            
            return best_node, 1.0  # HEFT总是有高置信度
        
        elif teacher_type == 'WASS-Heuristic':
            # WASS-Heuristic策略：考虑数据局部性
            best_node = None
            best_score = float('inf')
            
            for node in available_nodes:
                # 计算EFT
                capacity = node_capacities.get(node, 1.0)
                load = node_loads.get(node, 0.0)
                exec_time = task.get_flops() / (capacity * 1e9)
                eft = load + exec_time
                
                # 简化的DRT计算（模拟数据局部性）
                drt = 0.0
                for parent in task.get_parents():
                    # 假设父任务可能在任何节点上执行
                    if random.random() > 0.5:  # 50%概率需要数据传输
                        file_size = task.get_flops() * 0.1  # 假设数据大小
                        network_bandwidth = 1e9  # 1GB/s
                        drt += file_size / network_bandwidth
                
                # WASS评分
                w = 0.5  # 数据局部性权重
                score = (1 - w) * eft + w * drt
                
                if score < best_score:
                    best_score = score
                    best_node = node
            
            return best_node, 1.0  # WASS-Heuristic总是有高置信度
        
        else:
            # 默认随机选择
            return random.choice(available_nodes), 0.0
    
    def adaptive_fusion(self, drl_q_values, rag_suggestion, rag_confidence, teacher_suggestion, teacher_confidence):
        """自适应融合DRL、RAG和教师建议"""
        # 根据历史性能调整权重
        if self.performance_history:
            recent_performance = np.mean(self.performance_history[-10:])
            if recent_performance > 0.8:  # 如果近期性能好，增加DRL权重
                self.drl_weight = min(0.6, self.drl_weight + self.adaptation_rate)
                self.rag_weight = max(0.2, self.rag_weight - self.adaptation_rate * 0.5)
                self.teacher_weight = max(0.2, self.teacher_weight - self.adaptation_rate * 0.5)
            else:  # 如果近期性能差，增加教师和RAG权重
                self.drl_weight = max(0.1, self.drl_weight - self.adaptation_rate)
                self.rag_weight = min(0.5, self.rag_weight + self.adaptation_rate * 0.5)
                self.teacher_weight = min(0.4, self.teacher_weight + self.adaptation_rate * 0.5)
        
        # 归一化权重
        total_weight = self.drl_weight + self.rag_weight + self.teacher_weight
        drl_w = self.drl_weight / total_weight
        rag_w = self.rag_weight / total_weight
        teacher_w = self.teacher_weight / total_weight
        
        # 计算融合分数
        fusion_scores = {}
        
        # DRL分数
        for i, q_value in enumerate(drl_q_values):
            node = f"ComputeHost{i+1}"
            fusion_scores[node] = drl_w * float(q_value)
        
        # RAG分数
        if rag_suggestion and rag_confidence > self.confidence_threshold:
            rag_score = rag_w * rag_confidence
            fusion_scores[rag_suggestion] = fusion_scores.get(rag_suggestion, 0) + rag_score
        
        # 教师分数
        if teacher_suggestion and teacher_confidence > self.confidence_threshold:
            teacher_score = teacher_w * teacher_confidence
            fusion_scores[teacher_suggestion] = fusion_scores.get(teacher_suggestion, 0) + teacher_score
        
        # 选择最佳节点
        best_node = max(fusion_scores, key=fusion_scores.get)
        best_score = fusion_scores[best_node]
        
        return best_node, best_score
    
    def schedule(self, workflow, compute_service, drl_agent=None):
        """执行调度"""
        try:
            import wrench
            
            # 获取可用节点
            available_nodes = [host.get_name() for host in compute_service.get_hosts()]
            
            # 节点容量和负载
            node_capacities = {
                "ComputeHost1": 100.0,
                "ComputeHost2": 150.0,
                "ComputeHost3": 200.0,
                "ComputeHost4": 250.0
            }
            node_loads = {node: 0.0 for node in available_nodes}
            
            # 提取工作流特征
            workflow_features = self.extract_workflow_features(workflow)
            
            # 获取任务列表
            tasks = workflow.get_tasks()
            
            # 按依赖关系排序任务
            ready_tasks = [t for t in tasks if len(t.get_parents()) == 0]
            completed_tasks = set()
            
            # 调度决策
            scheduling_decisions = []
            
            while ready_tasks:
                # 选择当前任务（按优先级）
                current_task = ready_tasks[0]
                
                # 提取任务特征
                task_features = self.extract_task_features(current_task)
                
                # 获取RAG建议
                rag_suggestion, rag_confidence = self.get_rag_suggestion(
                    workflow_features, task_features, available_nodes
                )
                
                # 获取教师建议
                teacher_suggestion, teacher_confidence = self.get_teacher_suggestion(
                    current_task, available_nodes, node_capacities, node_loads
                )
                
                # 获取DRL建议
                drl_q_values = None
                if drl_agent:
                    # 这里需要从DRL智能体获取Q值
                    # 简化版本：使用随机Q值
                    drl_q_values = np.random.rand(4)
                
                # 融合决策
                if drl_q_values is not None:
                    best_node, fusion_score = self.adaptive_fusion(
                        drl_q_values, rag_suggestion, rag_confidence,
                        teacher_suggestion, teacher_confidence
                    )
                else:
                    # 没有DRL智能体，只使用RAG和教师建议
                    if rag_suggestion and teacher_suggestion:
                        # 简单选择置信度更高的
                        if rag_confidence > teacher_confidence:
                            best_node = rag_suggestion
                        else:
                            best_node = teacher_suggestion
                    elif rag_suggestion:
                        best_node = rag_suggestion
                    elif teacher_suggestion:
                        best_node = teacher_suggestion
                    else:
                        best_node = random.choice(available_nodes)
                
                # 执行调度
                try:
                    # 找到对应的主机对象
                    host = None
                    for h in compute_service.get_hosts():
                        if h.get_name() == best_node:
                            host = h
                            break
                    
                    if host:
                        # 创建标准作业
                        standard_job = wrench.StandardJob([current_task])
                        
                        # 提交作业
                        compute_service.submit_standard_job(standard_job, {host})
                        
                        # 更新节点负载
                        capacity = node_capacities[best_node]
                        exec_time = current_task.get_flops() / (capacity * 1e9)
                        node_loads[best_node] += exec_time
                        
                        # 记录决策
                        decision = {
                            'task_id': current_task.get_id(),
                            'assigned_node': best_node,
                            'rag_suggestion': rag_suggestion,
                            'rag_confidence': rag_confidence,
                            'teacher_suggestion': teacher_suggestion,
                            'teacher_confidence': teacher_confidence,
                            'fusion_score': fusion_score if drl_q_values is not None else 0.0,
                            'weights': {
                                'drl': self.drl_weight,
                                'rag': self.rag_weight,
                                'teacher': self.teacher_weight
                            }
                        }
                        scheduling_decisions.append(decision)
                        
                        # 更新任务状态
                        completed_tasks.add(current_task)
                        ready_tasks.remove(current_task)
                        
                        # 更新就绪任务列表
                        for child in current_task.get_children():
                            if all(parent in completed_tasks for parent in child.get_parents()):
                                if child not in ready_tasks:
                                    ready_tasks.append(child)
                    
                except Exception as e:
                    print(f"调度任务失败: {e}")
                    # 如果调度失败，随机选择一个节点
                    best_node = random.choice(available_nodes)
            
            # 计算性能指标
            makespan = max(node_loads.values()) if node_loads else 0.0
            
            # 记录性能
            performance_score = 1.0 / (makespan + 1e-6)
            self.performance_history.append(performance_score)
            self.decision_history.append(scheduling_decisions)
            
            return scheduling_decisions, makespan
            
        except Exception as e:
            print(f"调度失败: {e}")
            return [], float('inf')
    
    def save_performance_stats(self, filepath):
        """保存性能统计"""
        stats = {
            'performance_history': self.performance_history,
            'decision_history': self.decision_history,
            'final_weights': {
                'drl': self.drl_weight,
                'rag': self.rag_weight,
                'teacher': self.teacher_weight
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"性能统计已保存: {filepath}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WASS-RAG教师引导调度器')
    parser.add_argument('--config', type=str, default='configs/experiment.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default='results/teacher_guided_rag_stats.json', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 创建调度器
    scheduler = TeacherGuidedRAGScheduler(args.config)
    
    print("🚀 教师引导的RAG调度器已创建")
    print(f"   DRL权重: {scheduler.drl_weight:.3f}")
    print(f"   RAG权重: {scheduler.rag_weight:.3f}")
    print(f"   教师权重: {scheduler.teacher_weight:.3f}")
    print(f"   置信度阈值: {scheduler.confidence_threshold:.3f}")
    
    # 保存初始统计
    scheduler.save_performance_stats(args.output)

if __name__ == '__main__':
    main()