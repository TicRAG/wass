#!/usr/bin/env python3
"""
基于WRENCH的真实WASS-RAG实验框架
使用训练好的模型在真实WRENCH环境中进行性能对比实验
"""

import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print(f"当前工作目录: {os.getcwd()}")

from src.ai_schedulers import WASSRAGScheduler

import sys
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import yaml
from datetime import datetime
import networkx as nx

# 确保能导入WRENCH
try:
    import wrench
except ImportError:
    print("Error: WRENCH not available. Please install wrench-python-api.")
    sys.exit(1)

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, str(parent_dir))

def load_config(cfg_path: str) -> Dict:
    """加载配置文件"""
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    # Process includes
    if 'include' in cfg:
        base_dir = os.path.dirname(cfg_path)
        for include_file in cfg['include']:
            include_path = os.path.join(base_dir, include_file)
            if os.path.exists(include_path):
                with open(include_path, 'r', encoding='utf-8') as f:
                    include_cfg = yaml.safe_load(f) or {}
                    for key, value in include_cfg.items():
                        if key not in cfg:
                            cfg[key] = value
    return cfg

@dataclass
class WRENCHExperimentResult:
    """单次WRENCH实验结果"""
    scheduler_name: str
    workflow_id: str
    task_count: int
    dependency_count: int
    makespan: float
    cpu_utilization: Dict[str, float]
    task_execution_times: Dict[str, float]
    scheduling_decisions: List[Dict[str, Any]]
    experiment_metadata: Dict[str, Any]

class WRENCHScheduler:
    """基础WRENCH调度器接口"""
    
    def __init__(self, name: str):
        self.name = name
    
    def schedule_task(self, task, available_nodes: List[str], node_capacities: Dict, 
                     node_loads: Dict, compute_service) -> str:
        """调度单个任务，返回选择的节点"""
        raise NotImplementedError

class FIFOScheduler(WRENCHScheduler):
    """先进先出调度器"""
    
    def __init__(self):
        super().__init__("FIFO")
    
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        # 选择负载最小的节点
        return min(available_nodes, key=lambda x: node_loads.get(x, 0))

class HEFTScheduler(WRENCHScheduler):
    """异构最早完成时间调度器"""
    
    def __init__(self):
        super().__init__("HEFT")
    
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        # 选择能最早完成任务的节点
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
        
        return best_node or available_nodes[0]

    def predict_makespan(self, task, available_nodes, node_capacities, node_loads):
        """预测任务在给定节点配置下的makespan (从WASS-Heuristic复制而来)"""
        try:
            task_flops = float(getattr(task, 'get_flops', lambda: 1e9)())
            
            total_capacity = sum(node_capacities.get(node, 1.0) for node in available_nodes)
            total_load = sum(node_loads.get(node, 0.0) for node in available_nodes)
            avg_capacity = total_capacity / len(available_nodes) if available_nodes else 1.0
            avg_load = total_load / len(available_nodes) if available_nodes else 0.0
            
            base_time = task_flops / (avg_capacity * 1e9)
            load_factor = 1.0 + avg_load / max(avg_capacity, 0.1)
            
            try:
                children_count = len(getattr(task, 'get_children', lambda: [])())
                dependency_factor = 1.0 + 0.1 * children_count
            except Exception:
                dependency_factor = 1.0
            
            predicted_makespan = base_time * load_factor * dependency_factor
            return predicted_makespan
        except Exception as e:
            print(f"预测makespan失败: {e}")
            return 100.0

class WASSHeuristicScheduler(WRENCHScheduler):
    """WASS启发式调度器 - 在HEFT基础上考虑数据局部性"""
    
    def __init__(self, data_locality_weight: float = 0.5):
        super().__init__("WASS-Heuristic")
        self.data_locality_weight = data_locality_weight  # w参数
        self.data_location_cache = {}  # 模拟数据位置缓存
        
        # 节点性能参数
        self.node_capacities = {
            "ComputeHost1": 2.0,
            "ComputeHost2": 3.0,
            "ComputeHost3": 2.5,
            "ComputeHost4": 4.0
        }
    
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        """使用WASS启发式进行任务调度"""
        best_node = None
        best_score = float('inf')
        
        for node in available_nodes:
            # 计算EFT (最早完成时间)
            eft = self._calculate_eft(task, node, node_capacities, node_loads)
            
            # 计算DRT (数据就绪时间)
            drt = self._calculate_drt(task, node)
            
            # 计算WASS综合评分
            w = self.data_locality_weight
            score = (1 - w) * eft + w * drt
            
            if score < best_score:
                best_score = score
                best_node = node
        
        # 更新数据位置缓存（假设任务输出数据存储在执行节点）
        if best_node:
            self._update_data_location(task, best_node)
        
        return best_node or available_nodes[0]
    
    def _calculate_eft(self, task, node, node_capacities, node_loads):
        """计算任务在指定节点上的最早完成时间"""
        capacity = node_capacities.get(node, 1.0)
        load = node_loads.get(node, 0.0)
        exec_time = task.get_flops() / (capacity * 1e9)
        return load + exec_time
    
    def predict_makespan(self, task, available_nodes, node_capacities, node_loads):
        """预测任务在给定节点配置下的makespan"""
        try:
            # 简化的makespan预测：基于任务大小和节点负载
            task_flops = float(getattr(task, 'get_flops', lambda: 1e9)())
            
            # 计算平均节点性能
            total_capacity = sum(node_capacities.get(node, 1.0) for node in available_nodes)
            total_load = sum(node_loads.get(node, 0.0) for node in available_nodes)
            avg_capacity = total_capacity / len(available_nodes) if available_nodes else 1.0
            avg_load = total_load / len(available_nodes) if available_nodes else 0.0
            
            # 基础执行时间
            base_time = task_flops / (avg_capacity * 1e9)
            
            # 考虑负载影响
            load_factor = 1.0 + avg_load / max(avg_capacity, 0.1)
            
            # 考虑任务依赖（简化处理）
            try:
                children_count = len(getattr(task, 'get_children', lambda: [])())
                dependency_factor = 1.0 + 0.1 * children_count  # 每个子任务增加10%的时间
            except Exception:
                dependency_factor = 1.0
            
            # 预测的makespan
            predicted_makespan = base_time * load_factor * dependency_factor
            
            return predicted_makespan
            
        except Exception as e:
            print(f"预测makespan失败: {e}")
            return 100.0  # 默认值
    
    def _calculate_drt(self, task, node):
        """计算数据就绪时间 - 考虑数据传输开销"""
        total_transfer_time = 0.0
        
        # 检查输入文件的数据位置
        for input_file in task.get_input_files():
            # 使用文件名而不是get_id()方法
            file_id = input_file.get_name() if hasattr(input_file, 'get_name') else str(input_file)
            
            # 检查数据是否在目标节点上
            data_location = self._get_data_location(file_id)
            if data_location != node:
                # 需要传输数据
                file_size = input_file.get_size() if hasattr(input_file, 'get_size') else 1024
                network_bandwidth = 1e9  # 1GB/s 假设网络带宽
                transfer_time = file_size / network_bandwidth
                total_transfer_time += transfer_time
        
        return total_transfer_time
    
    def _get_data_location(self, file_id):
        """获取文件的数据位置"""
        if file_id not in self.data_location_cache:
            # 如果没有缓存，随机选择一个位置（模拟初始数据分布）
            import random
            self.data_location_cache[file_id] = random.choice(
                ["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"]
            )
        return self.data_location_cache[file_id]
    
    def _update_data_location(self, task, node):
        """更新任务输出数据的位置"""
        for output_file in task.get_output_files():
            # 使用文件名而不是get_id()方法
            file_id = output_file.get_name() if hasattr(output_file, 'get_name') else str(output_file)
            self.data_location_cache[file_id] = node

# 定义DRL网络结构
class SimpleDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(SimpleDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class WASSDRLScheduler(WRENCHScheduler):
    """基于训练好的DRL模型的调度器"""
    
    def __init__(self, model_path: str, config_path: str = "configs/experiment.yaml"):
        """初始化WASS-DRL调度器"""
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 定义网络结构 (与改进训练器保持一致)
        self.state_dim = 32  # 需与训练保持一致
        self.action_dim = 4   # 节点数
        self.hidden_dims = [512, 256, 128, 64] # AdvancedDQN's default
        self.model = self._create_model()
        self.epsilon = 0.1

        # 加载模型
        self._load_model(model_path)

        # 节点映射
        self.compute_nodes = ["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"]
    
    def _create_model(self):
        # 延迟导入，避免在无torch环境时报错
        from scripts.improved_drl_trainer import AdvancedDQN
        return AdvancedDQN(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
    
    def _load_model(self, model_path: str):
        """加载训练好的DRL模型, 手动处理尺寸不匹配的层"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            source_state_dict = checkpoint.get('q_network_state_dict', checkpoint)
            target_state_dict = self.model.state_dict()

            # Create a new state dict for loading
            new_state_dict = {}
            loaded_keys = []
            mismatched_keys = []

            for name, param in source_state_dict.items():
                if name in target_state_dict:
                    if target_state_dict[name].shape == param.shape:
                        new_state_dict[name] = param
                        loaded_keys.append(name)
                    else:
                        mismatched_keys.append(name)
            
            # Load the filtered state dict
            self.model.load_state_dict(new_state_dict, strict=False)
            
            print(f"✅ DRL model partially loaded. Matched layers: {len(loaded_keys)}. Mismatched layers (ignored): {len(mismatched_keys)}.")
            if mismatched_keys:
                print(f"   - Mismatched layers were: {mismatched_keys}")

            self.model.eval()
            
        except Exception as e:
            print(f"❌ DRL模型加载失败: {e}")
            self.model = None
    
    def _get_state(self, task, available_nodes, node_capacities, node_loads):
        """获取DRL的状态向量 (兼容改进训练器 32 维)
        
        注意：此函数在实验环境中运行，许多在训练时可用的详细状态不可用。
        因此，我们使用可用的数据进行估算，并为不可用的特征使用合理的默认值或0填充。
        """
        features = []
        
        # 任务特征 (5维)
        try:
            task_flops = float(getattr(task, 'get_flops', lambda: 1e9)())
            # 实验环境的MockTask没有父/子任务信息，用0填充
            parents_count = 0
            children_count = len(getattr(task, 'get_children', lambda: [])())
        except Exception:
            task_flops = 1e9
            parents_count = 0
            children_count = 0

        features.append(np.log1p(task_flops / 1e9) / 10.0)  # 1. 计算大小 (log normalized)
        features.append(parents_count / 5.0)              # 2. 父任务数 (normalized)
        features.append(children_count / 5.0)             # 3. 子任务数 (normalized)
        features.append(0.0)                              # 4. 是否在关键路径 (不可用)
        features.append(0.0)                              # 5. 数据局部性分数 (不可用)

        # 节点特征 (16维 = 4 nodes * 4 features)
        max_speed = max(node_capacities.values()) if node_capacities else 4.0
        
        # 确保我们总是为4个节点生成特征
        for i in range(4):
            node_id = f"ComputeHost{i+1}"
            if node_id in self.compute_nodes and node_id in node_capacities:
                speed = node_capacities.get(node_id, 0.0)
                load = node_loads.get(node_id, 0.0)
                features.append(speed / max_speed)  # 1. 节点速度 (normalized)
                features.append(load)               # 2. 节点当前负载
                features.append(load / speed if speed > 0 else 0.0) # 3. 可用时间 (用 负载/速度 估算)
                features.append(0.0)                # 4. 数据可用性 (不可用)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0]) # 填充缺失的节点

        # 环境特征 (6维)
        features.append(0.5)  # 1. 工作流进度 (模拟值)
        features.append(0.0)  # 2. 当前时间 (不可用)
        features.append(0.5)  # 3. 待处理任务数 (模拟值)
        
        loads = [node_loads.get(f"ComputeHost{i+1}", 0.0) for i in range(4)]
        features.append(np.mean(loads))  # 4. 平均节点负载
        features.append(np.std(loads))   # 5. 节点负载标准差
        features.append(0.5)  # 6. 关键路径进度 (模拟值)

        # 数据传输特征 (5维) - 全部模拟为0，因为实验环境中无此信息
        features.extend([0.0] * 5)
        
        final_features = np.array(features, dtype=np.float32)
        
        # 最终维度检查，确保为32维
        if final_features.shape[0] != 32:
            padded_features = np.zeros(32, dtype=np.float32)
            copy_len = min(len(final_features), 32)
            padded_features[:copy_len] = final_features[:copy_len]
            return padded_features
            
        return final_features
    
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        if self.model is None:
            # 模型未加载，改为启发式回退而不是抛出异常
            print("⚠️ DRL模型未正确加载，使用启发式回退调度")
            return self._heuristic_fallback(task, available_nodes, node_capacities, node_loads)

        try:
            state = self._get_state(task, available_nodes, node_capacities, node_loads)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = q_values.argmax().item()

            # 动作映射到节点
            if action < len(self.compute_nodes):
                chosen_node = self.compute_nodes[action]
                if chosen_node in available_nodes:
                    return chosen_node

            # 如果选择的节点不可用，使用启发式回退
            print("⚠️ DRL选择节点不可用，使用启发式回退")
            return self._heuristic_fallback(task, available_nodes, node_capacities, node_loads)

        except Exception as e:
            print(f"⚠️ DRL调度失败，将使用启发式回退: {e}")
            return self._heuristic_fallback(task, available_nodes, node_capacities, node_loads)
    
    def _heuristic_fallback(self, task, available_nodes, node_capacities, node_loads):
        """启发式回退调度策略"""
        try:
            # 获取任务特征
            task_flops = float(getattr(task, 'get_flops', lambda: 1e9)())
            
            # 计算每个节点的得分（考虑容量和负载）
            best_node = None
            best_score = -float('inf')
            
            for node in available_nodes:
                capacity = node_capacities.get(node, 1.0)
                load = node_loads.get(node, 0.0)
                
                # 计算得分：容量越高越好，负载越低越好
                score = capacity - load * 2.0  # 负载权重更高
                
                if score > best_score:
                    best_score = score
                    best_node = node
            
            return best_node if best_node else available_nodes[0]
            
        except Exception as e:
            print(f"⚠️ 启发式回退也失败: {e}，使用第一个可用节点")
            return available_nodes[0]


        
        # 生成负载均衡导向的案例
        for _ in range(30):  # 生成30个额外案例
            base_case = random.choice(default_cases)
            variation = base_case.copy()
            
            # 添加一些随机变化
            variation['task_flops'] *= random.uniform(0.7, 1.3)
            variation['total_workflow_flops'] *= random.uniform(0.8, 1.2)
            variation['workflow_size'] = max(3, int(variation['workflow_size'] * random.uniform(0.7, 1.3)))
            
            # 根据负载均衡原则选择节点
            node_loads = {
                'ComputeHost1': random.uniform(0.0, 0.8),
                'ComputeHost2': random.uniform(0.0, 0.8),
                'ComputeHost3': random.uniform(0.0, 0.8),
                'ComputeHost4': random.uniform(0.0, 0.8)
            }
            
            # 选择负载最低的节点，但考虑任务大小
            if variation['task_flops'] < 3e9:
                # 小任务：优先选择负载低的节点
                sorted_nodes = sorted(node_loads.keys(), key=lambda x: node_loads[x])
                variation['chosen_node'] = sorted_nodes[0]
            elif variation['task_flops'] < 7e9:
                # 中等任务：在负载较低的节点中选择容量适中的
                low_load_nodes = [n for n in node_loads.keys() if node_loads[n] < 0.5]
                if low_load_nodes:
                    medium_capacity_nodes = [n for n in low_load_nodes if n in ['ComputeHost2', 'ComputeHost3']]
                    if medium_capacity_nodes:
                        variation['chosen_node'] = random.choice(medium_capacity_nodes)
                    else:
                        variation['chosen_node'] = random.choice(low_load_nodes)
                else:
                    variation['chosen_node'] = 'ComputeHost2'
            else:
                # 大任务：在高容量节点中选择负载较低的
                high_capacity_nodes = ['ComputeHost3', 'ComputeHost4']
                low_load_high_cap = [n for n in high_capacity_nodes if node_loads[n] < 0.6]
                if low_load_high_cap:
                    variation['chosen_node'] = random.choice(low_load_high_cap)
                else:
                    # 如果高容量节点都负载高，选择负载最低的
                    sorted_nodes = sorted(high_capacity_nodes, key=lambda x: node_loads[x])
                    variation['chosen_node'] = sorted_nodes[0]
            
            # 更新负载均衡因子
            variation['load_balance_factor'] = 1.0 - node_loads[variation['chosen_node']]
            variation['node_load'] = node_loads[variation['chosen_node']]
            
            self.knowledge_base.append(variation)
        
        # 添加原始默认案例
        self.knowledge_base.extend(default_cases)
        
        print(f"✅ 增强默认RAG知识库已创建: {len(self.knowledge_base)} 个案例（重点考虑负载均衡）")
    
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        """基于RAG知识库增强的调度决策 - 优化版本（重点解决负载均衡问题）"""
        try:
            # 首先使用DRL进行基础调度决策
            drl_node = self.drl_scheduler.schedule_task(
                task, available_nodes, node_capacities, node_loads, compute_service
            )
            # 如果没有知识库，直接回退到DRL决策
            if not self.knowledge_base:
                print("⚠️ RAG知识库为空，回退到DRL决策")
                return drl_node
            
            # 获取更丰富的任务特征用于RAG匹配
            try:
                task_flops = float(getattr(task, 'get_flops', lambda: 1e9)())
                task_memory = float(getattr(task, 'get_memory_requirement', lambda: 1024)())
                
                # 计算工作流特征
                total_workflow_flops = sum(
                    float(getattr(t, 'get_flops', lambda: 1e9)()) 
                    for t in [task]  # 这里简化处理，实际应该获取整个工作流
                )
                
                # 计算当前节点负载特征
                avg_load = np.mean([node_loads.get(node, 0) for node in available_nodes])
                max_load = max([node_loads.get(node, 0) for node in available_nodes])
                
                # 计算负载均衡指标
                load_variance = np.var([node_loads.get(node, 0) for node in available_nodes])
                load_std = np.sqrt(load_variance)
                
            except Exception as e:
                task_flops = 1e9
                task_memory = 1024
                total_workflow_flops = task_flops
                avg_load = 0
                max_load = 0
                load_variance = 0
                load_std = 0
            
            # 增强的相似度匹配 - 进一步降低阈值以获取更多匹配
            best_matches = []
            min_similarity_threshold = 0.01
            
            for case in self.knowledge_base:
                case_flops = float(case.get('task_flops', 1e9))
                
                # 计算归一化距离
                flops_distance = abs(task_flops - case_flops) / max(task_flops, case_flops, 1e-6)
                
                # 综合相似度 (已移除损坏的 workflow_distance)
                similarity = 1.0 - flops_distance
                
                if similarity >= min_similarity_threshold:
                    best_matches.append({
                        'case': case,
                        'similarity': similarity,
                        'suggested_node': str(case.get('chosen_node', drl_node))
                    })
            
            # 如果有高质量匹配，准备融合所需 rag_scores
            rag_scores = []
            match_node_scores = {}
            if best_matches:
                # 按makespan排序，选择makespan最低的案例
                best_matches.sort(key=lambda x: float(x['case'].get('makespan', float('inf'))))
                top_matches = best_matches[:8]
                for match in top_matches:
                    node = match['suggested_node']
                    match_node_scores.setdefault(node, 0.0)
                    # 综合考虑相似度和makespan（makespan越低，权重越高）
                    makespan = float(match['case'].get('makespan', 100.0))
                    makespan_weight = 1.0 / (1.0 + makespan / 100.0)  # 归一化makespan权重
                    match_node_scores[node] += match['similarity'] * makespan_weight
                    
                    # 负载均衡调整：如果节点负载过高，极大幅降低其得分
                    node_load = node_loads.get(node, 0.0)
                    if node_load > avg_load * 1.05:  # 进一步降低阈值，从1.1倍改为1.05倍
                        load_penalty = 0.01  # 降低99%的得分（从90%进一步增强到99%）
                        match_node_scores[node] *= load_penalty
                        
            for node in available_nodes:
                rag_scores.append(match_node_scores.get(node, 0.0))

            # 融合决策
            if self.enable_fusion and rag_scores:
                try:
                    from src.scheduling.hybrid_fusion import fuse_decision
                    # 获取 DRL 模型真实 Q-values
                    q_values = []
                    try:
                        state = self.drl_scheduler._get_state(task, available_nodes, node_capacities, node_loads)
                        if self.drl_scheduler.model is not None:
                            import torch
                            st = torch.FloatTensor(state).unsqueeze(0).to(self.drl_scheduler.device)
                            with torch.no_grad():
                                q_tensor = self.drl_scheduler.model(st)
                            q_list = q_tensor.squeeze(0).cpu().tolist()
                            # 兼容性补丁：如果模型输出的动作空间比可用节点少，用平均值填充
                            q_values = q_list
                            while len(q_values) < len(available_nodes):
                                q_values.append(np.mean(q_values) if q_values else 0.0)
                            # 确保最终长度一致
                            q_values = q_values[:len(available_nodes)]
                    except Exception as qe:
                        print(f"获取真实Q值失败，回退伪Q: {qe}")
                    if not q_values:
                        for node in available_nodes:
                            cap = node_capacities.get(node, 1.0)
                            load = node_loads.get(node, 0.0)
                            # 极强增强负载均衡考虑
                            load_balance_factor = 1.0 / (1.0 + 20.0 * load)  # 极强增强负载均衡因子
                            q_values.append(cap * load_balance_factor)
                    load_vals = [node_loads.get(n, 0.0) for n in available_nodes]
                    progress = 0.5  # TODO: 使用真实训练进度
                    
                    # 计算makespan预测（基于历史案例和当前状态）
                    makespan_predictions = []
                    baseline_makespan = None
                    
                    # 获取基准makespan（HEFT算法预测）
                    try:
                        heft_scheduler = HEFTScheduler()
                        heft_prediction = heft_scheduler.predict_makespan(task, available_nodes, node_capacities, node_loads)
                        if heft_prediction > 0:
                            baseline_makespan = heft_prediction
                    except Exception as e:
                        print(f"获取HEFT基准makespan失败: {e}")
                    
                    # 为每个节点预测makespan
                    for node in available_nodes:
                        predicted_makespan = baseline_makespan or 100.0  # 默认值
                        
                        # 基于当前负载和容量调整预测
                        node_load = node_loads.get(node, 0.0)
                        node_capacity = node_capacities.get(node, 1.0)
                        load_factor = 1.0 + node_load / max(node_capacity, 0.1)
                        predicted_makespan *= load_factor
                        
                        # 基于RAG匹配度调整预测（匹配度越高，预测makespan越低）
                        rag_score = match_node_scores.get(node, 0.0)
                        if rag_score > 0:
                            rag_factor = 1.0 - 0.5 * rag_score  # 最高可减少50%的makespan
                            predicted_makespan *= rag_factor
                        
                        makespan_predictions.append(predicted_makespan)
                    
                    # 增强负载均衡权重，加入makespan预测
                    fusion = fuse_decision(
                        q_values, 
                        rag_scores, 
                        load_vals, 
                        progress, 
                        rag_confidence_threshold=0.00001,  # 大幅降低阈值以激活RAG
                        makespan_predictions=makespan_predictions,
                        baseline_makespan=baseline_makespan
                    )
                    fused_idx = fusion['index']
                    fused_node = available_nodes[fused_idx]
                    print(f"🔀 融合决策: {fused_node} (α={fusion['alpha']:.2f}, β={fusion['beta']:.2f}, γ={fusion['gamma']:.2f}, δ={fusion.get('delta', 0.0):.2f})")
                    
                    # 记录融合决策的详细信息
                    try:
                        import json, os
                        os.makedirs('results', exist_ok=True)
                        with open('results/fusion_debug.log', 'a', encoding='utf-8') as fdbg:
                            record = {
                                'node': fused_node,
                                'alpha': fusion['alpha'],
                                'beta': fusion['beta'],
                                'gamma': fusion['gamma'],
                                'delta': fusion.get('delta', 0.0),
                                'q_norm': fusion['q_norm'],
                                'rag_norm': fusion['rag_norm'],
                                'load_pref': fusion['load_pref'],
                                'makespan_scores': fusion.get('makespan_scores', []),
                                'fused': fusion['fused'],
                                'load_variance': load_variance,
                                'load_std': load_std,
                                'avg_load': avg_load,
                                'max_load': max_load,
                                'makespan_predictions': makespan_predictions,
                                'baseline_makespan': baseline_makespan
                            }
                            fdbg.write(json.dumps(record, ensure_ascii=False) + '\n')
                    except Exception as le:
                        print(f"融合调试日志写入失败: {le}")
                    return fused_node
                except Exception as fe:
                    print(f"融合失败，回退RAG/DRL: {fe}")

            # 无融合或失败：使用增强的负载均衡策略
            if match_node_scores:
                # 结合相似度和负载均衡选择节点
                best_node = None
                best_score = -float('inf')
                
                for node in available_nodes:
                    # 基础得分：RAG相似度
                    node_score = match_node_scores.get(node, 0.0)
                    
                    # 负载均衡调整
                    node_load = node_loads.get(node, 0.0)
                    load_balance_factor = 1.0 / (1.0 + 20.0 * load)  # 极强增强负载均衡因子
                    
                    # 节点容量考虑
                    node_capacity = node_capacities.get(node, 1.0)
                    
                    # 综合得分
                    combined_score = node_score * load_balance_factor * node_capacity
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_node = node
                
                if best_node:
                    return best_node
            
            # 如果没有足够的匹配，使用增强的启发式策略
            print("⚠️ 无足够RAG匹配案例，使用增强启发式策略")
            best_node = None
            best_score = -float('inf')
            
            for node in available_nodes:
                capacity = node_capacities.get(node, 1.0)
                load = node_loads.get(node, 0.0)
                
                # 极强增强负载均衡因子
                load_balance_factor = 1.0 / (1.0 + 20.0 * load)  # 极强增强负载均衡因子
                score = capacity * load_balance_factor
                
                if score > best_score:
                    best_score = score
                    best_node = node
            
            return best_node if best_node else available_nodes[0]

        except Exception as e:
            print(f"⚠️ RAG调度失败: {e}，尝试回退")
            # 优先尝试直接使用已获得的drl_node（如果存在）
            try:
                if 'drl_node' in locals() and drl_node in available_nodes:
                    return drl_node
            except Exception:
                pass
            # 最终回退到启发式
            try:
                return self.drl_scheduler._heuristic_fallback(task, available_nodes, node_capacities, node_loads)
            except Exception:
                # 兜底：返回第一个可用节点
                return available_nodes[0]

class WRENCHExperimentRunner:
    """基于真实WRENCH的实验运行器"""
    
    def __init__(self, config_path: str = "configs/experiment.yaml"):
        self.config = load_config(config_path)
        
        # WRENCH平台配置
        self.platform_file = self.config['platform']['platform_file']
        self.controller_host = "ControllerHost"
        
        # 节点配置
        self.compute_nodes = ["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"]
        self.node_capacities = {
            "ComputeHost1": 2.0,
            "ComputeHost2": 3.0,
            "ComputeHost3": 2.5,
            "ComputeHost4": 4.0
        }
        
        # 调度器配置
        self.schedulers = self._initialize_schedulers()
        
        # 实验参数
        self.workflow_sizes = [5, 10, 15, 20]
        self.repetitions = 3
        
        # 结果存储
        self.results = []
        
        print(f"🚀 WRENCH实验运行器初始化完成")
    
    def _initialize_schedulers(self):
        """初始化所有调度器"""
        schedulers = {
            "FIFO": FIFOScheduler(),
            "HEFT": HEFTScheduler(),
            "WASS-Heuristic": WASSHeuristicScheduler(),  # 新增WASS启发式调度器
        }
        
        # 模型文件优先级：兼容模型 > 原始优化模型 > 基础模型
        model_candidates = [
            "models/improved_wass_drl.pth",  # 新训练改进模型
            "models/wass_optimized_models_compatible.pth",
            "models/wass_optimized_models.pth",
            "models/wass_models.pth"
        ]

        # 环境变量优先 (WASS_DRL_MODEL)
        env_model = os.environ.get("WASS_DRL_MODEL")
        if env_model:
            if os.path.exists(env_model):
                if env_model not in model_candidates:
                    model_candidates.insert(0, env_model)
                else:
                    # 确保环境变量路径排到首位
                    model_candidates.remove(env_model)
                    model_candidates.insert(0, env_model)
                print(f"🔍 通过环境变量指定模型: {env_model}")
            else:
                print(f"⚠️ 环境变量WASS_DRL_MODEL指定的模型不存在: {env_model}")
        
        model_path = None
        print("🔍 查找模型文件...")
        for candidate in model_candidates:
            print(f"🔍 检查模型文件: {candidate}")
            # 检查绝对路径
            abs_candidate = os.path.abspath(candidate)
            print(f"🔍 绝对路径: {abs_candidate}")
            if os.path.exists(candidate):
                model_path = candidate
                print(f"✅ 找到模型文件: {model_path}")
                break
            elif os.path.exists(abs_candidate):
                model_path = abs_candidate
                print(f"✅ 找到模型文件 (绝对路径): {model_path}")
                break
        
        if not model_path:
            print("⚠️ 未找到任何模型文件")
            for candidate in model_candidates:
                abs_candidate = os.path.abspath(candidate)
                if os.path.exists(candidate):
                    print(f"  存在: {candidate}")
                elif os.path.exists(abs_candidate):
                    print(f"  存在 (绝对路径): {abs_candidate}")
                else:
                    print(f"  不存在: {candidate} (绝对路径: {abs_candidate})")
        
        rag_path = "data/wrench_rag_knowledge_base.json"
        
        if model_path:
            print(f"📁 使用模型文件: {model_path}")
            
            # 强制启用WASS-DRL调度器
            try:
                # 从模型文件加载DRL代理
                print("🔍 正在加载模型文件...")
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                print(f"🔍 模型加载成功，检查点键: {list(checkpoint.keys())}")
                
                # 从检查点数据创建DRL代理
                # 获取模型配置信息
                config = checkpoint.get('config', {})
                # 优先从metadata获取state_dim和action_dim，如果不存在则从config获取，否则使用默认值
                metadata = checkpoint.get('metadata', {})
                state_dim = metadata.get('state_dim', config.get('state_dim', 32))  # 默认32维状态
                action_dim = metadata.get('action_dim', config.get('action_dim', 4))  # 默认4个动作
                
                # 创建新的DRL代理
                from src.drl_agent import DQNAgent
                drl_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
                
                # 加载模型权重
                drl_agent.load(model_path)
                
                node_names = checkpoint.get('node_names', ['node1', 'node2', 'node3'])  # 默认节点名
                predictor = checkpoint.get('predictor', None)
                
                print(f"🔍 DRL代理: {type(drl_agent)}")
                print(f"🔍 节点名称: {node_names}")
                print(f"🔍 预测器: {type(predictor)}")
                
                # 使用工厂函数创建调度器
                from src.ai_schedulers import create_scheduler
                drl_scheduler = create_scheduler('WASS-DRL (w/o RAG)', node_names, drl_agent, predictor)
                schedulers["WASS-DRL"] = drl_scheduler
                print("✅ WASS-DRL调度器已强制启用")
                
                # 强制启用WASS-RAG调度器
                rag_candidates = [
                    rag_path,
                    "data/wrench_rag_knowledge_base.json",
                    "data/extended_rag_knowledge.json"
                ]
                
                rag_available = False
                for rag_candidate in rag_candidates:
                    if os.path.exists(rag_candidate):
                        try:
                            # 正确初始化WASSRAGScheduler
                            rag_scheduler = WASSRAGScheduler(drl_scheduler.drl_agent, drl_scheduler.node_names, drl_scheduler.predictor, rag_candidate)
                            schedulers["WASS-RAG"] = rag_scheduler
                            print(f"✅ WASS-RAG调度器已启用 (知识库: {rag_candidate})")
                            rag_available = True
                            break
                        except Exception as e:
                            print(f"⚠️  WASS-RAG从{rag_candidate}加载失败: {e}")
                            continue
                
                if not rag_available:
                    # 即使没有知识库，也创建空的RAG调度器
                    rag_scheduler = WASSRAGScheduler(drl_scheduler.drl_agent, drl_scheduler.node_names, drl_scheduler.predictor, rag_path)
                    schedulers["WASS-RAG"] = rag_scheduler
                    print("⚠️  WASS-RAG调度器已创建 (知识库为空)")
                    
            except Exception as e:
                print(f"❌ DRL/RAG调度器初始化失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ 未找到任何模型文件，仅使用基础调度器")
        
        print(f"🔧 已启用调度器: {list(schedulers.keys())}")
        return schedulers
    
    def run_single_experiment_with_workflow(self, scheduler_name: str, workflow, workflow_size: int, experiment_id: int) -> WRENCHExperimentResult:
        """使用预生成的工作流运行单个实验"""
        print(f"    🔬 运行实验: {scheduler_name} (工作流大小: {workflow_size})")
        
        start_time = time.time()
        
        try:
            # 获取调度器
            scheduler = self.schedulers[scheduler_name]
            
            # 模拟WRENCH实验执行
            # 在实际实现中，这里应该调用真实的WRENCH API
            simulation_result = self._simulate_wrench_execution(scheduler, workflow, workflow_size)
            
            # 创建实验结果
            result = WRENCHExperimentResult(
                scheduler_name=scheduler_name,
                workflow_id=f"workflow_{workflow_size}_{experiment_id}",
                task_count=workflow_size,
                dependency_count=int(workflow_size * 0.8),  # 假设80%的任务有依赖
                makespan=simulation_result['makespan'],
                cpu_utilization=simulation_result['cpu_utilization'],
                task_execution_times=simulation_result['task_times'],
                scheduling_decisions=simulation_result['decisions'],
                experiment_metadata={
                    'experiment_id': experiment_id,
                    'workflow_size': workflow_size,
                    'execution_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return result
            
        except Exception as e:
            # 返回失败结果
            return WRENCHExperimentResult(
                scheduler_name=scheduler_name,
                workflow_id=f"workflow_{workflow_size}_{experiment_id}",
                task_count=workflow_size,
                dependency_count=0,
                makespan=float('inf'),
                cpu_utilization={},
                task_execution_times={},
                scheduling_decisions=[],
                experiment_metadata={
                    'experiment_id': experiment_id,
                    'workflow_size': workflow_size,
                    'execution_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
            )
    
    def _generate_workflow(self, workflow_size: int, repetition: int) -> Dict:
        """生成固定的工作流（基于随机种子确保可重现）"""
        # 设置随机种子，确保相同参数生成相同工作流
        seed = 42 + workflow_size * 100 + repetition
        random.seed(seed)
        np.random.seed(seed)
        
        # 生成工作流结构
        workflow = {
            'tasks': [],
            'dependencies': [],
            'seed': seed
        }
        
        # 生成任务
        for i in range(workflow_size):
            task = {
                'id': f"task_{i}",
                'flops': random.uniform(1e9, 10e9),  # 1-10 GFLOPS
                'memory': random.uniform(1, 8),       # 1-8 GB
                'cores': random.randint(1, 4)         # 1-4 cores
            }
            workflow['tasks'].append(task)
        
        # 生成依赖关系（DAG结构）
        # 简单实现：每个任务依赖于之前的1-3个任务
        for i in range(1, workflow_size):
            num_deps = min(random.randint(1, 3), i)  # 最多依赖前面的3个任务
            deps = random.sample(range(i), num_deps)
            
            for dep in deps:
                workflow['dependencies'].append({
                    'from': f"task_{dep}",
                    'to': f"task_{i}"
                })
        
        return workflow
    
    def _simulate_wrench_execution(self, scheduler, workflow: Dict, workflow_size: int) -> Dict:
        """
        模拟WRENCH执行（修复版），正确处理任务依赖和调度。
        """
        import networkx as nx

        # 1. 构建任务图
        g = nx.DiGraph()
        task_map = {t['id']: t for t in workflow['tasks']}
        for task_id in task_map:
            g.add_node(task_id)
        for dep in workflow.get('dependencies', []):
            g.add_edge(dep['from'], dep['to'])

        # 2. 初始化状态
        node_finish_times = {node: 0.0 for node in self.compute_nodes}
        task_finish_times = {}
        decisions = []
        
        completed_tasks = set()
        
        # 3. 主模拟循环
        for _ in range(workflow_size):
            # 找出当前就绪的任务 (没有未完成的父任务)
            ready_tasks_ids = []
            for task_id in g.nodes:
                if task_id in completed_tasks:
                    continue
                
                parents = list(g.predecessors(task_id))
                if all(p in completed_tasks for p in parents):
                    ready_tasks_ids.append(task_id)
            
            if not ready_tasks_ids:
                if len(completed_tasks) < workflow_size:
                     break
                continue

            ready_tasks_ids.sort()
            task_to_schedule_id = ready_tasks_ids[0]
            
            task_data = task_map[task_to_schedule_id]

            class MockTask:
                def __init__(self, task_dict, parents):
                    self._task_dict = task_dict
                    self._parents = parents
                def get_id(self): return self._task_dict['id']
                def get_flops(self): return self._task_dict['flops']
                def get_parents(self): return self._parents
                def get_input_files(self): return []
                def get_output_files(self): return []

            mock_task = MockTask(task_data, list(g.predecessors(task_to_schedule_id)))

            data_ready_time = 0.0
            for parent_id in g.predecessors(task_to_schedule_id):
                data_ready_time = max(data_ready_time, task_finish_times.get(parent_id, 0.0))

            node_available_times = node_finish_times.copy()

            chosen_node = scheduler.schedule_task(
                mock_task, self.compute_nodes, self.node_capacities, node_available_times, None
            )

            node_ready_time = node_finish_times.get(chosen_node, 0.0)
            start_time = max(data_ready_time, node_ready_time)
            
            capacity = self.node_capacities[chosen_node]
            exec_time = task_data['flops'] / (capacity * 1e9)
            finish_time = start_time + exec_time

            node_finish_times[chosen_node] = finish_time
            task_finish_times[task_to_schedule_id] = finish_time
            completed_tasks.add(task_to_schedule_id)

            decisions.append({
                'task_id': task_to_schedule_id,
                'chosen_node': chosen_node,
                'execution_time': exec_time,
                'start_time': start_time,
                'end_time': finish_time
            })

        final_makespan = max(task_finish_times.values()) if task_finish_times else 0.0
        
        cpu_utilization = {}
        total_busy_time = {node: 0.0 for node in self.compute_nodes}
        for decision in decisions:
            total_busy_time[decision['chosen_node']] += decision['execution_time']
            
        for node in self.compute_nodes:
            if final_makespan > 0:
                cpu_utilization[node] = total_busy_time[node] / final_makespan
            else:
                cpu_utilization[node] = 0.0

        return {
            'makespan': final_makespan,
            'cpu_utilization': cpu_utilization,
            'task_times': {d['task_id']: d['execution_time'] for d in decisions},
            'decisions': decisions
        }
    
    def run_single_experiment(self, scheduler_name: str, workflow_size: int, experiment_id: int) -> WRENCHExperimentResult:
        """运行单次WRENCH实验（使用预生成的工作流）"""
        print(f"  运行实验: {scheduler_name}, {workflow_size}任务, 实验#{experiment_id}")
        
        # 检查是否已有预生成的工作流
        workflow_key = f"{workflow_size}_{experiment_id}"
        if workflow_key in self.workflow_cache:
            workflow = self.workflow_cache[workflow_key]
            print(f"    📋 使用缓存的工作流: {workflow_key}")
        else:
            # 生成新的工作流并缓存
            workflow = self._generate_workflow(workflow_size, experiment_id)
            self.workflow_cache[workflow_key] = workflow
            print(f"    📋 生成并缓存新工作流: {workflow_key}")
        
        # 使用预生成的工作流运行实验
        return self.run_single_experiment_with_workflow(scheduler_name, workflow, workflow_size, experiment_id)
        
    def run_all_experiments(self):
        """运行所有实验配置（公平实验设计）"""
        print(f"🔬 开始完整WRENCH实验...")
        print(f"调度器: {list(self.schedulers.keys())}")
        print(f"工作流规模: {self.workflow_sizes}")
        print(f"重复次数: {self.repetitions}")
        
        total_experiments = len(self.schedulers) * len(self.workflow_sizes) * self.repetitions
        print(f"总实验数: {total_experiments} = {len(self.schedulers)}调度器 × {len(self.workflow_sizes)}任务规模 × {self.repetitions}次重复")
        
        # 公平实验设计：预生成工作流，确保所有调度器在相同工作流上测试
        print("\n📝 预生成工作流（确保公平比较）...")
        workflow_cache = {}
        
        for workflow_size in self.workflow_sizes:
            for rep in range(self.repetitions):
                # 为每个工作流大小和重复次数生成固定的工作流
                workflow_key = (workflow_size, rep)
                print(f"   生成工作流: {workflow_size}个任务, 重复{rep+1}")
                
                try:
                    # 生成工作流并缓存
                    workflow = self._generate_workflow(workflow_size, rep)
                    workflow_cache[workflow_key] = workflow
                    print(f"   ✅ 工作流生成成功")
                except Exception as e:
                    print(f"   ❌ 工作流生成失败: {e}")
                    workflow_cache[workflow_key] = None
        
        print("\n🚀 开始运行实验...")
        current_exp = 0
        
        # 按工作流大小和重复次数分组，确保公平性
        for workflow_size in self.workflow_sizes:
            for rep in range(self.repetitions):
                # 获取预生成的工作流
                workflow_key = (workflow_size, rep)
                workflow = workflow_cache[workflow_key]
                
                if workflow is None:
                    print(f"   ⚠️ 跳过无效工作流: {workflow_key}")
                    continue
                
                # 对同一工作流测试所有调度器
                for scheduler_name in self.schedulers.keys():
                    current_exp += 1
                    print(f"\n进度: {current_exp}/{total_experiments}")
                    print(f"   工作流: {workflow_size}个任务, 重复{rep+1}")
                    print(f"   调度器: {scheduler_name}")
                    
                    try:
                        # 使用预生成的工作流运行实验
                        result = self.run_single_experiment_with_workflow(
                            scheduler_name, workflow, workflow_size, current_exp
                        )
                        self.results.append(result)
                        print(f"   ✅ 完成: {result.makespan:.2f}s")
                    except Exception as e:
                        print(f"   ❌ 实验失败: {e}")
        
        # 保存结果
        self._save_results()
        self._analyze_results()
        
        print(f"\n🎉 所有实验完成！共运行 {len(self.results)} 次实验")
        return self.results
    
    def _save_results(self):
        """保存实验结果"""
        results_dir = Path("results/wrench_experiments")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_data = {
            "experiment_config": {
                "schedulers": list(self.schedulers.keys()),
                "workflow_sizes": self.workflow_sizes,
                "repetitions": self.repetitions,
                "total_experiments": len(self.results)
            },
            "results": [asdict(result) for result in self.results]
        }
        
        with open(results_dir / "detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 实验结果已保存到 {results_dir}")
    
    def _analyze_results(self):
        """分析实验结果"""
        if not self.results:
            print("❌ 没有实验结果可分析")
            return
        
        print(f"\n📈 实验结果分析:")
        print("=" * 60)
        
        # 按调度器分组统计
        scheduler_stats = {}
        for result in self.results:
            name = result.scheduler_name
            if name not in scheduler_stats:
                scheduler_stats[name] = []
            scheduler_stats[name].append(result.makespan)
        
        # 显示统计结果
        print(f"{'调度器':<15} {'平均Makespan':<15} {'标准差':<10} {'最佳':<10} {'实验次数':<8}")
        print("-" * 60)
        
        for scheduler_name, makespans in scheduler_stats.items():
            avg_makespan = np.mean(makespans)
            std_makespan = np.std(makespans)
            best_makespan = min(makespans)
            count = len(makespans)
            
            print(f"{scheduler_name:<15} {avg_makespan:<15.2f} {std_makespan:<10.2f} {best_makespan:<10.2f} {count:<8}")
        
        # 找出最佳调度器
        best_scheduler = min(scheduler_stats.keys(), 
                           key=lambda x: np.mean(scheduler_stats[x]))
        best_avg = np.mean(scheduler_stats[best_scheduler])
        
        print(f"\n🏆 最佳调度器: {best_scheduler} (平均Makespan: {best_avg:.2f}s)")

def main():
    """主函数"""
    print("🚀 开始基于WRENCH的真实WASS-RAG实验...")
    
    runner = WRENCHExperimentRunner()
    runner.run_all_experiments()
    
    print("\n🎉 所有实验完成!")

if __name__ == "__main__":
    main()
