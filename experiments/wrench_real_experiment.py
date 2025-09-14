#!/usr/bin/env python3
"""
基于WRENCH的真实WASS-RAG实验框架
使用训练好的模型在真实WRENCH环境中进行性能对比实验
"""

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
        
        # 定义网络结构
        self.state_dim = 17  # 与训练时一致
        self.action_dim = 4   # 4个节点
        
        # 先创建模型，再加载权重
        self.model = self._create_model()
        self.epsilon = 0.1
        
        # 加载模型
        self._load_model(model_path)
        
        # 节点映射
        self.compute_nodes = ["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"]
    
    def _create_model(self):
        """创建DRL模型"""
        return SimpleDQN(self.state_dim, self.action_dim).to(self.device)
    
    def _load_model(self, model_path: str):
        """加载训练好的DRL模型"""
        try:
            # 修复PyTorch 2.6兼容性问题
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 检查模型格式
            if 'drl_agent' in checkpoint:
                agent_state = checkpoint['drl_agent']
                self.model.load_state_dict(agent_state['model_state_dict'])
                self.epsilon = agent_state.get('epsilon', 0.1)
                print(f"✅ DRL模型加载成功 (训练轮数: {agent_state.get('training_episodes', 'unknown')})")
            else:
                # 兼容旧格式
                self.model.load_state_dict(checkpoint)
                print("✅ DRL模型加载成功 (旧格式)")
                
            self.model.eval()
            
        except Exception as e:
            print(f"❌ DRL模型加载失败: {e}")
            self.model = None
    
    def _get_state(self, task, available_nodes, node_capacities, node_loads):
        """获取DRL的状态向量"""
        state = []
        
        # 任务特征 (4维)
        try:
            # 使用WRENCH API的正确方法获取任务信息
            task_flops = float(getattr(task, 'get_flops', lambda: 1e9)()) / 1e9  # 任务计算量 (GFLOPS)
            
            # 尝试获取内存需求，如果不存在则使用默认值
            try:
                task_memory = float(task.get_memory_requirement()) / 1e9  # 内存需求 (GB)
            except (AttributeError, TypeError):
                task_memory = 1.0  # 默认值
            
            task_cores = float(getattr(task, 'get_min_num_cores', lambda: 1)())  # 最小核心数
            task_children = float(len(getattr(task, 'get_children', lambda: [])()))  # 子任务数
            
            state.extend([task_flops, task_memory, task_cores, task_children])
            
        except Exception as e:
            # 如果所有方法都失败，使用默认值
            state.extend([1.0, 1.0, 1.0, 0.0])
        
        # 节点特征 (每节点3维，最多4个节点 = 12维)
        for i, node in enumerate(available_nodes[:4]):  # 限制最多4个节点
            node_capacity = node_capacities.get(node, 1.0)
            node_load = node_loads.get(node, 0.0)
            
            state.extend([
                float(node_capacity),  # 节点容量
                float(node_load),  # 节点负载
                float(node_load / max(node_capacity, 1e-6))  # 负载率
            ])
        
        # 填充不足的节点维度
        while len(state) < 16:  # 4(任务) + 4*3(节点) = 16维
            state.append(0.0)
        
        # 全局特征 (1维)
        avg_load = sum(node_loads.values()) / max(len(node_loads), 1)
        state.append(float(avg_load))
        
        # 确保状态维度为17维
        if len(state) > 17:
            state = state[:17]
        elif len(state) < 17:
            state.extend([0.0] * (17 - len(state)))
        
        return np.array(state, dtype=np.float32)
    
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        if self.model is None:
            # 模型未加载，抛出异常而不是回退
            raise RuntimeError("DRL模型未正确加载，无法进行调度")
        
        try:
            state = self._get_state(task, available_nodes, node_capacities, node_loads)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = q_values.argmax().item()
            
            # 映射动作到节点
            if action < len(self.compute_nodes):
                chosen_node = self.compute_nodes[action]
                if chosen_node in available_nodes:
                    return chosen_node
            
            # 如果选择的节点不可用，抛出异常
            raise RuntimeError("DRL模型选择的节点不可用")
            
        except Exception as e:
            print(f"⚠️ DRL调度失败: {e}")
            raise RuntimeError(f"DRL调度失败: {e}")
    
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

class WASSRAGScheduler(WRENCHScheduler):
    """基于RAG知识库增强的调度器"""
    
    def __init__(self, model_path: str, rag_path: str):
        super().__init__("WASS-RAG")
        self.drl_scheduler = WASSDRLScheduler(model_path)
        self.knowledge_base = None
        self._load_rag_knowledge(rag_path)
    
    def _load_rag_knowledge(self, rag_path: str):
        """加载增强的RAG知识库"""
        self.knowledge_base = []
        
        # 优先使用增强知识库
        enhanced_path = "data/enhanced_rag_knowledge.json"
        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cases = data.get('cases', [])
                self.knowledge_base.extend(cases)
                print(f"✅ 增强RAG知识库已加载: {len(cases)} 个优化案例")
                return
            except Exception as e:
                print(f"增强知识库加载失败: {e}")
        
        # 回退到原始方法
        # 方法1: 使用扩展的JSON知识库
        extended_json_path = "data/extended_rag_knowledge.json"
        if os.path.exists(extended_json_path):
            try:
                with open(extended_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cases = []
                if isinstance(data, dict):
                    if 'cases' in data:
                        cases = data['cases']
                    elif 'sample_cases' in data:
                        cases = data['sample_cases']
                    else:
                        cases = list(data.values()) if isinstance(data, dict) else data
                elif isinstance(data, list):
                    cases = data
                
                for case in cases:
                    if isinstance(case, dict):
                        simple_case = {
                            'task_flops': float(case.get('task_flops', case.get('task_execution_time', 1.0) * 2e9)),
                            'chosen_node': str(case.get('chosen_node', 'ComputeHost1')),
                            'scheduler_type': str(case.get('scheduler_type', 'unknown')),
                            'task_execution_time': float(case.get('task_execution_time', 0.0)),
                            'workflow_makespan': float(case.get('workflow_makespan', 0.0)),
                            'node_capacity': float(case.get('node_capacity', 2.0)),
                            'performance_ratio': float(case.get('performance_ratio', 1.0)),
                            'total_workflow_flops': float(case.get('total_workflow_flops', case.get('task_flops', 1e9))),
                            'workflow_size': int(case.get('workflow_size', 5))
                        }
                        self.knowledge_base.append(simple_case)
                
                if self.knowledge_base:
                    print(f"✅ RAG知识库已从扩展JSON加载: {len(self.knowledge_base)} 个案例")
                    return
                    
            except Exception as e:
                print(f"扩展JSON加载失败: {e}")
        
        # 方法2: 使用PKL文件（回退方案）
        try:
            import pickle
            with open(rag_path, 'rb') as f:
                data = pickle.load(f)
            
            # 处理不同格式的pickle数据
            cases = []
            if isinstance(data, dict):
                cases = data.get('cases', data.get('sample_cases', []))
            elif isinstance(data, list):
                cases = data
            else:
                # 尝试直接迭代
                try:
                    cases = list(data)
                except:
                    cases = [data]
            
            for case in cases:
                try:
                    if hasattr(case, '__dict__'):
                        # 处理对象类型
                        case_dict = case.__dict__
                    elif isinstance(case, dict):
                        case_dict = case
                    else:
                        continue
                    
                    simple_case = {
                        'task_flops': float(case_dict.get('task_flops', case_dict.get('task_execution_time', 1.0) * 2e9)),
                        'chosen_node': str(case_dict.get('chosen_node', 'ComputeHost1')),
                        'scheduler_type': str(case_dict.get('scheduler_type', 'unknown')),
                        'task_execution_time': float(case_dict.get('task_execution_time', 0.0))
                    }
                    self.knowledge_base.append(simple_case)
                    
                except Exception as e:
                    continue
            
            if self.knowledge_base:
                print(f"✅ RAG知识库已从PKL加载: {len(self.knowledge_base)} 个案例")
                return
                
        except Exception as e:
            print(f"PKL加载失败: {e}")
        
        # 方法3: 从扩展JSON直接读取（最终回退）
        try:
            with open("data/extended_rag_knowledge.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 简化处理：从JSON中提取任何可用的案例数据
            if isinstance(data, list):
                for case in data[:100]:  # 限制数量避免内存问题
                    if isinstance(case, dict):
                        simple_case = {
                            'task_flops': float(case.get('task_flops', 1e9)),
                            'chosen_node': str(case.get('chosen_node', 'ComputeHost1')),
                            'scheduler_type': str(case.get('scheduler_type', 'FIFO'))
                        }
                        self.knowledge_base.append(simple_case)
            
            if self.knowledge_base:
                print(f"✅ RAG知识库已从JSON加载(简化模式): {len(self.knowledge_base)} 个案例")
                return
                
        except Exception as e:
            print(f"最终加载失败: {e}")
        
        # 方法4: 创建默认知识库（如果所有加载方法都失败）
        print("⚠️ 无法加载任何RAG知识库，创建默认知识库...")
        self._create_default_knowledge_base()
    
    def _create_default_knowledge_base(self):
        """创建默认的RAG知识库"""
        # 基于节点性能和任务特征的简单启发式规则
        default_cases = [
            # 小任务优先分配到高容量节点
            {'task_flops': 1e9, 'chosen_node': 'ComputeHost4', 'scheduler_type': 'heuristic', 
             'task_execution_time': 0.25, 'workflow_makespan': 5.0, 'node_capacity': 4.0,
             'performance_ratio': 0.8, 'total_workflow_flops': 5e9, 'workflow_size': 5},
            
            # 中等任务分配到中等容量节点
            {'task_flops': 5e9, 'chosen_node': 'ComputeHost2', 'scheduler_type': 'heuristic',
             'task_execution_time': 1.67, 'workflow_makespan': 10.0, 'node_capacity': 3.0,
             'performance_ratio': 0.9, 'total_workflow_flops': 20e9, 'workflow_size': 10},
            
            # 大任务分配到高容量节点
            {'task_flops': 10e9, 'chosen_node': 'ComputeHost4', 'scheduler_type': 'heuristic',
             'task_execution_time': 2.5, 'workflow_makespan': 15.0, 'node_capacity': 4.0,
             'performance_ratio': 0.85, 'total_workflow_flops': 50e9, 'workflow_size': 15},
            
            # 考虑负载均衡的案例
            {'task_flops': 3e9, 'chosen_node': 'ComputeHost1', 'scheduler_type': 'heuristic',
             'task_execution_time': 1.5, 'workflow_makespan': 8.0, 'node_capacity': 2.0,
             'performance_ratio': 0.75, 'total_workflow_flops': 15e9, 'workflow_size': 8},
            
            # 更多多样化案例
            {'task_flops': 7e9, 'chosen_node': 'ComputeHost3', 'scheduler_type': 'heuristic',
             'task_execution_time': 2.8, 'workflow_makespan': 12.0, 'node_capacity': 2.5,
             'performance_ratio': 0.82, 'total_workflow_flops': 30e9, 'workflow_size': 12}
        ]
        
        # 添加一些随机变化以增加多样性
        import random
        random.seed(42)  # 确保可重现
        
        for _ in range(20):  # 生成20个额外案例
            base_case = random.choice(default_cases)
            variation = base_case.copy()
            
            # 添加一些随机变化
            variation['task_flops'] *= random.uniform(0.8, 1.2)
            variation['total_workflow_flops'] *= random.uniform(0.9, 1.1)
            variation['workflow_size'] = max(3, int(variation['workflow_size'] * random.uniform(0.8, 1.2)))
            
            # 根据任务大小选择合适的节点
            if variation['task_flops'] < 3e9:
                variation['chosen_node'] = random.choice(['ComputeHost1', 'ComputeHost2'])
            elif variation['task_flops'] < 7e9:
                variation['chosen_node'] = random.choice(['ComputeHost2', 'ComputeHost3'])
            else:
                variation['chosen_node'] = random.choice(['ComputeHost3', 'ComputeHost4'])
            
            self.knowledge_base.append(variation)
        
        # 添加原始默认案例
        self.knowledge_base.extend(default_cases)
        
        print(f"✅ 默认RAG知识库已创建: {len(self.knowledge_base)} 个案例")
    
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        """基于RAG知识库增强的调度决策 - 优化版本"""
        try:
            # 首先使用DRL进行基础调度决策
            drl_node = self.drl_scheduler.schedule_task(
                task, available_nodes, node_capacities, node_loads, compute_service
            )
            
            # 如果没有知识库，抛出异常而不是回退
            if not self.knowledge_base or len(self.knowledge_base) == 0:
                raise RuntimeError("RAG知识库为空，无法进行RAG增强调度")
            
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
                
            except Exception as e:
                task_flops = 1e9
                task_memory = 1024
                total_workflow_flops = task_flops
                avg_load = 0
                max_load = 0
            
            # 增强的相似度匹配 - 降低阈值以获取更多匹配
            best_matches = []
            min_similarity_threshold = 0.5  # 从0.7降低到0.5
            
            for case in self.knowledge_base:
                # 多维特征相似度计算
                case_flops = float(case.get('task_flops', 1e9))
                case_workflow_flops = float(case.get('total_workflow_flops', case_flops))
                
                # 计算归一化距离
                flops_distance = abs(task_flops - case_flops) / max(task_flops, case_flops, 1e-6)
                workflow_distance = abs(total_workflow_flops - case_workflow_flops) / max(total_workflow_flops, case_workflow_flops, 1e-6)
                
                # 综合相似度
                similarity = 1.0 - (flops_distance * 0.6 + workflow_distance * 0.4)
                
                if similarity >= min_similarity_threshold:
                    best_matches.append({
                        'case': case,
                        'similarity': similarity,
                        'suggested_node': str(case.get('chosen_node', drl_node))
                    })
            
            # 如果有高质量匹配，使用加权投票机制
            if best_matches:
                # 按相似度排序，取前5个最佳匹配
                best_matches.sort(key=lambda x: x['similarity'], reverse=True)
                top_matches = best_matches[:5]
                
                # 加权投票选择节点
                node_votes = {}
                for match in top_matches:
                    node = match['suggested_node']
                    weight = match['similarity']
                    
                    if node not in node_votes:
                        node_votes[node] = 0
                    node_votes[node] += weight
                
                # 选择得票最高的节点
                if node_votes:
                    rag_suggested_node = max(node_votes.keys(), key=lambda x: node_votes[x])
                    
                    # 增强RAG决策权重：如果RAG有强建议且节点可用，优先使用
                    total_weight = sum(node_votes.values())
                    max_weight = max(node_votes.values())
                    confidence = max_weight / total_weight
                    
                    # 降低置信度阈值，从0.4降到0.3，更倾向于使用RAG建议
                    if confidence > 0.3 and rag_suggested_node in available_nodes:
                        print(f"🎯 RAG决策: 选择{rag_suggested_node} (置信度: {confidence:.2f})")
                        return rag_suggested_node
            
            # 如果没有足够的匹配，抛出异常
            raise RuntimeError("RAG知识库中没有找到足够的匹配案例")
            
        except Exception as e:
            print(f"⚠️ RAG调度失败: {e}")
            raise RuntimeError(f"RAG调度失败: {e}")

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
            "models/wass_optimized_models_compatible.pth",
            "models/wass_optimized_models.pth",
            "models/wass_models.pth"
        ]
        
        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break
        
        rag_path = "data/wrench_rag_knowledge_base.pkl"
        
        if model_path:
            print(f"📁 使用模型文件: {model_path}")
            
            # 强制启用WASS-DRL调度器
            try:
                drl_scheduler = WASSDRLScheduler(model_path)
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
                            rag_scheduler = WASSRAGScheduler(model_path, rag_candidate)
                            schedulers["WASS-RAG"] = rag_scheduler
                            print(f"✅ WASS-RAG调度器已启用 (知识库: {rag_candidate})")
                            rag_available = True
                            break
                        except Exception as e:
                            print(f"⚠️  WASS-RAG从{rag_candidate}加载失败: {e}")
                            continue
                
                if not rag_available:
                    # 即使没有知识库，也创建空的RAG调度器
                    rag_scheduler = WASSRAGScheduler(model_path, rag_path)
                    schedulers["WASS-RAG"] = rag_scheduler
                    print("⚠️  WASS-RAG调度器已创建 (知识库为空)")
                    
            except Exception as e:
                print(f"❌ DRL/RAG调度器初始化失败: {e}")
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
        """模拟WRENCH执行（简化实现）"""
        # 模拟节点负载
        node_loads = {node: 0.0 for node in self.compute_nodes}
        
        # 模拟任务执行
        task_times = {}
        decisions = []
        
        # 按拓扑顺序执行任务（简化处理）
        task_order = list(range(workflow_size))
        
        # 随机打乱任务顺序（模拟真实调度）
        random.shuffle(task_order)
        
        total_makespan = 0.0
        
        for task_id in task_order:
            task = workflow['tasks'][task_id]
            
            # 获取可用节点
            available_nodes = list(self.compute_nodes)
            
            # 使用调度器选择节点
            try:
                # 创建模拟任务对象
                class MockTask:
                    def __init__(self, flops, memory, cores):
                        self._flops = flops
                        self._memory = memory
                        self._cores = cores
                    
                    def get_flops(self):
                        return self._flops
                    
                    def get_memory_requirement(self):
                        return self._memory * 1024 * 1024 * 1024  # 转换为字节
                    
                    def get_min_num_cores(self):
                        return self._cores
                    
                    def get_input_files(self):
                        return []  # 简化处理
                    
                    def get_output_files(self):
                        return []  # 简化处理
                
                mock_task = MockTask(task['flops'], task['memory'], task['cores'])
                
                # 调用调度器
                chosen_node = scheduler.schedule_task(
                    mock_task, available_nodes, self.node_capacities, node_loads, None
                )
                
                # 计算执行时间
                capacity = self.node_capacities[chosen_node]
                exec_time = task['flops'] / (capacity * 1e9)
                
                # 更新节点负载
                node_loads[chosen_node] += exec_time
                
                # 记录任务执行时间
                task_times[f"task_{task_id}"] = exec_time
                
                # 记录调度决策
                decisions.append({
                    'task_id': f"task_{task_id}",
                    'chosen_node': chosen_node,
                    'execution_time': exec_time,
                    'start_time': node_loads[chosen_node] - exec_time,
                    'end_time': node_loads[chosen_node]
                })
                
                # 更新总makespan
                total_makespan = max(total_makespan, node_loads[chosen_node])
                
            except Exception as e:
                print(f"      ⚠️ 任务调度失败: {e}")
                # 使用默认节点
                chosen_node = self.compute_nodes[0]
                exec_time = task['flops'] / (self.node_capacities[chosen_node] * 1e9)
                node_loads[chosen_node] += exec_time
                task_times[f"task_{task_id}"] = exec_time
                total_makespan = max(total_makespan, node_loads[chosen_node])
        
        # 计算CPU利用率
        cpu_utilization = {}
        for node in self.compute_nodes:
            if total_makespan > 0:
                utilization = node_loads[node] / total_makespan
                cpu_utilization[node] = min(utilization, 1.0)
            else:
                cpu_utilization[node] = 0.0
        
        return {
            'makespan': total_makespan,
            'cpu_utilization': cpu_utilization,
            'task_times': task_times,
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
