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
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import yaml

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

class WASSDRLScheduler(WRENCHScheduler):
    """基于训练好的DRL模型的调度器"""
    
    def __init__(self, model_path: str):
        super().__init__("WASS-DRL")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_path)
        
        # 节点映射
        self.compute_nodes = ["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"]
        self.node_capacities = {
            "ComputeHost1": 2.0,
            "ComputeHost2": 3.0, 
            "ComputeHost3": 2.5,
            "ComputeHost4": 4.0
        }
    
    def _load_model(self, model_path: str):
        """加载训练好的DRL模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if "drl_agent" in checkpoint:
                # 直接定义网络类，避免导入问题
                import torch.nn as nn
                
                class SimpleDQN(nn.Module):
                    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
                        super().__init__()
                        self.network = nn.Sequential(
                            nn.Linear(state_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, action_dim)
                        )
                    
                    def forward(self, x):
                        return self.network(x)
                
                state_dim = 17  # 与训练时一致
                action_dim = 4   # 4个节点
                self.model = SimpleDQN(state_dim, action_dim).to(self.device)
                self.model.load_state_dict(checkpoint["drl_agent"])
                self.model.eval()
                print(f"✅ DRL模型已加载: {model_path}")
            else:
                print(f"❌ 未找到DRL模型参数")
                self.model = None
        except Exception as e:
            print(f"❌ DRL模型加载失败: {e}")
            self.model = None
    
    def _get_state(self, task, available_nodes, node_capacities, node_loads):
        """构造状态向量"""
        # 任务特征
        task_features = [
            task.get_flops() / 1e9,  # 标准化到GFlops
            len(task.get_input_files()),
            task.get_number_of_children(),
            4,  # 总节点数
            0.5  # 假设完成进度
        ]
        
        # 节点特征
        node_features = []
        for node in self.compute_nodes:
            if node in available_nodes:
                capacity = node_capacities.get(node, 0.0)
                load = node_loads.get(node, 0.0)
                availability = max(0.0, capacity - load)
                
                node_features.extend([
                    capacity / 4.0,      # 标准化容量
                    load / 4.0,          # 标准化负载
                    availability / 4.0   # 标准化可用性
                ])
            else:
                node_features.extend([0.0, 0.0, 0.0])
        
        state = np.array(task_features + node_features, dtype=np.float32)
        return state
    
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        if self.model is None:
            # 模型未加载，抛出异常
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
            
            # 如果选择的节点不可用，选择第一个可用节点
            return available_nodes[0]
            
        except Exception as e:
            raise RuntimeError(f"DRL调度失败: {e}")

class WASSRAGScheduler(WRENCHScheduler):
    """基于RAG知识库增强的调度器"""
    
    def __init__(self, model_path: str, rag_path: str):
        super().__init__("WASS-RAG")
        self.drl_scheduler = WASSDRLScheduler(model_path)
        self.knowledge_base = None
        self._load_rag_knowledge(rag_path)
    
    def _load_rag_knowledge(self, rag_path: str):
        """加载RAG知识库"""
        # 优先使用扩展的JSON知识库
        extended_json_path = "data/extended_rag_knowledge.json"
        if os.path.exists(extended_json_path):
            try:
                with open(extended_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 从JSON数据构建知识库
                self.knowledge_base = []
                if 'cases' in data:
                    for case in data['cases']:
                        simple_case = {
                            'task_flops': case.get('task_flops', case.get('task_execution_time', 1.0) * 2e9),
                            'chosen_node': case.get('chosen_node', 'ComputeHost1'),
                            'scheduler_type': case.get('scheduler_type', 'unknown'),
                            'task_execution_time': case.get('task_execution_time', 0.0),
                            'workflow_makespan': case.get('workflow_makespan', 0.0),
                            'node_capacity': case.get('node_capacity', 2.0),
                            'performance_ratio': case.get('performance_ratio', 1.0)
                        }
                        self.knowledge_base.append(simple_case)
                
                print(f"✅ RAG知识库已从扩展JSON加载: {len(self.knowledge_base)} 个案例")
                return
            except Exception as e:
                print(f"扩展JSON加载失败: {e}")
        
        # 回退到原始JSON格式
        json_path = rag_path.replace('.pkl', '.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 从JSON数据构建知识库
                self.knowledge_base = []
                if 'sample_cases' in data:
                    for case in data['sample_cases']:
                        # 估算task_flops（基于执行时间和节点性能）
                        exec_time = case.get('task_execution_time', 1.0)
                        node = case.get('chosen_node', 'ComputeHost4')
                        node_capacity = self.get_node_capacity(node)
                        estimated_flops = exec_time * node_capacity * 1e9
                        
                        simple_case = {
                            'task_flops': estimated_flops,
                            'chosen_node': case.get('chosen_node', 'ComputeHost1'),
                            'scheduler_type': case.get('scheduler_type', 'unknown'),
                            'task_execution_time': case.get('task_execution_time', 0.0),
                            'workflow_makespan': case.get('workflow_makespan', 0.0)
                        }
                        self.knowledge_base.append(simple_case)
                
                print(f"✅ RAG知识库已从原始JSON加载: {len(self.knowledge_base)} 个案例")
                return
            except Exception as e:
                print(f"原始JSON加载失败: {e}")
        
        # 最后回退到pickle格式
        try:
            with open(rag_path, 'rb') as f:
                data = pickle.load(f)
                # 提取案例数据，忽略类型问题
                if 'cases' in data and data['cases']:
                    # 转换为简化格式
                    self.knowledge_base = []
                    for case in data['cases']:
                        # 假设case有这些属性，用字典格式存储
                        if hasattr(case, 'task_flops') and hasattr(case, 'chosen_node'):
                            simple_case = {
                                'task_flops': case.task_flops,
                                'chosen_node': case.chosen_node,
                                'scheduler_type': getattr(case, 'scheduler_type', 'unknown'),
                                'task_execution_time': getattr(case, 'task_execution_time', 0.0)
                            }
                            self.knowledge_base.append(simple_case)
                    print(f"✅ RAG知识库已从PKL加载: {len(self.knowledge_base)} 个案例")
                else:
                    print(f"❌ RAG知识库格式错误")
                    self.knowledge_base = []
        except Exception as e:
            print(f"❌ RAG知识库加载失败: {e}")
            self.knowledge_base = []
    
    def _retrieve_similar_cases(self, task, k=3):
        """检索相似案例"""
        if not self.knowledge_base:
            return []
        
        # 简化的相似度计算
        task_flops = task.get_flops()
        similar_cases = []
        
        for case in self.knowledge_base:
            # 基于任务计算量的相似度
            case_flops = case.get('task_flops', 0)
            if case_flops > 0:
                flops_diff = abs(case_flops - task_flops) / max(case_flops, task_flops)
                similarity = 1.0 - flops_diff
                similar_cases.append((case, similarity))
        
        # 返回最相似的k个案例
        similar_cases.sort(key=lambda x: x[1], reverse=True)
        return similar_cases[:k]
    
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        # 检查DRL调度器是否可用
        if self.drl_scheduler.model is None:
            raise RuntimeError("DRL模型未加载，WASS-RAG无法运行")
        
        # 检查RAG知识库是否可用
        if not self.knowledge_base:
            raise RuntimeError("RAG知识库未加载，WASS-RAG无法运行")
        
        # 首先使用DRL获得基础决策
        drl_choice = self.drl_scheduler.schedule_task(task, available_nodes, node_capacities, node_loads, compute_service)
        
        # 使用RAG知识增强决策
        similar_cases = self._retrieve_similar_cases(task)
        
        if similar_cases:
            # 分析相似案例的调度决策
            node_votes = {}
            for case, similarity in similar_cases:
                node = case.get('chosen_node', '')
                if node in available_nodes:
                    node_votes[node] = node_votes.get(node, 0) + similarity
            
            if node_votes:
                # 选择投票最高的节点
                rag_choice = max(node_votes.keys(), key=lambda x: node_votes[x])
                
                # 结合DRL和RAG的决策（偏向RAG推荐）
                if rag_choice in available_nodes and random.random() < 0.7:
                    return rag_choice
        
        return drl_choice

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
        
        # 检查训练好的模型
        model_path = "models/wass_models.pth"
        rag_path = "data/wrench_rag_knowledge_base.pkl"
        
        if os.path.exists(model_path):
            # 尝试加载DRL调度器
            try:
                drl_scheduler = WASSDRLScheduler(model_path)
                if drl_scheduler.model is not None:
                    schedulers["WASS-DRL"] = drl_scheduler
                    print("✅ WASS-DRL调度器已启用")
                    
                    # 只有在DRL成功加载后才尝试RAG
                    if os.path.exists(rag_path):
                        try:
                            rag_scheduler = WASSRAGScheduler(model_path, rag_path)
                            if rag_scheduler.knowledge_base:
                                schedulers["WASS-RAG"] = rag_scheduler
                                print("✅ WASS-RAG调度器已启用")
                            else:
                                print("⚠️  RAG知识库为空，跳过WASS-RAG")
                        except Exception as e:
                            print(f"⚠️  WASS-RAG初始化失败: {e}")
                    else:
                        print(f"⚠️  RAG知识库文件未找到: {rag_path}")
                else:
                    print("⚠️  DRL模型加载失败，跳过WASS-DRL和WASS-RAG")
            except Exception as e:
                print(f"⚠️  DRL调度器初始化失败: {e}")
        else:
            print(f"⚠️  训练模型文件未找到: {model_path}")
        
        print(f"🔧 已启用调度器: {list(schedulers.keys())}")
        return schedulers
    
    def run_single_experiment(self, scheduler_name: str, workflow_size: int, experiment_id: int) -> WRENCHExperimentResult:
        """运行单次WRENCH实验"""
        print(f"  运行实验: {scheduler_name}, {workflow_size}任务, 实验#{experiment_id}")
        
        with open(self.platform_file, 'r', encoding='utf-8') as f:
            platform_xml = f.read()
        
        # 创建仿真
        sim = wrench.Simulation()
        sim.start(platform_xml, self.controller_host)
        
        try:
            # 创建服务
            storage_service = sim.create_simple_storage_service("StorageHost", ["/storage"])
            
            compute_resources = {}
            for node in self.compute_nodes:
                compute_resources[node] = (4, 8_589_934_592)  # 4核, 8GB内存
            
            compute_service = sim.create_bare_metal_compute_service(
                "ComputeHost1", compute_resources, "/scratch", {}, {}
            )
            
            # 创建工作流
            workflow = sim.create_workflow()
            tasks = []
            files = []
            
            # 创建任务
            for i in range(workflow_size):
                flops = random.uniform(2e9, 10e9)
                task = workflow.add_task(f"task_{experiment_id}_{i}", flops, 1, 1, 0)
                tasks.append(task)
                
                # 创建输出文件
                if i < workflow_size - 1:
                    output_file = sim.add_file(f"output_{experiment_id}_{i}", random.randint(1024, 10240))
                    task.add_output_file(output_file)
                    files.append(output_file)
            
            # 创建依赖关系
            dependency_count = 0
            for i in range(1, min(workflow_size, len(files) + 1)):
                if i > 1 and random.random() < 0.3:  # 30%概率有依赖
                    dep_idx = random.randint(0, i-2)
                    if dep_idx < len(files):
                        tasks[i].add_input_file(files[dep_idx])
                        dependency_count += 1
            
            # 为文件创建副本
            for file in files:
                storage_service.create_file_copy(file)
            
            # 获取调度器
            scheduler = self.schedulers[scheduler_name]
            
            # 执行调度
            node_loads = {node: 0.0 for node in self.compute_nodes}
            task_execution_times = {}
            scheduling_decisions = []
            
            # 模拟调度过程
            ready_tasks = workflow.get_ready_tasks()
            while ready_tasks:
                current_task = ready_tasks[0]
                
                # 调度决策
                chosen_node = scheduler.schedule_task(
                    current_task, self.compute_nodes, self.node_capacities, node_loads, compute_service
                )
                
                # 记录调度决策
                scheduling_decisions.append({
                    "task": current_task.get_name(),
                    "node": chosen_node,
                    "scheduler": scheduler_name,
                    "task_flops": current_task.get_flops()
                })
                
                # 提交作业
                file_locations = {}
                for f in current_task.get_input_files():
                    file_locations[f] = storage_service
                for f in current_task.get_output_files():
                    file_locations[f] = storage_service
                
                job = sim.create_standard_job([current_task], file_locations)
                compute_service.submit_standard_job(job)
                
                # 等待作业完成
                start_time = sim.get_simulated_time()
                while True:
                    event = sim.wait_for_next_event()
                    if event["event_type"] == "standard_job_completion":
                        completed_job = event["standard_job"]
                        if completed_job == job:
                            break
                    elif event["event_type"] == "simulation_termination":
                        break
                
                end_time = sim.get_simulated_time()
                execution_time = end_time - start_time
                
                # 记录执行时间
                task_execution_times[current_task.get_name()] = execution_time
                node_loads[chosen_node] += execution_time
                
                # 获取下一批就绪任务
                ready_tasks = workflow.get_ready_tasks()
            
            # 计算最终性能指标
            makespan = sim.get_simulated_time()
            
            # 计算CPU利用率
            total_work = sum(task_execution_times.values())
            cpu_utilization = {}
            for node in self.compute_nodes:
                node_work = sum(execution_time for task_name, execution_time in task_execution_times.items() 
                               if any(d["task"] == task_name and d["node"] == node for d in scheduling_decisions))
                cpu_utilization[node] = node_work / makespan if makespan > 0 else 0.0
            
            return WRENCHExperimentResult(
                scheduler_name=scheduler_name,
                workflow_id=f"workflow_{experiment_id}",
                task_count=workflow_size,
                dependency_count=dependency_count,
                makespan=makespan,
                cpu_utilization=cpu_utilization,
                task_execution_times=task_execution_times,
                scheduling_decisions=scheduling_decisions,
                experiment_metadata={
                    "experiment_id": experiment_id,
                    "platform": self.platform_file,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            )
        
        finally:
            sim.terminate()
    
    def run_all_experiments(self):
        """运行所有实验配置"""
        print(f"🔬 开始完整WRENCH实验...")
        print(f"调度器: {list(self.schedulers.keys())}")
        print(f"工作流规模: {self.workflow_sizes}")
        print(f"重复次数: {self.repetitions}")
        
        total_experiments = len(self.schedulers) * len(self.workflow_sizes) * self.repetitions
        current_exp = 0
        
        for scheduler_name in self.schedulers.keys():
            for workflow_size in self.workflow_sizes:
                for rep in range(self.repetitions):
                    current_exp += 1
                    print(f"\n进度: {current_exp}/{total_experiments}")
                    
                    try:
                        result = self.run_single_experiment(scheduler_name, workflow_size, current_exp)
                        self.results.append(result)
                        print(f"  ✅ 完成: {result.makespan:.2f}s")
                    except Exception as e:
                        print(f"  ❌ 实验失败: {e}")
        
        # 保存结果
        self._save_results()
        self._analyze_results()
    
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
