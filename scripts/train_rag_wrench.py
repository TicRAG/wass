#!/usr/bin/env python3
"""
基于WRENCH的RAG知识库训练脚本
使用真实的WRENCH仿真数据构建RAG知识库
"""

import sys
import os
import json
import time
import random
import numpy as np
import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

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
class WRENCHKnowledgeCase:
    """基于WRENCH的知识案例"""
    # 工作流特征
    workflow_id: str
    task_count: int
    dependency_ratio: float
    critical_path_length: int
    workflow_embedding: np.ndarray
    
    # 任务特征
    task_id: str
    task_flops: float
    task_input_files: int
    task_output_files: int
    task_dependencies: int
    task_children: int
    task_features: np.ndarray
    
    # 节点特征
    available_nodes: List[str]
    node_capacities: Dict[str, float]
    node_loads: Dict[str, float]
    node_features: np.ndarray
    
    # 调度决策和结果
    scheduler_type: str
    chosen_node: str
    action_taken: int
    
    # 性能结果
    task_execution_time: float
    task_wait_time: float
    workflow_makespan: float
    node_utilization: Dict[str, float]
    
    # 元数据
    simulation_time: float
    platform_config: str
    metadata: Dict[str, Any]

class WRENCHWorkflowEmbedder:
    """基于WRENCH的工作流嵌入器"""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
    
    def encode_workflow(self, workflow, tasks: List) -> np.ndarray:
        """将WRENCH工作流编码为向量"""
        if not tasks:
            return np.zeros(self.embedding_dim)
        
        # 工作流级别特征
        num_tasks = len(tasks)
        
        # 计算依赖关系
        total_dependencies = 0
        total_children = 0
        task_flops = []
        
        for task in tasks:
            deps = len(task.get_input_files())
            children = task.get_number_of_children()
            flops = task.get_flops()
            
            total_dependencies += deps
            total_children += children
            task_flops.append(flops)
        
        # 基础统计特征
        features = [
            num_tasks,
            total_dependencies / max(num_tasks, 1),  # 平均依赖数
            total_children / max(num_tasks, 1),      # 平均子任务数
            np.mean(task_flops) if task_flops else 0,
            np.std(task_flops) if len(task_flops) > 1 else 0,
            max(task_flops) if task_flops else 0,
            min(task_flops) if task_flops else 0
        ]
        
        # 填充到指定维度
        while len(features) < self.embedding_dim:
            features.append(0.0)
        
        return np.array(features[:self.embedding_dim], dtype=np.float32)
    
    def encode_task(self, task) -> np.ndarray:
        """编码单个任务特征"""
        features = [
            task.get_flops() / 1e9,  # 标准化到GFlops
            len(task.get_input_files()),
            len(task.get_output_files()),
            task.get_number_of_children(),
            1.0  # 任务活跃标记
        ]
        return np.array(features, dtype=np.float32)
    
    def encode_nodes(self, node_capacities: Dict[str, float], 
                    node_loads: Dict[str, float]) -> np.ndarray:
        """编码节点特征"""
        if not node_capacities:
            return np.zeros(12)  # 4节点 * 3特征
        
        features = []
        for node in sorted(node_capacities.keys()):
            capacity = node_capacities.get(node, 0.0)
            load = node_loads.get(node, 0.0)
            availability = max(0.0, capacity - load)
            
            features.extend([
                capacity / 4.0,      # 标准化容量
                load / 4.0,          # 标准化负载
                availability / 4.0   # 标准化可用性
            ])
        
        # 确保固定长度
        while len(features) < 12:
            features.append(0.0)
        
        return np.array(features[:12], dtype=np.float32)

class WRENCHRAGKnowledgeBase:
    """基于WRENCH的RAG知识库"""
    
    def __init__(self, embedding_dim: int = 64):
        self.cases: List[WRENCHKnowledgeCase] = []
        self.embedder = WRENCHWorkflowEmbedder(embedding_dim)
        self.case_index = {}
    
    def add_case(self, case: WRENCHKnowledgeCase):
        """添加知识案例"""
        self.cases.append(case)
    
    def build_index(self):
        """构建检索索引"""
        if not self.cases:
            return
        
        print(f"构建检索索引，共 {len(self.cases)} 个案例...")
        
        # 提取所有工作流嵌入
        embeddings = np.array([case.workflow_embedding for case in self.cases])
        
        # 使用简单的k-means聚类
        n_clusters = min(20, len(self.cases))
        cluster_centers = self._simple_kmeans(embeddings, n_clusters)
        
        # 为每个案例分配到最近的聚类
        self.case_index = {i: [] for i in range(n_clusters)}
        
        for i, case in enumerate(self.cases):
            distances = [np.linalg.norm(case.workflow_embedding - center) 
                        for center in cluster_centers]
            cluster_id = np.argmin(distances)
            self.case_index[cluster_id].append(i)
        
        print(f"索引构建完成：{n_clusters} 个聚类")
        for i, cases in self.case_index.items():
            print(f"  聚类 {i}: {len(cases)} 个案例")
    
    def _simple_kmeans(self, data: np.ndarray, k: int, max_iters: int = 50) -> np.ndarray:
        """简单的K-means实现"""
        n, d = data.shape
        if n < k:
            return data
        
        centroids = data[np.random.choice(n, k, replace=False)]
        
        for _ in range(max_iters):
            # 分配点到最近的聚类中心
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            assignments = np.argmin(distances, axis=0)
            
            # 更新聚类中心
            new_centroids = np.array([data[assignments == i].mean(axis=0) 
                                    if np.any(assignments == i) else centroids[i]
                                    for i in range(k)])
            
            # 检查收敛
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return centroids
    
    def retrieve_similar_cases(self, query_embedding: np.ndarray, 
                             query_task_features: np.ndarray,
                             k: int = 5) -> List[Tuple[WRENCHKnowledgeCase, float]]:
        """检索相似案例"""
        if not self.cases:
            return []
        
        similarities = []
        for case in self.cases:
            # 工作流相似度
            workflow_sim = self._cosine_similarity(query_embedding, case.workflow_embedding)
            
            # 任务相似度
            task_sim = self._cosine_similarity(query_task_features, case.task_features)
            
            # 综合相似度
            total_sim = 0.7 * workflow_sim + 0.3 * task_sim
            similarities.append((case, total_sim))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def save(self, path: str):
        """保存知识库"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'cases': self.cases,
                'case_index': self.case_index,
                'embedding_dim': self.embedder.embedding_dim
            }, f)
        
        print(f"知识库已保存到 {path}")
    
    @classmethod
    def load(cls, path: str) -> 'WRENCHRAGKnowledgeBase':
        """加载知识库"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        kb = cls(data['embedding_dim'])
        kb.cases = data['cases']
        kb.case_index = data['case_index']
        
        return kb

class WRENCHRAGTrainer:
    """基于WRENCH的RAG训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.knowledge_base = WRENCHRAGKnowledgeBase()
        
        # WRENCH平台配置
        self.platform_file = config['platform']['platform_file']
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
        self.schedulers = ["HEFT", "WASS-Heuristic"]
        
        print(f"WRENCH RAG训练器初始化完成")
    
    def generate_wrench_cases(self, num_cases: int = 1000):
        """使用WRENCH生成知识案例"""
        print(f"🚀 开始生成 {num_cases} 个WRENCH知识案例...")
        
        with open(self.platform_file, 'r', encoding='utf-8') as f:
            platform_xml = f.read()
        
        cases_generated = 0
        
        for scheduler_type in self.schedulers:
            cases_per_scheduler = num_cases // len(self.schedulers)
            
            for case_idx in range(cases_per_scheduler):
                try:
                    case = self._generate_single_case(platform_xml, scheduler_type, case_idx)
                    if case:
                        self.knowledge_base.add_case(case)
                        cases_generated += 1
                        
                        if cases_generated % 50 == 0:
                            print(f"  已生成 {cases_generated}/{num_cases} 个案例...")
                
                except Exception as e:
                    print(f"生成案例失败 (调度器: {scheduler_type}, 索引: {case_idx}): {e}")
                    continue
        
        print(f"✅ 总共生成了 {cases_generated} 个WRENCH知识案例")
    
    def _generate_single_case(self, platform_xml: str, scheduler_type: str, case_idx: int) -> WRENCHKnowledgeCase:
        """生成单个知识案例"""
        # 创建仿真
        sim = wrench.Simulation()
        sim.start(platform_xml, self.controller_host)
        
        try:
            # 创建服务
            storage_service = sim.create_simple_storage_service("StorageHost", ["/storage"])
            
            compute_resources = {}
            for node in self.compute_nodes:
                compute_resources[node] = (4, 8_589_934_592)
            
            compute_service = sim.create_bare_metal_compute_service(
                "ComputeHost1", compute_resources, "/scratch", {}, {}
            )
            
            # 创建随机工作流
            workflow = sim.create_workflow()
            num_tasks = random.randint(5, 20)
            tasks = []
            files = []
            
            # 创建任务
            for i in range(num_tasks):
                flops = random.uniform(1e9, 10e9)
                task = workflow.add_task(f"task_{case_idx}_{i}", flops, 1, 1, 0)
                tasks.append(task)
                
                # 创建输出文件
                if i < num_tasks - 1:
                    output_file = sim.add_file(f"output_{case_idx}_{i}", random.randint(1024, 10240))
                    task.add_output_file(output_file)
                    files.append(output_file)
            
            # 创建依赖关系
            for i in range(1, min(num_tasks, len(files) + 1)):
                if i > 1 and random.random() < 0.3:
                    dep_idx = random.randint(0, i-2)
                    if dep_idx < len(files):
                        tasks[i].add_input_file(files[dep_idx])
            
            # 为文件创建副本
            for file in files:
                storage_service.create_file_copy(file)
            
            # 工作流嵌入
            workflow_embedding = self.knowledge_base.embedder.encode_workflow(workflow, tasks)
            
            # 模拟调度过程
            node_loads = {node: 0.0 for node in self.compute_nodes}
            task_results = []
            
            ready_tasks = workflow.get_ready_tasks()
            while ready_tasks:
                current_task = ready_tasks[0]
                
                # 任务特征
                task_features = self.knowledge_base.embedder.encode_task(current_task)
                
                # 节点特征
                node_features = self.knowledge_base.embedder.encode_nodes(
                    self.node_capacities, node_loads)
                
                # 根据调度器选择节点
                if scheduler_type == "HEFT":
                    # 最快处理器
                    chosen_node = max(self.compute_nodes, key=lambda x: self.node_capacities[x])
                elif scheduler_type == "FIFO":
                    # 最少负载
                    chosen_node = min(self.compute_nodes, key=lambda x: node_loads[x])
                else:  # Random
                    chosen_node = random.choice(self.compute_nodes)
                
                action_taken = self.compute_nodes.index(chosen_node)
                
                # 提交作业
                file_locations = {}
                for f in current_task.get_input_files():
                    file_locations[f] = storage_service
                for f in current_task.get_output_files():
                    file_locations[f] = storage_service
                
                job = sim.create_standard_job([current_task], file_locations)
                compute_service.submit_standard_job(job)
                
                # 等待完成
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
                
                # 更新节点负载
                node_loads[chosen_node] += execution_time
                
                # 创建知识案例
                case = WRENCHKnowledgeCase(
                    workflow_id=f"workflow_{case_idx}",
                    task_count=num_tasks,
                    dependency_ratio=sum(len(t.get_input_files()) for t in tasks) / num_tasks,
                    critical_path_length=num_tasks,  # 简化
                    workflow_embedding=workflow_embedding,
                    
                    task_id=current_task.get_name(),
                    task_flops=current_task.get_flops(),
                    task_input_files=len(current_task.get_input_files()),
                    task_output_files=len(current_task.get_output_files()),
                    task_dependencies=len(current_task.get_input_files()),
                    task_children=current_task.get_number_of_children(),
                    task_features=task_features,
                    
                    available_nodes=self.compute_nodes.copy(),
                    node_capacities=self.node_capacities.copy(),
                    node_loads=node_loads.copy(),
                    node_features=node_features,
                    
                    scheduler_type=scheduler_type,
                    chosen_node=chosen_node,
                    action_taken=action_taken,
                    
                    task_execution_time=execution_time,
                    task_wait_time=0.0,  # 简化
                    workflow_makespan=end_time,
                    node_utilization=node_loads.copy(),
                    
                    simulation_time=end_time,
                    platform_config=self.platform_file,
                    metadata={
                        "scheduler": scheduler_type,
                        "case_index": case_idx,
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                
                task_results.append(case)
                ready_tasks = workflow.get_ready_tasks()
                
                # 只记录第一个任务的案例（简化）
                break
        
        finally:
            sim.terminate()
        
        return task_results[0] if task_results else None
    
    def train_retriever(self):
        """训练RAG检索器"""
        print("🔧 训练RAG检索器...")
        
        # 构建索引
        self.knowledge_base.build_index()
        
        # 评估检索质量
        self._evaluate_retrieval()
        
        print("✅ RAG检索器训练完成")
    
    def _evaluate_retrieval(self):
        """评估检索质量"""
        if len(self.knowledge_base.cases) < 10:
            print("案例数量不足，跳过评估")
            return
        
        # 随机选择测试案例
        test_cases = np.random.choice(len(self.knowledge_base.cases), 
                                    min(20, len(self.knowledge_base.cases)), 
                                    replace=False)
        
        retrieval_scores = []
        
        for case_idx in test_cases:
            test_case = self.knowledge_base.cases[case_idx]
            
            # 检索相似案例
            retrieved = self.knowledge_base.retrieve_similar_cases(
                test_case.workflow_embedding, 
                test_case.task_features, 
                k=5
            )
            
            # 计算调度器一致性
            retrieved_schedulers = [case.scheduler_type for case, _ in retrieved]
            consistency = retrieved_schedulers.count(test_case.scheduler_type) / len(retrieved_schedulers)
            retrieval_scores.append(consistency)
        
        avg_consistency = np.mean(retrieval_scores) if retrieval_scores else 0.0
        print(f"📊 检索质量评估 - 调度器一致性: {avg_consistency:.3f}")
    
    def save_knowledge_base(self, path: str = "data/wrench_rag_knowledge_base.pkl"):
        """保存知识库"""
        self.knowledge_base.save(path)
        
        # 也保存为JSON格式便于查看
        json_path = path.replace('.pkl', '.json')
        json_data = {
            'metadata': {
                'total_cases': len(self.knowledge_base.cases),
                'schedulers': list(set(case.scheduler_type for case in self.knowledge_base.cases)),
                'generated_at': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'sample_cases': []
        }
        
        # 添加一些样本案例
        for i, case in enumerate(self.knowledge_base.cases[:10]):
            json_data['sample_cases'].append({
                'workflow_id': case.workflow_id,
                'task_id': case.task_id,
                'scheduler_type': case.scheduler_type,
                'chosen_node': case.chosen_node,
                'task_execution_time': case.task_execution_time,
                'workflow_makespan': case.workflow_makespan
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 知识库概要已保存到 {json_path}")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Usage: python scripts/train_rag_wrench.py <config.yaml>")
        sys.exit(1)
    
    config = load_config(sys.argv[1])
    trainer = WRENCHRAGTrainer(config)
    
    # 生成WRENCH案例
    num_cases = config.get('rag', {}).get('num_cases', 500)
    trainer.generate_wrench_cases(num_cases)
    
    # 训练检索器
    trainer.train_retriever()
    
    # 保存知识库
    trainer.save_knowledge_base()
    
    print(f"\n🎉 基于WRENCH的RAG训练完成!")
    print(f"📈 知识库包含 {len(trainer.knowledge_base.cases)} 个真实仿真案例")

if __name__ == "__main__":
    main()
