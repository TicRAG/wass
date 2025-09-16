from __future__ import annotations
"""Full legacy WRENCH RAG KB structures migrated from training script.
Minimized to data representation + basic retrieval for modular reuse.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class WRENCHKnowledgeCase:
    workflow_id: str
    task_count: int
    dependency_ratio: float
    critical_path_length: int
    workflow_embedding: np.ndarray
    task_id: str
    task_flops: float
    task_input_files: int
    task_output_files: int
    task_dependencies: int
    task_children: int
    task_features: np.ndarray
    available_nodes: List[str]
    node_capacities: Dict[str, float]
    node_loads: Dict[str, float]
    node_features: np.ndarray
    scheduler_type: str
    chosen_node: str
    action_taken: int
    task_execution_time: float
    task_wait_time: float
    workflow_makespan: float
    node_utilization: Dict[str, float]
    simulation_time: float
    platform_config: str
    metadata: Dict[str, Any]

class WRENCHRAGKnowledgeBase:
    def __init__(self, embedding_dim: int = 64):
        self.embedder_dim = embedding_dim
        self.cases: List[WRENCHKnowledgeCase] = []
        self.case_index: Dict[str, List[int]] = {}

    def add_case(self, case: WRENCHKnowledgeCase):
        idx = len(self.cases)
        self.cases.append(case)
        key = f"{case.workflow_id}:{case.task_id}"
        self.case_index.setdefault(key, []).append(idx)

    def retrieve_similar_cases(self, workflow_embedding: np.ndarray, task_features: np.ndarray, k: int = 5, sort_by_makespan: bool = True) -> List[Tuple[WRENCHKnowledgeCase, float]]:
        """
        检索相似案例，可选择按makespan排序
        
        Args:
            workflow_embedding: 工作流嵌入向量
            task_features: 任务特征向量
            k: 返回的案例数量
            sort_by_makespan: 是否按makespan排序（从低到高）
            
        Returns:
            相似案例列表，按相似度或makespan排序
        """
        if not self.cases:
            return []
        
        # 计算相似度
        sims = []
        for c in self.cases:
            wf_sim = self._cosine(workflow_embedding, c.workflow_embedding)
            task_sim = self._cosine(task_features, c.task_features)
            score = 0.7 * wf_sim + 0.3 * task_sim
            sims.append((c, score))
        
        # 按相似度排序
        sims.sort(key=lambda x: x[1], reverse=True)
        
        # 获取top_k案例
        top_cases = sims[:k]
        
        # 如果需要按makespan排序
        if sort_by_makespan:
            top_cases.sort(key=lambda x: x[0].workflow_makespan)
        
        return top_cases

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def save_knowledge_base(self, filename: str):
        """保存知识库到文件"""
        output_path = Path(filename)
        
        # 转换为可序列化的格式
        serializable_cases = []
        for case in self.cases:
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
                'case_index': self.case_index
            }, f, indent=2)
        
        logger.info(f"Knowledge base saved to {output_path}")

    def load_knowledge_base(self, filename: str):
        """从文件加载知识库，兼容JSONKnowledgeBase格式"""
        input_path = Path(filename)
        
        if not input_path.exists():
            logger.warning(f"Knowledge base file {input_path} not found")
            return
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # 清空当前知识库
        self.cases = []
        self.case_index = {}
        
        # 检查数据格式并适配
        if 'metadata' in data and 'cases' in data:
            # JSONKnowledgeBase格式
            logger.info("检测到JSONKnowledgeBase格式，进行格式转换...")
            self._load_json_knowledge_base_format(data)
        else:
            # 原始WRENCHRAGKnowledgeBase格式
            self._load_wrench_knowledge_base_format(data)
        
        logger.info(f"Knowledge base loaded from {input_path} with {len(self.cases)} cases")
    
    def _load_json_knowledge_base_format(self, data: Dict[str, Any]):
        """加载JSONKnowledgeBase格式的数据"""
        for case_dict in data['cases']:
            try:
                # 转换任务特征从字典到数组
                task_features_dict = case_dict.get('task_features', {})
                if isinstance(task_features_dict, dict):
                    task_features = np.array([
                        task_features_dict.get('task_flops', 0.0),
                        task_features_dict.get('task_memory', 0.0),
                        task_features_dict.get('task_inputs', 0.0),
                        task_features_dict.get('task_outputs', 0.0),
                        task_features_dict.get('task_dependencies', 0.0)
                    ])
                else:
                    task_features = np.array(task_features_dict) if task_features_dict else np.zeros(5)
                
                # 转换工作流嵌入
                workflow_embedding = np.array(case_dict.get('workflow_embedding', []))
                if len(workflow_embedding) == 0:
                    workflow_embedding = np.zeros(self.embedder_dim)
                
                # 创建WRENCH知识案例（使用默认值填充缺失字段）
                case = WRENCHKnowledgeCase(
                    workflow_id=case_dict.get('workflow_id', 'unknown'),
                    task_count=case_dict.get('platform_features', {}).get('num_nodes', 5),
                    dependency_ratio=0.3,  # 默认值
                    critical_path_length=3,  # 默认值
                    workflow_embedding=workflow_embedding,
                    task_id=case_dict.get('task_id', 'unknown'),
                    task_flops=task_features_dict.get('task_flops', 0.0) if isinstance(task_features_dict, dict) else 0.0,
                    task_input_files=int(task_features_dict.get('task_inputs', 0)) if isinstance(task_features_dict, dict) else 0,
                    task_output_files=int(task_features_dict.get('task_outputs', 0)) if isinstance(task_features_dict, dict) else 0,
                    task_dependencies=int(task_features_dict.get('task_dependencies', 0)) if isinstance(task_features_dict, dict) else 0,
                    task_children=0,  # 默认值
                    task_features=task_features,
                    available_nodes=['node_0', 'node_1', 'node_2', 'node_3', 'node_4'],  # 默认值
                    node_capacities={'node_0': 1.0, 'node_1': 1.0, 'node_2': 1.0, 'node_3': 1.0, 'node_4': 1.0},  # 默认值
                    node_loads={'node_0': 0.0, 'node_1': 0.0, 'node_2': 0.0, 'node_3': 0.0, 'node_4': 0.0},  # 默认值
                    node_features=np.zeros(10),  # 默认值
                    scheduler_type=case_dict.get('scheduler_type', 'HEFT'),
                    chosen_node=case_dict.get('chosen_node', 'node_0'),
                    action_taken=0,  # 默认值
                    task_execution_time=case_dict.get('task_execution_time', 0.0),
                    task_wait_time=0.0,  # 默认值
                    workflow_makespan=case_dict.get('makespan', 0.0),
                    node_utilization={'node_0': 0.0, 'node_1': 0.0, 'node_2': 0.0, 'node_3': 0.0, 'node_4': 0.0},  # 默认值
                    simulation_time=0.0,  # 默认值
                    platform_config='default',  # 默认值
                    metadata={'quality_score': case_dict.get('case_quality_score', 1.0), 'is_real_case': case_dict.get('is_real_case', True)}
                )
                
                self.add_case(case)
                
            except Exception as e:
                logger.warning(f"跳过案例加载失败: {e}")
                continue
    
    def _load_wrench_knowledge_base_format(self, data: Dict[str, Any]):
        """加载原始WRENCHRAGKnowledgeBase格式的数据"""
        for case_dict in data['cases']:
            # 转换列表为numpy数组
            case_dict['workflow_embedding'] = np.array(case_dict['workflow_embedding'])
            case_dict['task_features'] = np.array(case_dict['task_features'])
            case_dict['node_features'] = np.array(case_dict['node_features'])
            
            # 创建案例对象
            case = WRENCHKnowledgeCase(**case_dict)
            self.add_case(case)

__all__ = ["WRENCHKnowledgeCase", "WRENCHRAGKnowledgeBase"]
