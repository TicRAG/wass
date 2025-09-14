from __future__ import annotations
"""Full legacy WRENCH RAG KB structures migrated from training script.
Minimized to data representation + basic retrieval for modular reuse.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

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

__all__ = ["WRENCHKnowledgeCase", "WRENCHRAGKnowledgeBase"]
