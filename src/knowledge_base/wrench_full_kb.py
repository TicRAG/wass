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

    def retrieve_similar_cases(self, workflow_embedding: np.ndarray, task_features: np.ndarray, k: int = 5) -> List[Tuple[WRENCHKnowledgeCase, float]]:
        if not self.cases:
            return []
        sims = []
        for c in self.cases:
            wf_sim = self._cosine(workflow_embedding, c.workflow_embedding)
            task_sim = self._cosine(task_features, c.task_features)
            score = 0.7 * wf_sim + 0.3 * task_sim
            sims.append((c, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

__all__ = ["WRENCHKnowledgeCase", "WRENCHRAGKnowledgeBase"]
