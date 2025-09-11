from typing import List, Dict, Any
import numpy as np


class SimpleRAGCase:
    __slots__ = ("graph_emb", "task_flops", "action", "exec_time", "node_capacity")

    def __init__(self, graph_emb: np.ndarray, task_flops: float, action: int, exec_time: float, node_capacity: float):
        self.graph_emb = graph_emb.astype(np.float32)
        self.task_flops = float(task_flops)
        self.action = int(action)
        self.exec_time = float(exec_time)
        self.node_capacity = float(node_capacity)


class RAGTeacher:
    """Retrieval augmented reward shaping.

    reward = scale * (alpha * ratio + (1 - alpha) * vote)
      ratio:   best_predicted_ETC / chosen_ETC    (∈ (0,1])
      vote:    fraction of retrieved similar cases whose action == chosen action (0..1)
    Keeps reward bounded in [0, scale].
    """

    def __init__(self, use_predictor=None, top_k: int = 5):
        self.use_predictor = use_predictor
        self.top_k = top_k
        self.cases: List[SimpleRAGCase] = []

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)

    def add_case(self, graph_emb: np.ndarray, task_flops: float, action: int, exec_time: float, node_capacity: float):
        self.cases.append(SimpleRAGCase(graph_emb, task_flops, action, exec_time, node_capacity))

    def _retrieve(self, graph_emb: np.ndarray) -> List[SimpleRAGCase]:
        if not self.cases:
            return []
        scores = [(self._cosine(graph_emb, c.graph_emb), c) for c in self.cases]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scores[: self.top_k]]

    def predict_exec_times(self, graph_emb: np.ndarray, task_flops: float, node_capacities: List[float]) -> List[float]:
        base = [task_flops / max(1e-6, cap * 1e9) for cap in node_capacities]
        retrieved = self._retrieve(graph_emb)
        if not retrieved:
            return base
        ratios: Dict[int, List[float]] = {}
        for c in retrieved:
            analytical = c.task_flops / max(1e-6, c.node_capacity * 1e9)
            ratios.setdefault(c.action, []).append(c.exec_time / max(1e-6, analytical))
        adjusted: List[float] = []
        for i, t in enumerate(base):
            if i in ratios and ratios[i]:
                median_ratio = float(np.median(ratios[i]))
                adjusted.append(t * median_ratio)
            else:
                adjusted.append(t)
        return adjusted

    def rag_reward(
        self,
        graph_emb: np.ndarray,
        task_flops: float,
        node_capacities: List[float],
        chosen_action: int,
        scale: float = 0.3,
        alpha: float = 0.5,
    ) -> float:
        try:
            preds = self.predict_exec_times(graph_emb, task_flops, node_capacities)
            if not preds or chosen_action >= len(preds):
                return 0.0
            best = min(preds)
            chosen = preds[chosen_action]
            ratio = float(np.clip(best / (chosen + 1e-8), 0.0, 1.0))
            retrieved = self._retrieve(graph_emb)
            if retrieved:
                vote = sum(1 for c in retrieved if c.action == chosen_action) / len(retrieved)
            else:
                vote = 0.0
            vote = float(np.clip(vote, 0.0, 1.0))
            alpha = float(np.clip(alpha, 0.0, 1.0))
            return scale * (alpha * ratio + (1 - alpha) * vote)
        except Exception:
            return 0.0

