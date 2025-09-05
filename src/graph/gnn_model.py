"""GNN 占位模型 (伪训练)."""
from __future__ import annotations
from typing import Any
import random

class DummyGNN:
    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self._fitted = False

    def train(self, graph, labels, **kwargs):
        # 占位: 标记已训练
        self._fitted = True
        return {"loss": 0.0}

    def predict(self, graph) -> Any:
        # 占位: 为每个节点返回随机得分
        preds = {node: random.random() for node in graph.keys()}
        return preds
