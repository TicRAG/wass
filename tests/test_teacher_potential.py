import os
import tempfile

import numpy as np
import torch
from torch_geometric.data import Data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)

from src.rag.teacher import KnowledgeBase, KnowledgeableTeacher


class DummyEncoder(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.register_buffer("_embedding", torch.tensor(embedding, dtype=torch.float32))

    def forward(self, data: Data):  # pragma: no cover - simple deterministic stub
        return self._embedding.unsqueeze(0)


def _empty_graph(num_nodes: int = 2) -> Data:
    x = torch.zeros((num_nodes, 4), dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch)


def test_calculate_potential_weights_high_similarity():
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(dimension=2, storage_path=tmpdir)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        metadata = [
            {"workflow_file": "wf1", "scheduler_used": "HEFT", "q_value": 1.0},
            {"workflow_file": "wf2", "scheduler_used": "HEFT", "q_value": 0.2},
        ]
        kb.add(vectors, metadata)
        teacher = KnowledgeableTeacher(
            state_dim=2,
            knowledge_base=kb,
            gnn_encoder=DummyEncoder([1.0, 0.0]),
            reward_config={"top_k": 2, "temperature": 0.05, "scheduler_filter": "HEFT"},
        )
        potential = teacher.calculate_potential(_empty_graph())
        assert 0.95 <= potential <= 1.0  # heavily favors identical neighbor


def test_calculate_potential_respects_scheduler_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(dimension=2, storage_path=tmpdir)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        metadata = [
            {"workflow_file": "wf1", "scheduler_used": "HEFT", "q_value": 0.3},
            {"workflow_file": "wf2", "scheduler_used": "Random", "q_value": 0.9},
        ]
        kb.add(vectors, metadata)
        teacher = KnowledgeableTeacher(
            state_dim=2,
            knowledge_base=kb,
            gnn_encoder=DummyEncoder([0.0, 1.0]),
            reward_config={"top_k": 2, "scheduler_filter": "Random"},
        )
        potential = teacher.calculate_potential(_empty_graph())
        assert 0.85 <= potential <= 0.95  # selects Random entry despite HEFT presence


def test_calculate_potential_returns_zero_without_neighbors():
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(dimension=2, storage_path=tmpdir)
        teacher = KnowledgeableTeacher(
            state_dim=2,
            knowledge_base=kb,
            gnn_encoder=DummyEncoder([1.0, 0.0]),
            reward_config={"top_k": 1},
        )
        potential = teacher.calculate_potential(_empty_graph())
        assert potential == 0.0
