import os
import sys

import torch
from torch_geometric.data import Data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.drl.gnn_encoder import DecoupledGNNEncoder


def _toy_graph(num_tasks: int = 3) -> Data:
    x = torch.randn((num_tasks, 4), dtype=torch.float32)
    edge_index = torch.tensor([
        list(range(num_tasks)),
        [(i + 1) % num_tasks for i in range(num_tasks)],
    ], dtype=torch.long)
    batch = torch.zeros(num_tasks, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch)


def test_retrieval_encoder_frozen_by_default():
    encoder = DecoupledGNNEncoder(in_channels=4, hidden_channels=8, out_channels=16)
    assert all(not p.requires_grad for p in encoder.retrieval_encoder.parameters())
    assert any(p.requires_grad for p in encoder.policy_encoder.parameters())


def test_sync_retrieval_copies_policy_weights():
    encoder = DecoupledGNNEncoder(in_channels=4, hidden_channels=8, out_channels=16)
    with torch.no_grad():
        for param in encoder.policy_encoder.parameters():
            param.add_(1.0)
    encoder.sync_retrieval_encoder()
    for policy_param, retrieval_param in zip(
        encoder.policy_encoder.parameters(),
        encoder.retrieval_encoder.parameters(),
    ):
        assert torch.allclose(policy_param, retrieval_param)


def test_policy_backward_leaves_retrieval_gradients_zero():
    encoder = DecoupledGNNEncoder(in_channels=4, hidden_channels=8, out_channels=16)
    graph = _toy_graph()
    encoder.policy_encoder.zero_grad()
    encoder.retrieval_encoder.zero_grad()
    loss = encoder(graph, encoder="policy").sum()
    loss.backward()
    assert any(param.grad is not None for param in encoder.policy_encoder.parameters())
    assert all(param.grad is None for param in encoder.retrieval_encoder.parameters())


def test_retrieval_forward_has_no_grad_requirements():
    encoder = DecoupledGNNEncoder(in_channels=4, hidden_channels=8, out_channels=16)
    graph = _toy_graph()
    with torch.no_grad():
        retrieval_embedding = encoder(graph, encoder="retrieval")
    assert retrieval_embedding.shape[-1] == 16
    assert not retrieval_embedding.requires_grad
