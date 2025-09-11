import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class GraphFeatureEncoder(nn.Module):
    """Lightweight GNN-style encoder without external deps.

    Treats workflow as a DAG. We build an adjacency matrix A (parents->children).
    Performs K rounds of message passing: h^{k+1} = ReLU( W_self h^k + W_msg (A h^k) ).
    Finally returns a global graph embedding via mean + max pooling concat.
    """
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64, layers: int = 2):
        super().__init__()
        self.layers = layers
        self.self_linears = nn.ModuleList()
        self.msg_linears = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.self_linears.append(nn.Linear(last, hidden_dim))
            self.msg_linears.append(nn.Linear(last, hidden_dim))
            last = hidden_dim
        self.out_dim = hidden_dim * 2  # mean + max pooling concatenated

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """node_feat: [N, F]; adj: [N, N] binary (row i -> children indices j where adj[i,j]=1)."""
        h = node_feat
        for w_self, w_msg in zip(self.self_linears, self.msg_linears):
            # Aggregate messages from parents: we want children to receive from parents.
            # Given adj[parent, child] = 1, parents influence child; so parent messages = A^T h
            parent_agg = torch.matmul(adj.T, h)  # [N,F]
            h = F.relu(w_self(h) + w_msg(parent_agg))
        mean_pool = h.mean(dim=0)
        max_pool, _ = h.max(dim=0)
        return torch.cat([mean_pool, max_pool], dim=0)  # [2H]

    @staticmethod
    def build_graph(workflow_or_tasks) -> Dict[str, Any]:
        """Extract simple node features + adjacency.

        Accepts either a workflow object with get_tasks() or a list of task objects.
        Gracefully degrades if task entries are plain strings (no attributes) by
        returning a single zero feature node to avoid runtime crashes.
        Each task feature: [norm_flops, in_deg, out_deg, is_source, is_sink].
        """
        # Determine task list
        if isinstance(workflow_or_tasks, list):
            tasks = workflow_or_tasks
        elif hasattr(workflow_or_tasks, 'get_tasks'):
            try:
                tasks = workflow_or_tasks.get_tasks()
                # Some WRENCH versions may return a dict {name:task}; normalize
                if isinstance(tasks, dict):
                    tasks = list(tasks.values())
            except Exception:
                tasks = []
        else:
            tasks = []

        # If tasks are strings or empty, fallback
        if not tasks:
            return {
                'node_feat': torch.zeros((1, 5), dtype=torch.float32),
                'adj': torch.zeros((1, 1), dtype=torch.float32),
                'index': {}
            }
        # Guard against non-task entries
        first = tasks[0]
        if isinstance(first, str):
            return {
                'node_feat': torch.zeros((1, 5), dtype=torch.float32),
                'adj': torch.zeros((1, 1), dtype=torch.float32),
                'index': {}
            }

        # Build index map
        try:
            idx = {t.get_name(): i for i, t in enumerate(tasks)}
        except AttributeError:
            # Fallback if objects unexpectedly lack get_name
            return {
                'node_feat': torch.zeros((1, 5), dtype=torch.float32),
                'adj': torch.zeros((1, 1), dtype=torch.float32),
                'index': {}
            }

        N = len(tasks)
        import numpy as np  # local to avoid global dependency if unused elsewhere
        flops_vals = []
        for t in tasks:
            try:
                flops_vals.append(t.get_flops())
            except Exception:
                flops_vals.append(1.0)
        max_flops = max(flops_vals) or 1.0

        node_feat_rows: List[List[float]] = []
        adj = torch.zeros((N, N), dtype=torch.float32)
        for t in tasks:
            try:
                name = t.get_name()
            except Exception:
                continue
            i = idx.get(name, None)
            if i is None:
                continue
            # Children collection
            children = []
            if hasattr(t, 'get_children'):
                try:
                    children = t.get_children()
                except Exception:
                    children = []
            for c in children:
                try:
                    cname = c.get_name()
                    if cname in idx:
                        adj[i, idx[cname]] = 1.0
                except Exception:
                    continue
            # Degrees
            if hasattr(t, 'get_parents'):
                try:
                    in_deg = len(t.get_parents())
                except Exception:
                    in_deg = 0
            else:
                in_deg = 0
            out_deg = len(children)
            is_source = 1.0 if in_deg == 0 else 0.0
            is_sink = 1.0 if out_deg == 0 else 0.0
            # FLOPS normalized
            try:
                flops = t.get_flops()/max_flops
            except Exception:
                flops = 0.0
            node_feat_rows.append([
                float(flops),
                float(in_deg),
                float(out_deg),
                is_source,
                is_sink
            ])

        if not node_feat_rows:
            return {
                'node_feat': torch.zeros((1, 5), dtype=torch.float32),
                'adj': torch.zeros((1, 1), dtype=torch.float32),
                'index': {}
            }

        node_feat = torch.tensor(node_feat_rows, dtype=torch.float32)
        return {'node_feat': node_feat, 'adj': adj, 'index': idx}
