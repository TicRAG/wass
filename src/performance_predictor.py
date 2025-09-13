import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class PerformancePredictor(nn.Module):
    """Graph Neural Network based performance predictor that understands workflow topology"""
    def __init__(self, node_feat_dim=8, hidden_dim=128, model_path: Optional[str] = None):
        super().__init__()
        
        # Graph Convolutional Network for processing workflow DAGs
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim//2)
        self.conv3 = GCNConv(hidden_dim//2, hidden_dim//4)
        
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//8, 1)
        )
        
        # Load pretrained weights if provided
        if model_path:
            try:
                state = torch.load(model_path, map_location="cpu", weights_only=False)
                if isinstance(state, dict) and 'performance_predictor' in state:
                    self.load_state_dict(state['performance_predictor'])
            except Exception:
                pass

class SimplePerformancePredictor(nn.Module):
    """Simple feedforward neural network for performance prediction from flat features"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, model_path: Optional[str] = None):
        super().__init__()
        
        # Simple feedforward network for flat feature vectors
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//4, 1)
        )
        
        # Load pretrained weights if provided
        if model_path:
            try:
                state = torch.load(model_path, map_location="cpu", weights_only=False)
                if isinstance(state, dict) and 'performance_predictor' in state:
                    self.load_state_dict(state['performance_predictor'])
            except Exception:
                pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with feedforward layers"""
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> float:
        """Predict interface for flat feature vectors."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x).item()
        return out
    
    def extract_graph_features(self, dag: nx.DiGraph, node_features: Dict[str, Dict], focus_task_id: str) -> Data:
        """Convert workflow DAG to PyTorch Geometric Data object with rich features.
        
        Args:
            dag: NetworkX DiGraph representing workflow
            node_features: Dict mapping node names to their features
            focus_task_id: ID of the task we're currently scheduling
            
        Returns:
            PyG Data object ready for GNN processing
        """
        # Create node feature matrix
        node_ids = list(dag.nodes())
        node_idx_map = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Node features: [computation_size, is_submitted, is_finished, assigned_host_id,
        #                 focus_task_indicator, node_speed, available_time, queue_length]
        num_nodes = len(node_ids)
        node_feat_dim = 8  # Update to match the actual feature count
        x = torch.zeros((num_nodes, node_feat_dim), dtype=torch.float32)
        
        for i, node_id in enumerate(node_ids):
            task = dag.nodes[node_id]
            
            # Basic task features
            computation_size = task.get('computation_size', 0.0)
            
            # Status indicators (0.0 or 1.0)
            is_submitted = float(task.get('is_submitted', False))
            is_finished = float(task.get('is_finished', False))
            
            # Host assignment (-1 if unassigned)
            assigned_host_id = float(task.get('assigned_host_id', -1))
            
            # Focus task indicator
            is_focus_task = 1.0 if node_id == focus_task_id else 0.0
            
            # Node/platform features (if available)
            node_feat = node_features.get(node_id, {})
            node_speed = node_feat.get('speed', 0.0)
            available_time = node_feat.get('available_time', 0.0)
            queue_length = float(node_feat.get('queue_length', 0))
            
            x[i] = torch.tensor([
                computation_size,
                is_submitted,
                is_finished,
                assigned_host_id,
                is_focus_task,
                node_speed,
                available_time,
                queue_length
            ], dtype=torch.float32)
        
        # Create edge index tensor
        edges = list(dag.edges())
        if edges:
            edge_index = torch.tensor([[node_idx_map[src], node_idx_map[dst]] for src, dst in edges], 
                                      dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Edge features: data_size
        if edges:
            edge_attr = torch.tensor([[dag.edges[src, dst].get('data_size', 0.0)] for src, dst in edges], 
                                     dtype=torch.float32)
        else:
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class RAGKnowledgeBase:
    """Simple in-memory vector store for state embeddings.

    Stores (embedding, meta, actions, outcome) tuples. Retrieval uses cosine similarity.
    """
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.cases: List[Dict[str, Any]] = []

    def add_case(self, embedding: np.ndarray, meta: Dict[str, Any], actions: List[Any], outcome: float):
        if embedding.shape[-1] != self.embedding_dim:
            # Auto-adjust on first insert
            if not self.cases:
                self.embedding_dim = embedding.shape[-1]
            else:
                raise ValueError("Embedding dim mismatch")
        self.cases.append({
            'embedding': embedding.astype(np.float32),
            'meta': meta,
            'actions': actions,
            'outcome': float(outcome)
        })

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)

    def retrieve_similar_cases(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.cases:
            return []
        scores = []
        for c in self.cases:
            scores.append((self._cosine(query_embedding, c['embedding']), c))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scores[:top_k]]

    def size(self) -> int:
        return len(self.cases)
