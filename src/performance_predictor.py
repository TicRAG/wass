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
    def __init__(self, node_feat_dim=12, hidden_dim=256, model_path: Optional[str] = None):
        super().__init__()
        
        # Enhanced Graph Convolutional Network for processing workflow DAGs
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim//2)
        self.bn3 = nn.BatchNorm1d(hidden_dim//2)
        self.conv4 = GCNConv(hidden_dim//2, hidden_dim//4)
        
        # Global mean pooling for graph-level representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Enhanced prediction layers with skip connections
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim//4 + hidden_dim//2, hidden_dim//4),  # Skip connection
            nn.ReLU(),
            nn.Dropout(0.2),
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
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through GNN layers.
        
        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for graph-level prediction
        
        Returns:
            Predicted makespan value
        """
        # Graph convolution layers with residual connections
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = torch.relu(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = torch.relu(x2)
        
        # Skip connection
        x2 = x2 + x1
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = torch.relu(x3)
        
        x4 = self.conv4(x3, edge_index)
        x4 = torch.relu(x4)
        
        # Global mean pooling for graph-level representation
        if batch is None:
            # If no batch provided, assume single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Pool node embeddings to get graph representation
        graph_emb = torch.zeros(batch.max().item() + 1, x4.size(1), device=x.device)
        graph_emb = graph_emb.scatter_add_(0, batch.unsqueeze(1).expand(-1, x4.size(1)), x4)
        graph_emb = graph_emb / torch.bincount(batch).unsqueeze(1).float()
        
        # Concatenate with high-level features (skip connection)
        high_level_emb = torch.zeros(batch.max().item() + 1, x3.size(1), device=x.device)
        high_level_emb = high_level_emb.scatter_add_(0, batch.unsqueeze(1).expand(-1, x3.size(1)), x3)
        high_level_emb = high_level_emb / torch.bincount(batch).unsqueeze(1).float()
        
        combined_emb = torch.cat([graph_emb, high_level_emb], dim=1)
        
        # Final prediction
        output = self.predictor(combined_emb)
        
        return output
    
    def predict(self, data: Data) -> float:
        """Predict interface for PyG Data objects.
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            Predicted makespan value
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(data.x, data.edge_index, data.batch)
        return out.item()

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
        
        # Enhanced node features: [computation_size, is_submitted, is_finished, assigned_host_id,
        #                 focus_task_indicator, node_speed, available_time, queue_length,
        #                 critical_path_indicator, children_count, parents_count, data_locality]
        num_nodes = len(node_ids)
        node_feat_dim = 12  # Enhanced feature dimension
        x = torch.zeros((num_nodes, node_feat_dim), dtype=torch.float32)
        
        # Calculate critical path information
        critical_path = self._calculate_critical_path(dag)
        
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
            
            # Node/platform features
            node_feat = node_features.get(node_id, {})
            node_speed = node_feat.get('speed', 0.0)
            available_time = node_feat.get('available_time', 0.0)
            queue_length = float(node_feat.get('queue_length', 0))
            
            # Topological features
            children_count = float(len(task.get('children', [])))
            parents_count = float(len(task.get('parents', [])))
            
            # Critical path indicator
            is_critical = 1.0 if node_id in critical_path else 0.0
            
            # Data locality score (simplified)
            data_locality = self._calculate_data_locality(dag, node_id, task)
            
            x[i] = torch.tensor([
                computation_size,
                is_submitted,
                is_finished,
                assigned_host_id,
                is_focus_task,
                node_speed,
                available_time,
                queue_length,
                is_critical,
                children_count,
                parents_count,
                data_locality
            ], dtype=torch.float32)
        
        # Create edge index tensor
        edges = list(dag.edges())
        if edges:
            edge_index = torch.tensor([[node_idx_map[src], node_idx_map[dst]] for src, dst in edges], 
                                      dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Enhanced edge features: [data_size, communication_cost, dependency_strength]
        if edges:
            edge_features = []
            for src, dst in edges:
                data_size = dag.edges[src, dst].get('data_size', 0.0)
                communication_cost = self._calculate_communication_cost(dag, src, dst)
                dependency_strength = self._calculate_dependency_strength(dag, src, dst)
                
                edge_features.append([data_size, communication_cost, dependency_strength])
            
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_attr = torch.empty((0, 3), dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _calculate_critical_path(self, dag: nx.DiGraph) -> List[str]:
        """Calculate the critical path in the DAG."""
        try:
            # Find longest path (critical path)
            if nx.is_directed_acyclic_graph(dag):
                # Use longest path algorithm
                longest_path = nx.dag_longest_path(dag, weight='computation_size')
                return longest_path
            else:
                return []
        except Exception:
            return []
    
    def _calculate_data_locality(self, dag: nx.DiGraph, node_id: str, task: Dict) -> float:
        """Calculate data locality score for a task."""
        # Simplified data locality calculation
        # Higher score means better data locality
        parents = task.get('parents', [])
        if not parents:
            return 1.0  # Root tasks have perfect locality
        
        # Check if parents are assigned to the same host
        parent_hosts = [dag.nodes[p].get('assigned_host_id', -1) for p in parents]
        task_host = task.get('assigned_host_id', -1)
        
        same_host_count = sum(1 for host in parent_hosts if host == task_host)
        return same_host_count / len(parents)
    
    def _calculate_communication_cost(self, dag: nx.DiGraph, src: str, dst: str) -> float:
        """Calculate communication cost between tasks."""
        data_size = dag.edges[src, dst].get('data_size', 0.0)
        # Simplified communication cost
        return data_size / 1e9  # Normalize by 1GB
    
    def _calculate_dependency_strength(self, dag: nx.DiGraph, src: str, dst: str) -> float:
        """Calculate dependency strength between tasks."""
        # Stronger dependencies have higher data transfer relative to computation
        src_task = dag.nodes[src]
        dst_task = dag.nodes[dst]
        
        data_size = dag.edges[src, dst].get('data_size', 0.0)
        dst_computation = dst_task.get('computation_size', 1.0)
        
        return data_size / (dst_computation + 1e-6)

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
