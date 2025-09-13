import math
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

class PerformancePredictor(nn.Module):
    """Simple feedforward neural network predictor"""
    def __init__(self, input_dim=32, hidden_dim=128, model_path: Optional[str] = None):
        super().__init__()
        
        # Simple feedforward network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 1)
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
        return self.net(x)

    def predict(self, x_np: np.ndarray) -> np.ndarray:
        """Sklearn-like predict interface."""
        self.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x_np.astype(np.float32))
            out = self.forward(x_t).squeeze(-1).cpu().numpy()
        return out

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
