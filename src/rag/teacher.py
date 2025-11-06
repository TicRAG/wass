# src/drl/knowledge_teacher.py
import faiss
import json
import numpy as np
import pandas as pd
import threading
import time
import torch
import torch.nn as nn
from torch_geometric.data import Data
from pathlib import Path
from typing import Any, Dict, Optional


class TeacherTraceLogger:
    """Simple JSONL logger for teacher interpretability traces."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"TeacherTraceLogger(output_path={self.output_path!s})"

class KnowledgeBase:
    """封装FAISS向量索引和元数据的知识库。"""
    def __init__(self, dimension: int, storage_path: str = "data/knowledge_base"):
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.storage_path / "workflow_embeddings.index"
        self.meta_file = self.storage_path / "workflow_metadata.csv"
        
        if self.index_file.exists() and self.meta_file.exists():
            print("🧠 [Teacher] Loading existing Knowledge Base...")
            self.index = faiss.read_index(str(self.index_file))
            self.metadata = pd.read_csv(self.meta_file)
            if not isinstance(self.index, faiss.IndexFlatIP):
                print("⚠️ [Teacher] Existing FAISS index uses L2 metric; reinitializing with cosine metric. Rerun seeding to repopulate.")
                self.index = faiss.IndexFlatIP(dimension)
                self.metadata = pd.DataFrame()
            print("✅ [Teacher] Knowledge Base loaded.")
        else:
            print("⚠️ [Teacher] No existing Knowledge Base found. Initializing a new one.")
            self.index = faiss.IndexFlatIP(dimension)
            self.metadata = pd.DataFrame()

    def add(self, vectors: np.ndarray, metadata_list: list[dict]):
        if not hasattr(vectors, 'shape') or vectors.shape[0] == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        if not isinstance(self.index, faiss.IndexFlatIP):
            self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        new_metadata = pd.DataFrame(metadata_list)
        self.metadata = pd.concat([self.metadata, new_metadata], ignore_index=True)

    def search(self, query_vector: np.ndarray, k: int = 5) -> pd.DataFrame:
        if self.index.ntotal == 0:
            return pd.DataFrame()
        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        valid_mask = indices[0] != -1
        if not np.any(valid_mask):
            return pd.DataFrame()
        valid_indices = indices[0][valid_mask]
        similarities = distances[0][valid_mask]
        result = self.metadata.iloc[valid_indices].copy()
        result['similarity'] = similarities
        return result

    def save(self):
        print(f"💾 [KB] Saving Knowledge Base with {self.index.ntotal} entries...")
        faiss.write_index(self.index, str(self.index_file))
        self.metadata.to_csv(self.meta_file, index=False)
        print("✅ [KB] Knowledge Base saved.")

class KnowledgeableTeacher:
    """知识引导教师，负责生成RAG奖励并支持可解释性日志。"""

    def __init__(
        self,
        state_dim: int,
        knowledge_base: KnowledgeBase,
        gnn_encoder: nn.Module,
        reward_config: Optional[Dict[str, Any]] = None,
        trace_logger: Optional[TeacherTraceLogger] = None,
    ):
        self.kb = knowledge_base
        self.gnn_encoder = gnn_encoder
        self.gnn_encoder.eval()
        cfg = reward_config or {}
        self.top_k = int(cfg.get("top_k", 10))
        self.temperature = float(cfg.get("temperature", 0.1))
        scheduler_filter = cfg.get("scheduler_filter")
        self.scheduler_filter = scheduler_filter if scheduler_filter else None
        self.debug = bool(cfg.get("debug", False))
        self.lambda_scale = float(cfg.get("lambda", 1.0))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.fallback_use_all_if_empty = bool(cfg.get("fallback_use_all_if_empty", True))
        trace_cfg = cfg.get("trace_logging", {}) if isinstance(cfg.get("trace_logging", {}), dict) else {}
        self.trace_logger = trace_logger
        if trace_logger is None and trace_cfg.get("enabled"):
            output_path = trace_cfg.get("output_path")
            if output_path:
                self.trace_logger = TeacherTraceLogger(output_path)
                print(f"[TeacherTrace] Logging enabled -> {output_path}")
            else:
                print("[TeacherTrace] 'trace_logging.enabled' was true but no output_path provided; logging disabled.")
        self._trace_context: Dict[str, Any] = {}
        self._last_trace_payload: Optional[Dict[str, Any]] = None
        self._trace_neighbor_limit = int(trace_cfg.get("max_neighbors", self.top_k))

    def _select_neighbors(self, neighbors: pd.DataFrame) -> pd.DataFrame:
        if neighbors.empty:
            return neighbors
        if self.scheduler_filter:
            filtered = neighbors[neighbors['scheduler_used'] == self.scheduler_filter]
            if not filtered.empty:
                return filtered
            if not self.fallback_use_all_if_empty:
                return pd.DataFrame()
        return neighbors

    def enable_trace_logging(self, output_path: str | Path, max_neighbors: Optional[int] = None) -> None:
        self.trace_logger = TeacherTraceLogger(output_path)
        if max_neighbors is not None:
            self._trace_neighbor_limit = int(max_neighbors)
        print(f"[TeacherTrace] Logging enabled -> {self.trace_logger.output_path}")

    def set_trace_context(self, **context: Any) -> None:
        """Attach contextual metadata (workflow, episode, seed, etc.) for subsequent trace entries."""
        self._trace_context.update({k: v for k, v in context.items() if v is not None})

    def _build_trace_payload(
        self,
        neighbors: pd.DataFrame,
        weights: np.ndarray,
        potential: float,
        temperature: float,
    ) -> Dict[str, Any]:
        records = []
        limit = min(len(neighbors), self._trace_neighbor_limit)
        for idx in range(limit):
            row = neighbors.iloc[idx]
            q_value = row.get("q_value", 0.0)
            bias_multiplier = row.get("bias_multiplier", 1.0)
            if pd.isna(bias_multiplier):
                bias_multiplier = 1.0
            biased_q_value = row.get("biased_q_value", q_value)
            records.append({
                "workflow_file": row.get("workflow_file"),
                "scheduler_used": row.get("scheduler_used"),
                "q_value": float(q_value) if not pd.isna(q_value) else None,
                "similarity": float(row.get("similarity", 0.0)),
                "weight": float(weights[idx]) if idx < len(weights) else None,
                "bias_multiplier": float(bias_multiplier),
                "biased_q_value": float(biased_q_value) if not pd.isna(biased_q_value) else None,
            })
        payload: Dict[str, Any] = {
            "potential": float(potential),
            "temperature": float(temperature),
            "top_k": int(limit),
            "neighbors": records,
        }
        if self._trace_context:
            payload["context"] = dict(self._trace_context)
        return payload

    def calculate_potential(self, state: Data) -> float:
        if self.kb.index.ntotal == 0:
            return 0.0
        with torch.no_grad():
            query_embedding = self.gnn_encoder(state).detach().cpu().numpy().flatten()
        neighbors = self.kb.search(query_embedding, k=self.top_k)
        if neighbors.empty:
            if self.debug:
                print("[TeacherDebug] No neighbors found for potential calculation.")
            return 0.0
        neighbors = self._select_neighbors(neighbors)
        if neighbors.empty:
            if self.debug:
                print(f"[TeacherDebug] Scheduler filter '{self.scheduler_filter}' removed all neighbors.")
            return 0.0
        if 'q_value' not in neighbors.columns:
            if self.debug:
                print("[TeacherDebug] Knowledge records lack 'q_value'; returning zero potential.")
            return 0.0
        neighbors = neighbors.copy().reset_index(drop=True)
        similarities = neighbors['similarity'].to_numpy(dtype=np.float32)
        q_values = neighbors['q_value'].to_numpy(dtype=np.float32)
        if 'bias_multiplier' in neighbors.columns:
            multipliers = neighbors['bias_multiplier'].fillna(1.0).to_numpy(dtype=np.float32)
        else:
            multipliers = np.ones_like(q_values, dtype=np.float32)
            neighbors['bias_multiplier'] = multipliers
        biased_q_values = q_values * multipliers
        neighbors['biased_q_value'] = biased_q_values
        tau = max(self.temperature, 1e-6)
        scaled = similarities / tau
        scaled -= scaled.max()
        weights = np.exp(scaled)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            potential = float(biased_q_values.mean())
            weights = np.full_like(biased_q_values, fill_value=1.0 / len(biased_q_values)) if len(biased_q_values) else biased_q_values
        else:
            weights /= weights_sum
            potential = float(np.dot(weights, biased_q_values))
        self._last_trace_payload = None
        if self.trace_logger is not None and len(q_values) > 0:
            self._last_trace_payload = self._build_trace_payload(neighbors, weights, potential, tau)
        if self.debug:
            print(f"[TeacherDebug] potential={potential:.6f} neighbors={len(neighbors)} tau={tau}")
        return potential

    def generate_rag_reward(self, current_graph: Data, agent_eft: float, task_name: str) -> float:
        potential = self.calculate_potential(current_graph)
        shaped_reward = -self.lambda_scale * potential
        if self.debug:
            print(f"[TeacherDebug] task={task_name} potential={potential:.6f} reward={shaped_reward:.6f}")
        return shaped_reward