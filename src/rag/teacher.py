# src/drl/knowledge_teacher.py
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from pathlib import Path
from typing import Any, Dict, Optional
import json

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
            print("✅ [Teacher] Knowledge Base loaded.")
        else:
            print("⚠️ [Teacher] No existing Knowledge Base found. Initializing a new one.")
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = pd.DataFrame()

    def add(self, vectors: np.ndarray, metadata_list: list[dict]):
        if not hasattr(vectors, 'shape') or vectors.shape[0] == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        new_metadata = pd.DataFrame(metadata_list)
        self.metadata = pd.concat([self.metadata, new_metadata], ignore_index=True)

    def search(self, query_vector: np.ndarray, k: int = 5) -> pd.DataFrame:
        if self.index.ntotal == 0:
            return pd.DataFrame()
        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)
        valid_indices = indices[0][indices[0] != -1]
        if len(valid_indices) == 0:
            return pd.DataFrame()
        return self.metadata.iloc[valid_indices]

    def save(self):
        print(f"💾 [KB] Saving Knowledge Base with {self.index.ntotal} entries...")
        faiss.write_index(self.index, str(self.index_file))
        self.metadata.to_csv(self.meta_file, index=False)
        print("✅ [KB] Knowledge Base saved.")

class KnowledgeableTeacher:
    """知识引导教师，负责生成RAG奖励。现在自行对传入的标准化图执行冻结的GNN编码。"""
    def __init__(self, state_dim: int, knowledge_base: KnowledgeBase, gnn_encoder: nn.Module, reward_config: Optional[Dict[str, Any]] = None):
        self.kb = knowledge_base
        self.gnn_encoder = gnn_encoder
        self.gnn_encoder.eval()
        cfg = reward_config or {}
        self.top_k = int(cfg.get("top_k", 10))
        self.scheduler_filter = cfg.get("scheduler_filter", "HEFT")
        normalizer = cfg.get("reward_normalizer", 1000.0)
        self.reward_normalizer = float(normalizer) if normalizer not in (None, 0) else 1.0
        # Debug flags
        self.debug = bool(cfg.get("debug", False))
        # Fallback: if filtered cases empty, optionally use all similar cases
        self.fallback_use_all_if_empty = bool(cfg.get("fallback_use_all_if_empty", True))

    # --- 这是最终的奖励函数 ---
    def generate_rag_reward(self, current_graph: Data, agent_eft: float, task_name: str) -> float:
        """根据当前标准化图（已将 COMPLETED 状态还原为 WAITING/READY）生成 RAG 奖励。包含可选调试输出和回退逻辑。"""
        with torch.no_grad():
            emb = self.gnn_encoder(current_graph).detach().cpu().numpy().flatten()
        similar_cases = self.kb.search(emb, k=self.top_k)

        if similar_cases.empty:
            if self.debug:
                print(f"[TeacherDebug] similar_cases empty (index_size={self.kb.index.ntotal}) task={task_name}")
            return 0.0

        heft_cases = similar_cases[similar_cases['scheduler_used'] == self.scheduler_filter]
        used_cases = heft_cases
        if heft_cases.empty:
            if self.fallback_use_all_if_empty:
                used_cases = similar_cases
                if self.debug:
                    print(f"[TeacherDebug] No cases after filter '{self.scheduler_filter}'. Fallback to all similar. count={len(similar_cases)} task={task_name}")
            else:
                if self.debug:
                    print(f"[TeacherDebug] heft_cases empty after filter '{self.scheduler_filter}' task={task_name}")
                return 0.0

        historical_efts = []
        unmatched_samples = []
        for _, row in used_cases.iterrows():
            try:
                decisions = json.loads(row['decisions'])
                if task_name in decisions:
                    historical_efts.append(decisions[task_name].get('finish_time', decisions[task_name].get('eft', 0.0)))
                else:
                    if len(unmatched_samples) < 5:
                        unmatched_samples.append(list(decisions.keys())[:5])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        if not historical_efts:
            if self.debug:
                sample_str = '; '.join([','.join(s) for s in unmatched_samples]) if unmatched_samples else 'N/A'
                print(f"[TeacherDebug] No historical EFTs for task={task_name}. Sample decision key sets: {sample_str}")
            return 0.0

        best_historical_eft = np.min(historical_efts)
        reward = (best_historical_eft - agent_eft) / self.reward_normalizer

        if self.debug:
            print(f"[TeacherDebug] task={task_name} similar={len(similar_cases)} used={len(used_cases)} matches={len(historical_efts)} best_hist_eft={best_historical_eft:.3f} agent_eft={agent_eft:.3f} reward={reward:.6f}")
        return reward