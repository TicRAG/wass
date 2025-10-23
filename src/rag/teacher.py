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

    # --- 这是最终的奖励函数 ---
    def generate_rag_reward(self, current_graph: Data, agent_eft: float, task_name: str) -> float:
        """根据当前标准化图（已将 COMPLETED 状态还原为 WAITING/READY）生成 RAG 奖励。"""
        with torch.no_grad():
            emb = self.gnn_encoder(current_graph).detach().cpu().numpy().flatten()
        similar_cases = self.kb.search(emb, k=self.top_k)
        
        if similar_cases.empty:
            return 0.0

        # 2. 筛选出由HEFT调度器产生的、“好的”历史案例
        heft_cases = similar_cases[similar_cases['scheduler_used'] == self.scheduler_filter]
        if heft_cases.empty:
            return 0.0

        # 3. 从这些好的案例中，找到针对当前任务的历史决策
        historical_efts = []
        for _, row in heft_cases.iterrows():
            try:
                decisions = json.loads(row['decisions'])
                if task_name in decisions:
                    historical_efts.append(decisions[task_name]['finish_time'])
            except (json.JSONDecodeError, KeyError):
                continue
        
        if not historical_efts:
            return 0.0  # 知识库中没有关于这个任务的HEFT决策记录

        # 4. 找到历史上的最优EFT
        best_historical_eft = np.min(historical_efts)

        # 5. 计算奖励 (并进行缩放，使其数值稳定)
        reward = (best_historical_eft - agent_eft) / self.reward_normalizer
        return reward