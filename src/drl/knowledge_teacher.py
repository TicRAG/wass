# src/drl/knowledge_teacher.py
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

class KnowledgeBase:
    """
    封装FAISS向量索引和元数据的知识库。
    """
    def __init__(self, dimension: int, storage_path: str = "data/knowledge_base"):
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.storage_path / "workflow_embeddings.index"
        self.meta_file = self.storage_path / "workflow_metadata.csv"
        
        # 初始化或加载
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
        """向知识库中添加新的经验轨迹"""
        if not hasattr(vectors, 'shape') or vectors.shape[0] == 0:
            print("⚠️ [KB] Attempted to add empty vectors. Skipping.")
            return
            
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        new_metadata = pd.DataFrame(metadata_list)
        self.metadata = pd.concat([self.metadata, new_metadata], ignore_index=True)

    def search(self, query_vector: np.ndarray, k: int = 5) -> pd.DataFrame:
        """检索与查询向量最相似的k个案例"""
        if self.index.ntotal == 0:
            return pd.DataFrame() # 如果知识库为空，返回空的DataFrame

        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)
        
        # 过滤掉无效的索引 (-1)
        valid_indices = indices[0][indices[0] != -1]
        if len(valid_indices) == 0:
            return pd.DataFrame()
            
        return self.metadata.iloc[valid_indices]

    def save(self):
        """保存知识库到磁盘"""
        print(f"💾 [KB] Saving Knowledge Base with {self.index.ntotal} entries...")
        faiss.write_index(self.index, str(self.index_file))
        self.metadata.to_csv(self.meta_file, index=False)
        print("✅ [KB] Knowledge Base saved.")


class PerformancePredictor(nn.Module):
    """一个简单的MLP，输入是手工设计的统计特征。"""
    def __init__(self, input_dim: int):
        super(PerformancePredictor, self).__init__()
        # 与 train_predictor.py 中的 SimplePredictor 结构保持一致
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

class KnowledgeableTeacher:
    """
    知识引导教师，负责生成RAG奖励。
    """
    def __init__(self, state_dim: int, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        # 注意：这里的输入维度应该与 state_embedding 的维度一致
        self.predictor = PerformancePredictor(input_dim=state_dim)
        
        # 加载预训练的性能预测器模型
        # 注意：这里的模型是用于GNN输出的，而不是train_predictor.py中的统计特征模型
        # 我们暂时假设有一个预训练好的模型，如果没有，它将以随机权重开始
        predictor_model_path = "models/saved_models/performance_predictor.pth"
        try:
            # 注意：这里的加载逻辑可能需要根据实际保存的模型结构调整
            # 这是一个简化的示例
            # self.predictor.load_state_dict(torch.load(predictor_model_path))
            # self.predictor.eval()
            print("✅ [Teacher] Performance predictor structure initialized.")
            print("⚠️ [Teacher] Note: Predictor is using initial random weights as it needs separate training.")
        except Exception as e:
            print(f"❌ [Teacher] Could not load performance predictor model: {e}. Using random weights.")


    def generate_rag_reward(self, state_embedding: torch.Tensor, current_action: int) -> float:
        """
        生成RAG奖励 (改进版)。
        奖励 = (检索到的历史案例的平均性能 - 预测的当前动作性能)
        """
        # 1. 检索相似案例
        # detach().numpy() 将其从计算图中分离并转换为numpy数组
        similar_cases = self.kb.search(state_embedding.detach().cpu().numpy(), k=5)
        
        if similar_cases.empty or 'makespan' not in similar_cases.columns:
            return 0.0 # 如果没有相似案例，不提供奖励

        # 2. 从元数据中获取历史性能
        historical_makespans = similar_cases['makespan'].values
        
        # 3. 计算奖励
        # 奖励核心思想：如果历史最优做法比平均好很多，那么这是一个值得探索的方向
        # 我们给予正奖励，鼓励智能体学习这种模式
        # 归一化奖励值，使其大小更稳定
        mean_perf = np.mean(historical_makespans)
        best_perf = np.min(historical_makespans)
        
        if mean_perf > 0:
            reward = (mean_perf - best_perf) / mean_perf
        else:
            reward = 0.0
        
        # 返回一个较小的、稳定的正奖励
        return float(reward)