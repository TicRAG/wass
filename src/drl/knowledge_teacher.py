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
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            self.metadata = pd.read_csv(self.meta_file)
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = pd.DataFrame()

    def add(self, vectors: np.ndarray, metadata_list: list[dict]):
        """向知识库中添加新的经验轨迹"""
        self.index.add(vectors)
        new_metadata = pd.DataFrame(metadata_list)
        self.metadata = pd.concat([self.metadata, new_metadata], ignore_index=True)

    def search(self, query_vector: np.ndarray, k: int = 5) -> pd.DataFrame:
        """检索与查询向量最相似的k个案例"""
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        return self.metadata.iloc[indices[0]]

    def save(self):
        """保存知识库到磁盘"""
        faiss.write_index(self.index, str(self.index_file))
        self.metadata.to_csv(self.meta_file, index=False)

class PerformancePredictor(nn.Module):
    """一个简单的MLP，用于根据历史案例预测性能 (Makespan)"""
    def __init__(self, state_dim: int):
        super(PerformancePredictor, self).__init__()
        # 简化版：仅用状态预测，更复杂的版本可以加入动作和上下文
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 输出预测的Makespan
        )
    def forward(self, state):
        return self.model(state)

class KnowledgeableTeacher:
    """
    知识引导教师，负责生成RAG奖励。
    """
    def __init__(self, state_dim: int, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.predictor = PerformancePredictor(state_dim)
        # 注意：predictor需要被单独训练，这里我们先定义结构

    def generate_rag_reward(self, state_embedding: torch.Tensor, current_action) -> float:
        """
        生成RAG奖励 (简化版)。
        一个简单的实现可以是：检索到的历史案例中，最好的Makespan与平均Makespan的差异。
        """
        # 1. 检索相似案例
        similar_cases = self.kb.search(state_embedding.detach().numpy(), k=5)
        
        if similar_cases.empty:
            return 0.0 # 如果没有相似案例，不提供奖励
        
        # 2. 从元数据中获取历史性能
        historical_makespans = similar_cases['makespan'].values
        
        # 3. 计算奖励
        # 这里的奖励函数可以设计得很复杂。
        # 简化版：如果最好的历史记录远好于平均记录，则给予正奖励，鼓励探索。
        reward = np.mean(historical_makespans) - np.min(historical_makespans)
        
        # 占位符逻辑，待后续完善
        # TODO: 训练性能预测器，并实现论文中描述的奖励函数
        
        return float(reward / 100.0) # 归一化