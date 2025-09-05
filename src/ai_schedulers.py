#!/usr/bin/env python3
"""
WASS-RAG AI调度器核心实现
包含所有AI调度方法：WASS (Heuristic), WASS-DRL (w/o RAG), WASS-RAG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import faiss
import pickle
from pathlib import Path
import json

# PyTorch Geometric imports for GNN
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch_geometric not installed. GNN functionality will be limited.")

@dataclass
class SchedulingState:
    """调度状态表示"""
    workflow_graph: Dict[str, Any]
    cluster_state: Dict[str, Any] 
    pending_tasks: List[str]
    current_task: str
    available_nodes: List[str]
    timestamp: float

@dataclass
class SchedulingAction:
    """调度动作"""
    task_id: str
    target_node: str
    confidence: float
    reasoning: Optional[str] = None

class BaseScheduler:
    """基础调度器抽象类"""
    
    def __init__(self, name: str):
        self.name = name
        
    def make_decision(self, state: SchedulingState) -> SchedulingAction:
        """做出调度决策"""
        raise NotImplementedError
        
    def reset(self):
        """重置调度器状态"""
        pass

class WASSHeuristicScheduler(BaseScheduler):
    """WASS启发式调度器 - 基于多数票和数据局部性的规则"""
    
    def __init__(self):
        super().__init__("WASS (Heuristic)")
        
    def make_decision(self, state: SchedulingState) -> SchedulingAction:
        """基于启发式规则的调度决策"""
        
        task_id = state.current_task
        available_nodes = state.available_nodes
        
        if not available_nodes:
            raise ValueError("No available nodes for scheduling")
            
        # 获取任务信息
        task_info = self._get_task_info(state.workflow_graph, task_id)
        
        # 规则1: 数据局部性 - 优先选择有输入数据的节点
        data_locality_scores = self._calculate_data_locality_scores(
            task_info, available_nodes, state.cluster_state
        )
        
        # 规则2: 资源匹配 - 优先选择资源最匹配的节点
        resource_match_scores = self._calculate_resource_match_scores(
            task_info, available_nodes, state.cluster_state
        )
        
        # 规则3: 负载均衡 - 优先选择负载较低的节点
        load_balance_scores = self._calculate_load_balance_scores(
            available_nodes, state.cluster_state
        )
        
        # 多数票决策：综合所有规则
        final_scores = {}
        for node in available_nodes:
            # 加权组合各项得分
            final_scores[node] = (
                0.4 * data_locality_scores.get(node, 0) +
                0.3 * resource_match_scores.get(node, 0) +
                0.3 * load_balance_scores.get(node, 0)
            )
        
        # 选择得分最高的节点
        best_node = max(final_scores.keys(), key=lambda k: final_scores[k])
        confidence = final_scores[best_node]
        
        # 生成决策解释
        reasoning = f"Heuristic decision: data_locality={data_locality_scores.get(best_node, 0):.2f}, " \
                   f"resource_match={resource_match_scores.get(best_node, 0):.2f}, " \
                   f"load_balance={load_balance_scores.get(best_node, 0):.2f}"
        
        return SchedulingAction(
            task_id=task_id,
            target_node=best_node,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _get_task_info(self, workflow_graph: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """获取任务信息"""
        for task in workflow_graph.get("tasks", []):
            if task["id"] == task_id:
                return task
        raise ValueError(f"Task {task_id} not found in workflow")
    
    def _calculate_data_locality_scores(self, task_info: Dict[str, Any], 
                                      available_nodes: List[str], 
                                      cluster_state: Dict[str, Any]) -> Dict[str, float]:
        """计算数据局部性得分"""
        scores = {}
        
        # 简化的数据局部性计算
        for node in available_nodes:
            score = 0.5  # 基础得分
            
            # 检查是否有依赖任务的输出数据在此节点
            dependencies = task_info.get("dependencies", [])
            if dependencies:
                # 假设有30%的概率依赖数据在此节点
                score += 0.3 * len(dependencies) / max(len(dependencies), 1)
            
            scores[node] = min(score, 1.0)
            
        return scores
    
    def _calculate_resource_match_scores(self, task_info: Dict[str, Any],
                                       available_nodes: List[str],
                                       cluster_state: Dict[str, Any]) -> Dict[str, float]:
        """计算资源匹配得分"""
        scores = {}
        
        task_cpu_req = task_info.get("flops", 1e9) / 1e9  # 转换为GFlops
        task_memory_req = task_info.get("memory", 1e9) / 1e9  # 转换为GB
        
        for node in available_nodes:
            node_info = cluster_state.get("nodes", {}).get(node, {})
            node_cpu_capacity = node_info.get("cpu_capacity", 10.0)  # GFlops
            node_memory_capacity = node_info.get("memory_capacity", 8.0)  # GB
            
            # 计算资源利用率匹配度
            cpu_utilization = task_cpu_req / node_cpu_capacity
            memory_utilization = task_memory_req / node_memory_capacity
            
            # 理想利用率在60-80%之间
            cpu_score = 1.0 - abs(cpu_utilization - 0.7)
            memory_score = 1.0 - abs(memory_utilization - 0.7)
            
            scores[node] = max(0.0, (cpu_score + memory_score) / 2.0)
            
        return scores
    
    def _calculate_load_balance_scores(self, available_nodes: List[str],
                                     cluster_state: Dict[str, Any]) -> Dict[str, float]:
        """计算负载均衡得分"""
        scores = {}
        
        for node in available_nodes:
            node_info = cluster_state.get("nodes", {}).get(node, {})
            current_load = node_info.get("current_load", 0.5)  # 0-1之间
            
            # 负载越低，得分越高
            scores[node] = 1.0 - current_load
            
        return scores

class WASSSmartScheduler(BaseScheduler):
    """WASS-DRL智能调度器 (w/o RAG) - 标准DRL方法"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("WASS-DRL (w/o RAG)")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化GNN模型
        if HAS_TORCH_GEOMETRIC:
            self.gnn_encoder = GraphEncoder(
                node_feature_dim=8,
                edge_feature_dim=4,
                hidden_dim=64,
                output_dim=32
            ).to(self.device)
        else:
            self.gnn_encoder = None
            
        # 初始化策略网络
        self.policy_network = PolicyNetwork(
            state_dim=32,
            action_dim=1,
            hidden_dim=128
        ).to(self.device)
        
        # 加载预训练模型
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print(f"Warning: No pretrained model found at {model_path}, using random initialization")
    
    def make_decision(self, state: SchedulingState) -> SchedulingAction:
        """基于DRL策略的调度决策"""
        
        try:
            # 编码状态图
            if self.gnn_encoder is not None:
                state_embedding = self._encode_state_graph(state)
            else:
                # 如果没有torch_geometric，使用简化的特征提取
                state_embedding = self._extract_simple_features(state)
            
            # 计算每个可用节点的Q值
            node_scores = {}
            for node in state.available_nodes:
                # 将节点信息编码到状态中
                node_state = self._encode_node_context(state_embedding, node, state)
                
                # 获取Q值
                with torch.no_grad():
                    q_value = self.policy_network(node_state).item()
                
                node_scores[node] = q_value
            
            # 选择Q值最高的节点
            best_node = max(node_scores.keys(), key=lambda k: node_scores[k])
            confidence = torch.sigmoid(torch.tensor(node_scores[best_node])).item()
            
            reasoning = f"DRL decision: Q-values = {dict(sorted(node_scores.items(), key=lambda x: x[1], reverse=True))}"
            
            return SchedulingAction(
                task_id=state.current_task,
                target_node=best_node,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"Error in DRL decision making: {e}")
            # 降级到随机选择
            return SchedulingAction(
                task_id=state.current_task,
                target_node=np.random.choice(state.available_nodes),
                confidence=0.1,
                reasoning=f"Fallback to random due to error: {e}"
            )
    
    def _encode_state_graph(self, state: SchedulingState) -> torch.Tensor:
        """使用GNN编码状态图"""
        if self.gnn_encoder is None:
            return self._extract_simple_features(state)
            
        # 构建PyTorch Geometric图数据
        graph_data = self._build_graph_data(state)
        
        # GNN前向传播
        with torch.no_grad():
            embedding = self.gnn_encoder(graph_data)
            
        return embedding
    
    def _extract_simple_features(self, state: SchedulingState) -> torch.Tensor:
        """提取简化的状态特征（当没有torch_geometric时使用）"""
        features = []
        
        # 工作流特征
        total_tasks = len(state.workflow_graph.get("tasks", []))
        pending_tasks = len(state.pending_tasks)
        features.extend([total_tasks, pending_tasks, pending_tasks/max(total_tasks, 1)])
        
        # 集群特征
        total_nodes = len(state.available_nodes)
        avg_load = 0.5  # 简化假设
        features.extend([total_nodes, avg_load])
        
        # 当前任务特征
        task_info = None
        for task in state.workflow_graph.get("tasks", []):
            if task["id"] == state.current_task:
                task_info = task
                break
                
        if task_info:
            task_flops = task_info.get("flops", 1e9) / 1e10  # 归一化
            task_memory = task_info.get("memory", 1e9) / 1e10  # 归一化
            task_deps = len(task_info.get("dependencies", []))
            features.extend([task_flops, task_memory, task_deps])
        else:
            features.extend([0.1, 0.1, 0])
        
        # 填充到固定长度
        while len(features) < 32:
            features.append(0.0)
            
        return torch.tensor(features[:32], dtype=torch.float32, device=self.device)
    
    def _encode_node_context(self, state_embedding: torch.Tensor, node: str, state: SchedulingState) -> torch.Tensor:
        """编码节点上下文信息"""
        # 简化的节点特征
        node_features = [
            hash(node) % 100 / 100.0,  # 节点ID的简单编码
            0.5,  # 假设的负载
            1.0,  # 假设的可用性
        ]
        
        # 将节点特征与状态嵌入连接
        node_tensor = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        
        # 截断或填充状态嵌入到合适的大小
        if len(state_embedding) > 29:
            state_embedding = state_embedding[:29]
        else:
            padding = torch.zeros(29 - len(state_embedding), device=self.device)
            state_embedding = torch.cat([state_embedding, padding])
            
        combined = torch.cat([state_embedding, node_tensor])
        return combined
    
    def load_model(self, model_path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if self.gnn_encoder is not None:
                self.gnn_encoder.load_state_dict(checkpoint.get("gnn_encoder", {}))
            self.policy_network.load_state_dict(checkpoint.get("policy_network", {}))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")

class WASSRAGScheduler(BaseScheduler):
    """WASS-RAG调度器 - RAG增强的DRL方法"""
    
    def __init__(self, model_path: Optional[str] = None, knowledge_base_path: Optional[str] = None):
        super().__init__("WASS-RAG")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 继承智能调度器的能力
        self.base_scheduler = WASSSmartScheduler(model_path)
        
        # RAG组件
        self.knowledge_base = RAGKnowledgeBase(knowledge_base_path)
        self.performance_predictor = PerformancePredictor(
            input_dim=96,  # state + action + context
            hidden_dim=128
        ).to(self.device)
        
        # 加载性能预测器
        if model_path:
            self._load_performance_predictor(model_path)
    
    def make_decision(self, state: SchedulingState) -> SchedulingAction:
        """基于RAG增强的DRL决策"""
        
        try:
            # 1. 编码当前状态
            if self.base_scheduler.gnn_encoder is not None:
                state_embedding = self.base_scheduler._encode_state_graph(state)
            else:
                state_embedding = self.base_scheduler._extract_simple_features(state)
            
            # 2. 从知识库检索相似历史案例
            retrieved_context = self.knowledge_base.retrieve_similar_cases(
                state_embedding.cpu().numpy(), top_k=5
            )
            
            # 3. 为每个可用节点计算RAG增强的得分
            node_scores = {}
            historical_optimal = None
            
            for node in state.available_nodes:
                # 编码动作（节点选择）
                action_embedding = self._encode_action(node, state)
                
                # 预测性能
                predicted_makespan = self._predict_performance(
                    state_embedding, action_embedding, retrieved_context
                )
                
                # 存储用于比较
                node_scores[node] = -predicted_makespan  # 负值，因为我们要最小化makespan
                
                # 记录历史最优
                if historical_optimal is None or predicted_makespan < historical_optimal:
                    historical_optimal = predicted_makespan
            
            # 4. 选择预测性能最好的节点
            best_node = max(node_scores.keys(), key=lambda k: node_scores[k])
            
            # 5. 计算RAG奖励信号（实际用于训练时）
            rag_reward = self._calculate_rag_reward(node_scores, best_node, retrieved_context)
            
            # 6. 生成可解释的决策理由
            reasoning = self._generate_explanation(best_node, retrieved_context, node_scores)
            
            confidence = torch.sigmoid(torch.tensor(node_scores[best_node])).item()
            
            return SchedulingAction(
                task_id=state.current_task,
                target_node=best_node,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"Error in RAG decision making: {e}")
            # 降级到基础DRL方法
            return self.base_scheduler.make_decision(state)
    
    def _encode_action(self, node: str, state: SchedulingState) -> torch.Tensor:
        """编码调度动作"""
        # 简化的动作编码
        action_features = [
            hash(node) % 100 / 100.0,  # 节点ID
            len(state.available_nodes),  # 可用节点数
            1.0 if node == state.available_nodes[0] else 0.0,  # 是否是第一个选择
        ]
        
        # 填充到固定长度
        while len(action_features) < 32:
            action_features.append(0.0)
            
        return torch.tensor(action_features[:32], dtype=torch.float32, device=self.device)
    
    def _predict_performance(self, state_embedding: torch.Tensor, 
                           action_embedding: torch.Tensor,
                           context: Dict[str, Any]) -> float:
        """使用性能预测器预测makespan"""
        
        # 编码检索到的上下文
        context_embedding = self._encode_context(context)
        
        # 连接所有特征
        combined_features = torch.cat([
            state_embedding[:32],
            action_embedding[:32], 
            context_embedding[:32]
        ])
        
        # 预测性能
        with torch.no_grad():
            predicted_makespan = self.performance_predictor(combined_features).item()
            
        return max(predicted_makespan, 0.1)  # 确保非负
    
    def _encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        """编码检索到的历史上下文"""
        if not context or "similar_cases" not in context:
            return torch.zeros(32, device=self.device)
        
        # 提取历史案例的特征
        similar_cases = context["similar_cases"]
        if not similar_cases:
            return torch.zeros(32, device=self.device)
        
        # 简化的上下文特征
        features = []
        
        # 平均makespan
        makespans = [case.get("makespan", 100.0) for case in similar_cases]
        avg_makespan = np.mean(makespans) if makespans else 100.0
        min_makespan = np.min(makespans) if makespans else 100.0
        max_makespan = np.max(makespans) if makespans else 100.0
        
        features.extend([avg_makespan/100.0, min_makespan/100.0, max_makespan/100.0])
        
        # 案例数量
        features.append(len(similar_cases) / 10.0)
        
        # 平均相似度
        similarities = [case.get("similarity", 0.5) for case in similar_cases]
        avg_similarity = np.mean(similarities) if similarities else 0.5
        features.append(avg_similarity)
        
        # 填充到32维
        while len(features) < 32:
            features.append(0.0)
            
        return torch.tensor(features[:32], dtype=torch.float32, device=self.device)
    
    def _calculate_rag_reward(self, node_scores: Dict[str, float], 
                            chosen_node: str, context: Dict[str, Any]) -> float:
        """计算RAG奖励信号"""
        
        # 找到历史最优动作
        if context and "similar_cases" in context:
            historical_best_makespan = float('inf')
            for case in context["similar_cases"]:
                case_makespan = case.get("makespan", float('inf'))
                if case_makespan < historical_best_makespan:
                    historical_best_makespan = case_makespan
        else:
            historical_best_makespan = 100.0
        
        # 当前动作的预测makespan
        current_predicted_makespan = -node_scores[chosen_node]
        
        # RAG奖励 = 历史最优 - 当前预测 (越大越好)
        rag_reward = historical_best_makespan - current_predicted_makespan
        
        return rag_reward
    
    def _generate_explanation(self, chosen_node: str, context: Dict[str, Any], 
                            node_scores: Dict[str, float]) -> str:
        """生成可解释的决策说明"""
        
        explanation_parts = []
        
        # 基础决策信息
        explanation_parts.append(f"RAG-enhanced decision: chose node {chosen_node}")
        
        # 性能预测信息
        predicted_makespan = -node_scores[chosen_node]
        explanation_parts.append(f"predicted makespan: {predicted_makespan:.2f}s")
        
        # 历史案例信息
        if context and "similar_cases" in context:
            similar_cases = context["similar_cases"]
            if similar_cases:
                avg_historical_makespan = np.mean([case.get("makespan", 100.0) for case in similar_cases])
                explanation_parts.append(f"based on {len(similar_cases)} similar historical cases")
                explanation_parts.append(f"historical avg makespan: {avg_historical_makespan:.2f}s")
        
        # 所有节点的得分
        sorted_scores = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_scores[:3]
        scores_str = ", ".join([f"{node}:{score:.2f}" for node, score in top_3])
        explanation_parts.append(f"top scores: {scores_str}")
        
        return "; ".join(explanation_parts)
    
    def _load_performance_predictor(self, model_path: str):
        """加载性能预测器模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "performance_predictor" in checkpoint:
                self.performance_predictor.load_state_dict(checkpoint["performance_predictor"])
                print("Successfully loaded performance predictor")
        except Exception as e:
            print(f"Failed to load performance predictor: {e}")

# 神经网络组件
class GraphEncoder(nn.Module):
    """GNN状态编码器"""
    
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, 
                 hidden_dim: int, output_dim: int):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for GraphEncoder")
            
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)
        
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, graph_data: Data) -> torch.Tensor:
        x, edge_index = graph_data.x, graph_data.edge_index
        
        # 节点特征嵌入
        x = F.relu(self.node_embedding(x))
        
        # GNN层
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # 全局池化
        batch = getattr(graph_data, 'batch', None)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
            
        return x

class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class PerformancePredictor(nn.Module):
    """性能预测器网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # 确保输出非负（makespan）
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

class RAGKnowledgeBase:
    """RAG知识库"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None, embedding_dim: int = 32):
        self.knowledge_base_path = knowledge_base_path
        self.embedding_dim = embedding_dim  # 保存embedding维度
        self.index = None
        self.cases = []
        
        if knowledge_base_path and Path(knowledge_base_path).exists():
            self.load_knowledge_base(knowledge_base_path)
        else:
            self._initialize_empty_kb()
    
    def _initialize_empty_kb(self):
        """初始化空的知识库"""
        # 创建一个简单的FAISS索引
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
        self.cases = []
        print("Initialized empty knowledge base")
    
    def retrieve_similar_cases(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """检索相似的历史案例"""
        
        if self.index is None or self.index.ntotal == 0:
            # 返回空结果
            return {
                "similar_cases": [],
                "query_embedding": query_embedding.tolist(),
                "top_k": top_k
            }
        
        try:
            # 确保查询向量是正确的形状和连续内存布局
            query_vector = np.ascontiguousarray(
                query_embedding.reshape(1, -1), 
                dtype=np.float32
            )
            
            # 检索
            similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # 构建结果
            similar_cases = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.cases):
                    case = self.cases[idx].copy()
                    case["similarity"] = float(similarity)
                    case["rank"] = i + 1
                    similar_cases.append(case)
            
            return {
                "similar_cases": similar_cases,
                "query_embedding": query_embedding.tolist(),
                "top_k": top_k
            }
            
        except Exception as e:
            print(f"Error in knowledge base retrieval: {e}")
            return {
                "similar_cases": [],
                "query_embedding": query_embedding.tolist(),
                "top_k": top_k
            }
    
    def add_case(self, embedding: np.ndarray, workflow_info: Dict[str, Any],
                actions: List[str], makespan: float):
        """添加新案例到知识库（增强版，修复FAISS兼容性问题）"""

        case = {
            "workflow_info": workflow_info,
            "actions": actions,
            "makespan": makespan,
            "timestamp": str(np.datetime64('now'))
        }

        self.cases.append(case)

        if self.index is None:
            self._initialize_empty_kb()

        try:
            # 定义一个健壮的、多策略的函数来准备用于FAISS的数组
            def prepare_for_faiss(arr):
                """准备数组用于FAISS - 多重fallback策略"""
                # 策略1: 基础转换，确保是 float32 的 numpy 数组
                if isinstance(arr, list):
                    arr = np.array(arr, dtype=np.float32)
                else:
                    arr = np.asarray(arr, dtype=np.float32)

                # 策略2: 确保是2D数组
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

                # 策略3: 强制创建一个全新的、内存连续的数组副本。这是最关键的一步。
                # copy=True 确保我们得到的是一个新数组，而不是一个视图
                # order='C' 确保是C语言风格的连续内存
                arr = np.array(arr, dtype=np.float32, copy=True, order='C')

                return arr

            embedding_vector = prepare_for_faiss(embedding)

            # 验证数组属性，确保万无一失
            if not isinstance(embedding_vector, np.ndarray):
                raise ValueError(f"转换后仍然不是Numpy数组: {type(embedding_vector)}")
            if embedding_vector.dtype != np.float32:
                raise ValueError(f"数据类型错误: {embedding_vector.dtype}")
            if not embedding_vector.flags.c_contiguous:
                raise ValueError("数组内存布局不是C-contiguous")
            if embedding_vector.shape[1] != self.embedding_dim:
                raise ValueError(f"错误的 embedding 维度: {embedding_vector.shape[1]} vs {self.embedding_dim}")

            # 直接添加，因为 prepare_for_faiss 已经处理了所有已知问题
            self.index.add(embedding_vector)

        except Exception as e:
            print(f"向知识库添加案例时发生严重错误: {e}")
            print(f"  原始 embedding 类型: {type(embedding)}")
            print(f"  原始 embedding 形状: {getattr(embedding, 'shape', 'N/A')}")
            # 如果添加失败，将刚刚添加的 case 移除，保持数据一致性
            self.cases.pop()
            raise

    def load_knowledge_base(self, path: str):
        """加载知识库"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.index = data.get("index")
                self.cases = data.get("cases", [])
            print(f"Loaded knowledge base with {len(self.cases)} cases")
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")
            self._initialize_empty_kb()
    
    def save_knowledge_base(self, path: str):
        """保存知识库"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({
                    "index": self.index,
                    "cases": self.cases
                }, f)
            print(f"Saved knowledge base with {len(self.cases)} cases to {path}")
        except Exception as e:
            print(f"Failed to save knowledge base: {e}")

def create_scheduler(method_name: str, **kwargs) -> BaseScheduler:
    """工厂函数：创建调度器实例"""
    
    if method_name == "WASS (Heuristic)":
        return WASSHeuristicScheduler()
    elif method_name == "WASS-DRL (w/o RAG)":
        return WASSSmartScheduler(kwargs.get("model_path"))
    elif method_name == "WASS-RAG":
        return WASSRAGScheduler(
            kwargs.get("model_path"),
            kwargs.get("knowledge_base_path")
        )
    else:
        raise ValueError(f"Unknown scheduler method: {method_name}")
