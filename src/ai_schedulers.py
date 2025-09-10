#!/usr/bin/env python3
"""
WASS-RAG AI调度器核心实现 (V5 - 时序感知修复版)
包含所有AI调度方法，并增强了特征工程，以充分利用离散事件仿真框架提供的动态时序信息。
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
        raise NotImplementedError
    def reset(self):
        pass

class WASSHeuristicScheduler(BaseScheduler):
    """WASS启发式调度器 - 简单的规则基调度器"""
    def __init__(self):
        super().__init__("WASS (Heuristic)")

    def make_decision(self, state: SchedulingState) -> SchedulingAction:
        """
        基于最早完成时间（Earliest Finish Time, EFT）的启发式决策。
        这是比之前更健壮和经典的启发式方法。
        """
        task_id = state.current_task
        task_info = self._get_task_info(state.workflow_graph, task_id)
        
        best_node = None
        earliest_finish_time = float('inf')

        for node in state.available_nodes:
            est = state.cluster_state['earliest_start_times'].get(node, 0)
            node_info = state.cluster_state.get("nodes", {}).get(node, {})
            node_cpu_gflops = node_info.get("cpu_capacity", 1.0)
            
            exec_time = task_info.get("flops", 1e9) / (node_cpu_gflops * 1e9)
            finish_time = est + exec_time
            
            if finish_time < earliest_finish_time:
                earliest_finish_time = finish_time
                best_node = node
        
        if best_node is None:
            best_node = state.available_nodes[0]

        return SchedulingAction(
            task_id=task_id,
            target_node=best_node,
            confidence=0.9, # Heuristic is always confident
            reasoning=f"Heuristic chose {best_node} with EFT: {earliest_finish_time:.2f}s"
        )

    def _get_task_info(self, workflow_graph: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        tasks = workflow_graph.get("tasks", [])
        for task in tasks:
            if isinstance(task, dict) and task.get("id") == task_id:
                return task
        return {"flops": 2e9, "memory": 4.0, "dependencies": []}


class WASSSmartScheduler(BaseScheduler):
    """WASS-DRL智能调度器 (w/o RAG) - 标准DRL方法"""
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("WASS-DRL (w/o RAG)")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.heuristic_scheduler = WASSHeuristicScheduler()
        # 创建一个临时的 RAG 调度器实例，仅用于借用其强大的特征编码器
        self.temp_rag_scheduler = WASSRAGScheduler() 

        if HAS_TORCH_GEOMETRIC:
            self.gnn_encoder = GraphEncoder(node_feature_dim=8, edge_feature_dim=4, hidden_dim=64, output_dim=32).to(self.device)
        else:
            self.gnn_encoder = None

        self.policy_network = PolicyNetwork(state_dim=32 + 32, hidden_dim=128).to(self.device)

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print(f"Warning: No pretrained model found at {model_path}, using random initialization")

    def make_decision(self, state: SchedulingState) -> SchedulingAction:
        """基于DRL的调度决策 (WASS-DRL)"""
        try:
            state_embedding = self._encode_state_graph(state)
            node_scores = {}
            with torch.no_grad():
                for node in state.available_nodes:
                    # 为了公平比较，DRL调度器使用与RAG相同的、增强后的特征编码器
                    action_embedding = self._encode_action_for_drl(node, state)
                    combined_embedding = torch.cat([state_embedding.flatten(), action_embedding.flatten()])
                    score = self.policy_network(combined_embedding).item()
                    node_scores[node] = score

            if not node_scores: raise ValueError("No nodes available for scoring.")

            best_node = max(node_scores, key=node_scores.get)
            confidence = torch.sigmoid(torch.tensor(node_scores[best_node])).item()
            reasoning = f"DRL decision: Node {best_node} has the highest score ({node_scores[best_node]:.3f})"
            return SchedulingAction(task_id=state.current_task, target_node=best_node, confidence=confidence, reasoning=reasoning)

        except Exception as e:
            print(f"  [DEGRADATION] DRL decision failed: {e}. Falling back to heuristic.")
            fallback_action = self.heuristic_scheduler.make_decision(state)
            fallback_action.reasoning = f"DEGRADED: DRL->Heuristic fallback due to error: {e}"
            return fallback_action

    def _encode_state_graph(self, state: SchedulingState) -> torch.Tensor:
        if self.gnn_encoder:
            graph_data = self._build_graph_data(state)
            if graph_data:
                with torch.no_grad(): return self.gnn_encoder(graph_data)
        return self.temp_rag_scheduler._extract_simple_features_fallback(state)

    def _encode_action_for_drl(self, node: str, state: SchedulingState) -> torch.Tensor:
        # 为了公平比较，DRL 使用与 RAG 相同的、强大的时序感知特征编码器
        return self.temp_rag_scheduler._encode_action(node, state)

    def _build_graph_data(self, state: SchedulingState) -> Optional[Data]:
        if not HAS_TORCH_GEOMETRIC: return None
        try:
            tasks = {t['id']: t for t in state.workflow_graph['tasks']}
            task_ids = list(tasks.keys())
            task_to_idx = {tid: i for i, tid in enumerate(task_ids)}

            node_features = []
            for tid in task_ids:
                task = tasks[tid]
                features = [
                    task.get('flops', 1e9) / 10e9,
                    task.get('memory', 4.0) / 16.0,
                    1.0 if tid == state.current_task else 0.0,
                    1.0 if tid in state.pending_tasks else 0.0,
                    len(task.get('dependencies', [])) / 10.0,
                    0.0, 0.0, 0.0
                ]
                node_features.append(features)

            edge_index = []
            for tid, task in tasks.items():
                for dep in task.get('dependencies', []):
                    if dep in task_to_idx:
                        edge_index.append([task_to_idx[dep], task_to_idx[tid]])

            x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            return Data(x=x, edge_index=edge_index_tensor)
        except Exception:
            return None

    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if self.gnn_encoder and "gnn_encoder" in checkpoint: self.gnn_encoder.load_state_dict(checkpoint["gnn_encoder"], strict=False)
            if "policy_network" in checkpoint: self.policy_network.load_state_dict(checkpoint["policy_network"], strict=False)
            print(f"Successfully loaded model for DRL from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")

class WASSRAGScheduler(BaseScheduler):
    """WASS-RAG调度器 - RAG增强的DRL方法"""

    def __init__(self, model_path: Optional[str] = None, knowledge_base_path: Optional[str] = None):
        super().__init__("WASS-RAG")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.heuristic_scheduler = WASSHeuristicScheduler()
        self.knowledge_base = RAGKnowledgeBase(knowledge_base_path)
        self.performance_predictor = PerformancePredictor(input_dim=96, hidden_dim=128).to(self.device)
        self._y_mean = 0.0
        self._y_std = 1.0
        if model_path and Path(model_path).exists():
            self._load_performance_predictor(model_path)
        else:
            print(f"Warning: No performance predictor model found at {model_path}, using random initialization.")

    def make_decision(self, state: SchedulingState) -> SchedulingAction:
        try:
            state_embedding = self._extract_simple_features_fallback(state)
            retrieved_context = self.knowledge_base.retrieve_similar_cases(state_embedding.cpu().numpy(), top_k=5)

            node_makespans = {}
            for node in state.available_nodes:
                action_embedding = self._encode_action(node, state)
                predicted_makespan = self._predict_performance(state_embedding, action_embedding, retrieved_context)
                node_makespans[node] = predicted_makespan

            if not node_makespans: raise ValueError("No nodes to schedule to.")
            
            unique_values = set(round(v, 3) for v in node_makespans.values())
            if len(unique_values) == 1:
                print(f"⚠️ [DEGRADATION] All predicted makespans are identical ({list(unique_values)[0]:.3f}). Applying heuristic tie-break.")
                return self.heuristic_scheduler.make_decision(state)

            best_node = min(node_makespans, key=node_makespans.get)
            best_makespan = node_makespans[best_node]
            makespan_values = list(node_makespans.values())
            confidence = 1.0 - (best_makespan / np.mean(makespan_values)) if len(makespan_values) > 1 and np.mean(makespan_values) > 0 else 0.8

            reasoning = self._generate_explanation(best_node, retrieved_context, node_makespans)
            return SchedulingAction(task_id=state.current_task, target_node=best_node, confidence=confidence, reasoning=reasoning)

        except Exception as e:
            import traceback
            print(f"⚠️  [DEGRADATION] RAG decision making failed: {e}\n{traceback.format_exc()}. Falling back to heuristic.")
            fallback_action = self.heuristic_scheduler.make_decision(state)
            fallback_action.reasoning = f"DEGRADED: RAG->Heuristic fallback due to error: {e}"
            return fallback_action

    def _encode_action(self, node: str, state: SchedulingState) -> torch.Tensor:
        """
        --- 核心修改：V4 时序感知特征编码器 ---
        该函数现在完全利用了离散事件仿真器提供的动态时序信息。
        """
        node_info = state.cluster_state.get("nodes", {}).get(node, {})
        task_info = self.heuristic_scheduler._get_task_info(state.workflow_graph, state.current_task)
        
        task_cpu_req = task_info.get("flops", 2e9) / 1e9
        task_mem_req = task_info.get("memory", 4.0)
        node_cpu_cap = node_info.get("cpu_capacity", 2.0)
        node_mem_cap = node_info.get("memory_capacity", 16.0)

        est = state.cluster_state.get('earliest_start_times', {}).get(node, state.timestamp)
        waiting_time = max(0, est - state.timestamp)
        
        initial_load = node_info.get("current_load", 0.5)
        available_cpu = node_cpu_cap * (1.0 - initial_load)
        
        est_exec_time = task_cpu_req / max(1e-6, available_cpu)
        est_finish_time = est + est_exec_time
        
        cpu_pressure = task_cpu_req / max(1e-6, node_cpu_cap)
        mem_pressure = task_mem_req / max(1e-6, node_mem_cap)
        
        all_est = state.cluster_state.get('earliest_start_times', {}).values()
        avg_cluster_finish_time = np.mean(list(all_est)) if all_est else state.timestamp
        finish_time_vs_avg = est_finish_time - avg_cluster_finish_time
        
        dependencies = task_info.get("dependencies", [])
        data_locality_score = ((hash(node) + hash(state.current_task) + len(dependencies)) % 100) / 100.0

        features = [
            node_cpu_cap / 10.0, node_mem_cap / 64.0,
            task_cpu_req / 20.0, task_mem_req / 16.0,
            cpu_pressure, mem_pressure,
            waiting_time / 10.0, est_exec_time / 100.0,
            est_finish_time / 200.0, finish_time_vs_avg / 50.0,
            data_locality_score, float(len(dependencies)),
        ]
        
        while len(features) < 32: features.append(0.0)
        return torch.tensor(features[:32], dtype=torch.float32, device=self.device)

    def _predict_performance(self, state_embedding: torch.Tensor,
                           action_embedding: torch.Tensor,
                           context: Dict[str, Any]) -> float:
        context_embedding = self._encode_context(context)
        
        combined_features = torch.cat([
            state_embedding.flatten(), action_embedding.flatten(), context_embedding.flatten()
        ])

        if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
            print("⚠️ [FEATURE] Invalid features detected, using fallback prediction")
            return 10.0

        with torch.no_grad():
            predicted_value_normalized = self.performance_predictor(combined_features).item()
            if hasattr(self, '_y_mean') and hasattr(self, '_y_std') and self._y_std > 1e-6:
                predicted_value = (predicted_value_normalized * self._y_std) + self._y_mean
            else:
                predicted_value = predicted_value_normalized * 10.0 + 20.0
            return max(0.1, predicted_value)

    def _encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        if not context or not context.get("similar_cases"):
            return torch.zeros(32, device=self.device)
        cases = context["similar_cases"]
        makespans = [c.get("makespan", 100.0) for c in cases]
        similarities = [c.get("similarity", 0.5) for c in cases]
        features = [np.mean(makespans)/100.0, np.min(makespans)/100.0, np.max(makespans)/100.0, len(cases) / 10.0, np.mean(similarities)]
        while len(features) < 32: features.append(0.0)
        return torch.tensor(features[:32], dtype=torch.float32, device=self.device)

    def _extract_simple_features_fallback(self, state: SchedulingState) -> torch.Tensor:
        tasks = state.workflow_graph.get('tasks', [])
        task_count = len(tasks)
        total_flops = sum(t.get('flops', 1e9) for t in tasks if isinstance(t, dict))
        avg_flops = total_flops / max(1, task_count)
        nodes = state.cluster_state.get('nodes', {})
        features = [task_count / 100.0, avg_flops / 10e9, len(state.pending_tasks) / max(1, task_count), len(nodes) / 20.0]
        while len(features) < 32: features.append(0.0)
        return torch.tensor(features[:32], dtype=torch.float32, device=self.device)

    def _generate_explanation(self, chosen_node: str, context: Dict[str, Any], node_makespans: Dict[str, float]) -> str:
        explanation = [f"RAG chose {chosen_node} (pred. makespan: {node_makespans[chosen_node]:.2f}s)"]
        if context and context.get("similar_cases"):
            explanation.append(f"based on {len(context['similar_cases'])} similar cases")
        top_3 = sorted(node_makespans.items(), key=lambda item: item[1])[:3]
        explanation.append(f"Top choices: " + ", ".join([f"{n}:{m:.2f}s" for n, m in top_3]))
        return "; ".join(explanation)

    def _load_performance_predictor(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if "performance_predictor" in checkpoint:
                self.performance_predictor.load_state_dict(checkpoint["performance_predictor"])
                print("Successfully loaded performance predictor")
                metadata = checkpoint.get("metadata", {}).get("performance_predictor", {})
                if isinstance(metadata, dict):
                    self._y_mean = metadata.get("y_mean", 0.0)
                    self._y_std = metadata.get("y_std", 1.0)
                    print(f"Loaded normalization params: mean={self._y_mean:.2f}, std={self._y_std:.2f}")
                else: self._y_mean, self._y_std = 0.0, 1.0
        except Exception as e:
            print(f"Failed to load performance predictor: {e}")
            self._y_mean, self._y_std = 0.0, 1.0

# --- Neural Network Components ---
class GraphEncoder(nn.Module):
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        if not HAS_TORCH_GEOMETRIC: raise ImportError("torch_geometric is required for GraphEncoder")
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, data.batch) if hasattr(data, 'batch') else x.mean(dim=0)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class PerformancePredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

class RAGKnowledgeBase:
    def __init__(self, knowledge_base_path: Optional[str] = None, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        if knowledge_base_path and Path(knowledge_base_path).exists():
            self.load_knowledge_base(knowledge_base_path)
        else: self._initialize_empty_kb()

    def _initialize_empty_kb(self):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.cases = []
        print("Initialized empty knowledge base")

    def retrieve_similar_cases(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        if not self.cases or self.index.ntotal == 0: return {"similar_cases": []}
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        return {"similar_cases": [{**self.cases[i], "similarity": 1 / (1 + d)} for i, d in zip(indices[0], distances[0]) if i < len(self.cases)]}

    def add_case(self, embedding: np.ndarray, workflow_info: Dict[str, Any], actions: List[str], makespan: float):
        embedding_np = embedding.reshape(1, -1).astype('float32')
        self.index.add(embedding_np)
        self.cases.append({"workflow_info": workflow_info, "actions": actions, "makespan": makespan})

    def load_knowledge_base(self, path: str):
        try:
            with open(path, 'rb') as f: data = pickle.load(f)
            self.index = data["index"]
            self.cases = data["cases"]
            print(f"Loaded knowledge base with {len(self.cases)} cases")
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")
            self._initialize_empty_kb()

    def save_knowledge_base(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({"index": self.index, "cases": self.cases}, f)
        print(f"Saved knowledge base with {len(self.cases)} cases to {path}")

def create_scheduler(method_name: str, **kwargs) -> BaseScheduler:
    if method_name == "WASS (Heuristic)": return WASSHeuristicScheduler()
    elif method_name == "WASS-DRL (w/o RAG)": return WASSSmartScheduler(kwargs.get("model_path"))
    elif method_name == "WASS-RAG":
        return WASSRAGScheduler(kwargs.get("model_path"), kwargs.get("knowledge_base_path"))
    else: raise ValueError(f"Unknown scheduler method: {method_name}")