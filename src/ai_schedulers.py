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
            
        task_info = self._get_task_info(state.workflow_graph, task_id)
        data_locality_scores = self._calculate_data_locality_scores(
            task_info, available_nodes, state.cluster_state
        )
        resource_match_scores = self._calculate_resource_match_scores(
            task_info, available_nodes, state.cluster_state
        )
        load_balance_scores = self._calculate_load_balance_scores(
            available_nodes, state.cluster_state
        )
        
        final_scores = {}
        for node in available_nodes:
            final_scores[node] = (
                0.4 * data_locality_scores.get(node, 0) +
                0.3 * resource_match_scores.get(node, 0) +
                0.3 * load_balance_scores.get(node, 0)
            )
        
        best_node = max(final_scores.keys(), key=lambda k: final_scores[k])
        confidence = final_scores[best_node]
        
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
        tasks = workflow_graph.get("tasks", [])
        task_requirements = workflow_graph.get("task_requirements", {})
        
        for task in tasks:
            if isinstance(task, str):
                if task == task_id:
                    return task_requirements.get(task_id, {
                        "cpu": 2.0, "memory": 4.0, "duration": 5.0,
                        "dependencies": workflow_graph.get("dependencies", {}).get(task_id, [])
                    })
            elif isinstance(task, dict):
                if task.get("id") == task_id:
                    return task
                    
        return {
            "cpu": 2.0, "memory": 4.0, "duration": 5.0,
            "dependencies": workflow_graph.get("dependencies", {}).get(task_id, [])
        }
    
    def _calculate_data_locality_scores(self, task_info: Dict[str, Any], 
                                      available_nodes: List[str], 
                                      cluster_state: Dict[str, Any]) -> Dict[str, float]:
        scores = {}
        for node in available_nodes:
            score = 0.5
            dependencies = task_info.get("dependencies", [])
            if dependencies:
                score += 0.3 * len(dependencies) / max(len(dependencies), 1)
            scores[node] = min(score, 1.0)
        return scores
    
    def _calculate_resource_match_scores(self, task_info: Dict[str, Any],
                                       available_nodes: List[str],
                                       cluster_state: Dict[str, Any]) -> Dict[str, float]:
        scores = {}
        task_cpu_req = task_info.get("cpu", task_info.get("flops", 2.0))
        task_memory_req = task_info.get("memory", 4.0)
        
        for node in available_nodes:
            node_info = cluster_state.get("nodes", {}).get(node, {})
            node_cpu_capacity = node_info.get("cpu_capacity", 2.0)
            node_memory_capacity = node_info.get("memory_capacity", 16.0)
            
            cpu_utilization = float(task_cpu_req) / float(node_cpu_capacity)
            memory_utilization = float(task_memory_req) / float(node_memory_capacity)
            
            cpu_score = 1.0 - abs(cpu_utilization - 0.7)
            memory_score = 1.0 - abs(memory_utilization - 0.7)
            
            scores[node] = max(0.0, (cpu_score + memory_score) / 2.0)
        return scores
    
    def _calculate_load_balance_scores(self, available_nodes: List[str],
                                     cluster_state: Dict[str, Any]) -> Dict[str, float]:
        scores = {}
        for node in available_nodes:
            node_info = cluster_state.get("nodes", {}).get(node, {})
            current_load = node_info.get("current_load", 0.5)
            scores[node] = 1.0 - current_load
        return scores
class WASSSmartScheduler(BaseScheduler):
    """WASS-DRL智能调度器 (w/o RAG) - 标准DRL方法"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("WASS-DRL (w/o RAG)")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 新增: 为 DRL 失败时提供回退机制 ---
        self.heuristic_scheduler = WASSHeuristicScheduler()
        
        if HAS_TORCH_GEOMETRIC:
            self.gnn_encoder = GraphEncoder(
                node_feature_dim=8, edge_feature_dim=4,
                hidden_dim=64, output_dim=32
            ).to(self.device)
        else:
            self.gnn_encoder = None
            
        self.policy_network = PolicyNetwork(
            state_dim=32, action_dim=1, hidden_dim=128
        ).to(self.device)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print(f"Warning: No pretrained model found at {model_path}, using random initialization")
    
    def make_decision(self, state: SchedulingState) -> SchedulingAction:
        """[已修正] 基于DRL的调度决策 (WASS-DRL)"""
        try:
            # 1. 将当前状态编码为张量嵌入
            state_embedding = self._encode_state_graph(state)

            # 2. 评估每个可行的动作（将任务分配给节点）
            node_scores = {}
            with torch.no_grad():
                for node in state.available_nodes:
                    # 创建一个组合的状态-动作上下文表示
                    state_action_context = self._encode_node_context(state_embedding, node, state)
                    
                    # 使用策略网络为该动作打分
                    score = self.policy_network(state_action_context).item()
                    node_scores[node] = score

            # 3. 选择得分最高的动作（节点）
            if not node_scores:
                 raise ValueError("没有可用的节点进行评分。")
            
            best_node = max(node_scores, key=node_scores.get)
            confidence = torch.sigmoid(torch.tensor(node_scores[best_node])).item()

            reasoning = f"DRL决策：节点 {best_node} 的得分最高，为 {node_scores[best_node]:.3f}"

            return SchedulingAction(
                task_id=state.current_task,
                target_node=best_node,
                confidence=confidence,
                reasoning=reasoning
            )
        except Exception as e:
            print(f"  [降级] DRL决策失败：{e}。回退到启发式方法。")
            # 回退到简单的启发式调度器
            fallback_action = self.heuristic_scheduler.make_decision(state)
            fallback_action.reasoning = f"降级：由于错误，从DRL回退到启发式方法：{e}"
            return fallback_action

    def _encode_state_graph(self, state: SchedulingState) -> torch.Tensor:
        if self.gnn_encoder is None:
            return self._extract_simple_features(state)
        graph_data = self._build_graph_data(state)
        if graph_data is None:
            return self._extract_simple_features(state)
        with torch.no_grad():
            embedding = self.gnn_encoder(graph_data)
        return embedding
    
    # ... (保留 WASSSmartScheduler 类中所有其他以 '_' 开头的方法，例如 _extract_simple_features 等) ...
    def _extract_simple_features(self, state: SchedulingState) -> torch.Tensor:
        features = []
        total_tasks = len(state.workflow_graph.get("tasks", []))
        pending_tasks = len(state.pending_tasks)
        features.extend([total_tasks, pending_tasks, pending_tasks/max(total_tasks, 1)])
        
        total_nodes = len(state.available_nodes)
        avg_load = 0.5
        features.extend([total_nodes, avg_load])
        
        task_info = None
        tasks = state.workflow_graph.get("tasks", [])
        for task in tasks:
            if isinstance(task, str):
                if task == state.current_task:
                    task_reqs = state.workflow_graph.get("task_requirements", {})
                    task_info = task_reqs.get(task, {})
                    break
            elif isinstance(task, dict):
                if task.get("id") == state.current_task:
                    task_info = task
                    break
        
        if task_info:
            task_cpu = task_info.get("cpu", task_info.get("flops", 2.0))
            task_memory = task_info.get("memory", 4.0)
            task_duration = task_info.get("duration", task_info.get("runtime", 5.0))
            cpu_norm = min(1.0, float(task_cpu) / 15e9)
            mem_norm = min(1.0, float(task_memory) / 8.0)
            dur_norm = min(1.0, float(task_duration) / 180.0)
            features.extend([cpu_norm, mem_norm, dur_norm])
        else:
            features.extend([0.2, 0.25, 0.25])
        
        while len(features) < 32:
            features.append(0.0)
        return torch.tensor(features[:32], dtype=torch.float32, device=self.device)
    
    def _encode_node_context(self, state_embedding: torch.Tensor, node: str, state: SchedulingState) -> torch.Tensor:
        node_features = [hash(node) % 100 / 100.0, 0.5, 1.0]
        node_tensor = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        if state_embedding.dim() > 1:
            state_embedding = state_embedding.flatten()
        if len(state_embedding) > 29:
            state_embedding = state_embedding[:29]
        else:
            padding = torch.zeros(29 - len(state_embedding), device=self.device)
            state_embedding = torch.cat([state_embedding, padding])
        return torch.cat([state_embedding, node_tensor])
    
    def _build_graph_data(self, state: SchedulingState):
        if not HAS_TORCH_GEOMETRIC:
            return None
        try:
            tasks = state.workflow_graph.get("tasks", [])
            task_requirements = state.workflow_graph.get("task_requirements", {})
            dependencies = state.workflow_graph.get("dependencies", {})
            
            node_features = []
            for task in tasks:
                if isinstance(task, str):
                    task_id, task_info = task, task_requirements.get(task, {})
                    task_deps = dependencies.get(task_id, [])
                else:
                    task_id, task_info = task.get("id", str(task)), task
                    task_deps = task.get("dependencies", [])
                
                cpu_req = task_info.get("cpu", task_info.get("flops", 2.0))
                mem_req = task_info.get("memory", 4.0)
                duration = task_info.get("duration", task_info.get("runtime", 5.0))
                
                task_features = [
                    min(1.0, float(cpu_req) / 10.0), min(1.0, float(mem_req) / 16.0),
                    min(1.0, float(duration) / 20.0), 1.0 if task_id == state.current_task else 0.0,
                    1.0 if task_id in state.pending_tasks else 0.0,
                    min(1.0, len(task_deps) / 5.0), 0.0, 0.0
                ]
                node_features.append(task_features)
            
            edge_index, task_to_index = [], {}
            for i, task in enumerate(tasks):
                task_id = task if isinstance(task, str) else task.get("id", str(task))
                task_to_index[task_id] = i
            
            for i, task in enumerate(tasks):
                task_id = task if isinstance(task, str) else task.get("id", str(task))
                task_deps = dependencies.get(task_id, []) if isinstance(task, str) else task.get("dependencies", [])
                for dep in task_deps:
                    if dep in task_to_index:
                        edge_index.append([task_to_index[dep], i])
            
            x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long, device=self.device)
            return Data(x=x, edge_index=edge_index_tensor)
        except Exception as e:
            print(f"⚠️  [DEGRADATION] Graph data construction failed: {e}")
            return None
    
    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if self.gnn_encoder is not None:
                self.gnn_encoder.load_state_dict(checkpoint.get("gnn_encoder", {}), strict=False)
            self.policy_network.load_state_dict(checkpoint.get("policy_network", {}))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")


class WASSRAGScheduler(BaseScheduler):
    """WASS-RAG调度器 - RAG增强的DRL方法"""
    
    def __init__(self, model_path: Optional[str] = None, knowledge_base_path: Optional[str] = None):
        super().__init__("WASS-RAG")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.base_scheduler = WASSSmartScheduler(model_path)
        except Exception as e:
            print(f"Warning: Failed to load WASSSmartScheduler: {e}")
            self.base_scheduler = None
            
        self.heuristic_scheduler = WASSHeuristicScheduler()
        self._last_state_embedding: Optional[torch.Tensor] = None
        
        self.knowledge_base = RAGKnowledgeBase(knowledge_base_path)
        self.performance_predictor = PerformancePredictor(
            input_dim=96, hidden_dim=128
        ).to(self.device)

        if model_path:
            self._load_performance_predictor(model_path)
    
    def make_decision(self, state: SchedulingState) -> SchedulingAction:
        try:
            if self.base_scheduler and self.base_scheduler.gnn_encoder is not None:
                state_embedding = self.base_scheduler._encode_state_graph(state)
            else:
                state_embedding = self._extract_simple_features_fallback(state) if self.base_scheduler is None else self.base_scheduler._extract_simple_features(state)
            self._last_state_embedding = state_embedding.detach().clone() if torch.is_tensor(state_embedding) else None
            
            retrieved_context = self.knowledge_base.retrieve_similar_cases(
                state_embedding.cpu().numpy(), top_k=5
            )
            
            node_makespans, node_scores = {}, {}
            for node in state.available_nodes:
                action_embedding = self._encode_action(node, state)
                predicted_makespan = self._predict_performance(
                    state_embedding, action_embedding, retrieved_context
                )
                node_makespans[node] = predicted_makespan
                node_scores[node] = 1.0 / max(predicted_makespan, 0.01)
            
            score_values = list(node_scores.values())
            if len(set(score_values)) == 1:
                print(f"⚠️ [DEGRADATION] All node scores identical ({score_values[0]:.3f}). Applying heuristic tie-break.")
                heuristic_action = self.heuristic_scheduler.make_decision(state)
                best_node = heuristic_action.target_node
                confidence = min(0.55, 0.35 + heuristic_action.confidence * 0.3)
            else:
                best_node = max(node_scores.keys(), key=lambda k: node_scores[k])
                score_range = max(score_values) - min(score_values)
                confidence = 0.5 + min(0.4, score_range * 2)
            
            reasoning = self._generate_explanation(best_node, retrieved_context, node_scores, node_makespans)
            
            return SchedulingAction(
                task_id=state.current_task,
                target_node=best_node,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"⚠️  [DEGRADATION] RAG decision making failed: {e}")
            fallback_action = self.base_scheduler.make_decision(state)
            fallback_action.reasoning = f" DEGRADED: RAG->DRL fallback due to error: {e}"
            return fallback_action
    
    def _encode_action(self, node: str, state: SchedulingState) -> torch.Tensor:
        node_info = state.cluster_state.get("nodes", {}).get(node, {})
        task_info = self.heuristic_scheduler._get_task_info(state.workflow_graph, state.current_task)
        
        task_cpu_req = float(task_info.get("flops", 2e9)) / 1e9
        task_mem_req = float(task_info.get("memory", 4.0))
        
        node_cpu_cap = float(node_info.get("cpu_capacity", 2.0))
        node_mem_cap = float(node_info.get("memory_capacity", 16.0))
        current_load = float(node_info.get("current_load", 0.5))

        available_cpu = node_cpu_cap * (1.0 - current_load)
        available_mem = node_mem_cap * (1.0 - current_load * 0.5)
        
        cpu_fit = (available_cpu - task_cpu_req) / max(node_cpu_cap, 1.0) if task_cpu_req <= available_cpu else (available_cpu - task_cpu_req) / max(task_cpu_req, 1.0)
        mem_fit = (available_mem - task_mem_req) / max(node_mem_cap, 1.0) if task_mem_req <= available_mem else (available_mem - task_mem_req) / max(task_mem_req, 1.0)
        
        base_exec_time_node = task_cpu_req / max(available_cpu, 0.1)
        performance_match = max(0.0, 1.0 - base_exec_time_node / 10.0)

        dependencies = task_info.get("dependencies", [])
        data_locality_score = ((hash(node) + hash(state.current_task) + len(dependencies)) % 100) / 100.0 if dependencies else 0.5
        
        all_loads = [n.get("current_load", 0.5) for n in state.cluster_state.get("nodes", {}).values()]
        avg_load = sum(all_loads) / max(len(all_loads), 1)
        load_balance_score = 1.0 - abs(current_load - avg_load)

        try:
            node_id_numeric = float(''.join(filter(str.isdigit, node))) / 20.0
        except (ValueError, ZeroDivisionError):
            node_id_numeric = hash(node) % 100 / 100.0

        action_features = [
            cpu_fit, mem_fit, performance_match, data_locality_score, load_balance_score,
            node_id_numeric, current_load, 1.0 - current_load,
            node_cpu_cap / 5.0, node_mem_cap / 64.0,
            task_cpu_req / 5.0, task_mem_req / 16.0,
            task_cpu_req / max(task_mem_req, 1.0) / 2.0,
            len(state.available_nodes) / 20.0,
        ]
        
        while len(action_features) < 32:
            action_features.append(0.0)
        return torch.tensor(action_features[:32], dtype=torch.float32, device=self.device)
    
    def _predict_performance(self, state_embedding: torch.Tensor,
                           action_embedding: torch.Tensor,
                           context: Dict[str, Any]) -> float:
        """[FINAL FIX] Predicts makespan with a robust, simplified logic."""
        
        context_embedding = self._encode_context(context)
        
        def pad_to_32(tensor):
            if len(tensor) < 32:
                padding = torch.zeros(32 - len(tensor), device=tensor.device)
                return torch.cat([tensor, padding])
            return tensor[:32]

        combined_features = torch.cat([
            pad_to_32(state_embedding.flatten()),
            pad_to_32(action_embedding.flatten()),
            pad_to_32(context_embedding.flatten())
        ])

        # Handle any potential lingering NaN/inf values gracefully
        if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
            print("⚠️ [FEATURE] Invalid features detected, using fallback prediction")
            return 10.0 # Return a neutral, stable fallback value

        with torch.no_grad():
            # The model now outputs a non-negative, normalized value directly
            predicted_makespan_normalized = self.performance_predictor(combined_features).item()
            
            # Use saved training stats to de-normalize the prediction
            if hasattr(self, '_y_mean') and hasattr(self, '_y_std') and self._y_std > 1e-6:
                predicted_makespan = (predicted_makespan_normalized * self._y_std) + self._y_mean
            else:
                # Fallback if normalization stats are missing
                predicted_makespan = predicted_makespan_normalized * 5.0 + 5.0

            # --- FINAL FIX: Add a stable minimum floor ---
            # Ensure the final prediction is always a sensible, non-zero value.
            return max(0.1, predicted_makespan)
    
    def _encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        if not context or "similar_cases" not in context or not context["similar_cases"]:
            return torch.zeros(32, device=self.device)
        
        similar_cases = context["similar_cases"]
        makespans = [case.get("makespan", 100.0) for case in similar_cases]
        avg_makespan, min_makespan, max_makespan = np.mean(makespans), np.min(makespans), np.max(makespans)
        
        similarities = [case.get("similarity", 0.5) for case in similar_cases]
        max_sim, min_sim = max(similarities), min(similarities)
        avg_similarity = ((np.mean(similarities) - min_sim) / (max_sim - min_sim)) if max_sim > min_sim else 0.5
        
        features = [
            avg_makespan/100.0, min_makespan/100.0, max_makespan/100.0,
            len(similar_cases) / 10.0, min(1.0, max(0.0, avg_similarity))
        ]
        
        while len(features) < 32:
            features.append(0.0)
        return torch.tensor(features[:32], dtype=torch.float32, device=self.device)
    
    def _extract_simple_features_fallback(self, state: SchedulingState) -> torch.Tensor:
        tasks = state.workflow_graph.get('tasks', [])
        task_count = len(tasks)
        total_flops = sum(task.get('flops', 1e9) for task in tasks if isinstance(task, dict))
        avg_flops = total_flops / max(task_count, 1)
        total_memory = sum(task.get('memory', 1.0) for task in tasks if isinstance(task, dict))
        avg_memory = total_memory / max(task_count, 1)
        
        features = [
            task_count / 100.0, avg_flops / 5e9, avg_memory / 8.0,
            len(state.pending_tasks) / max(task_count, 1),
        ]
        
        nodes = state.cluster_state.get('nodes', {})
        if nodes:
            cpu_caps = [node.get('cpu_capacity', 2.0) for node in nodes.values()]
            mem_caps = [node.get('memory_capacity', 16.0) for node in nodes.values()]
            loads = [node.get('current_load', 0.5) for node in nodes.values()]
            features.extend([
                len(nodes) / 20.0, np.mean(cpu_caps) / 5.0,
                np.mean(mem_caps) / 64.0, np.mean(loads),
            ])
        else:
            features.extend([0.1, 0.4, 0.25, 0.5])
        
        while len(features) < 32:
            features.append(0.0)
        return torch.tensor(features[:32], dtype=torch.float32, device=self.device)
    
    def _generate_explanation(self, chosen_node: str, context: Dict[str, Any], 
                            node_scores: Dict[str, float], node_makespans: Dict[str, float]) -> str:
        explanation_parts = [f"RAG-enhanced decision: chose node {chosen_node}"]
        explanation_parts.append(f"predicted makespan: {node_makespans[chosen_node]:.2f}s")
        
        if context and "similar_cases" in context and context["similar_cases"]:
            similar_cases = context["similar_cases"]
            avg_hist_makespan = np.mean([case.get("makespan", 100.0) for case in similar_cases])
            explanation_parts.append(f"based on {len(similar_cases)} similar historical cases")
            explanation_parts.append(f"historical avg makespan: {avg_hist_makespan:.2f}s")
        
        sorted_by_makespan = sorted(node_makespans.items(), key=lambda x: x[1])
        top_3 = sorted_by_makespan[:3]
        makespan_str = ", ".join([f"{node}:{makespan:.2f}s" for node, makespan in top_3])
        explanation_parts.append(f"top choices: {makespan_str}")
        return "; ".join(explanation_parts)
    
    def _load_performance_predictor(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if "performance_predictor" in checkpoint:
                self.performance_predictor.load_state_dict(checkpoint["performance_predictor"])
                print("Successfully loaded performance predictor")
                
                if "metadata" in checkpoint and "performance_predictor" in checkpoint["metadata"]:
                    metadata = checkpoint["metadata"]["performance_predictor"]
                    if isinstance(metadata, dict):
                        self._y_mean = metadata.get("y_mean", 0.0)
                        self._y_std = metadata.get("y_std", 1.0)
                        print(f"Loaded normalization params: mean={self._y_mean:.2f}, std={self._y_std:.2f}")
                    else: self._y_mean, self._y_std = 0.0, 1.0
                else: self._y_mean, self._y_std = 0.0, 1.0
        except Exception as e:
            print(f"Failed to load performance predictor: {e}")
            self._y_mean, self._y_std = 0.0, 1.0

# --- Neural Network Components ---
class GraphEncoder(nn.Module):
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, 
                 hidden_dim: int, output_dim: int):
        super().__init__()
        if not HAS_TORCH_GEOMETRIC: raise ImportError("torch_geometric is required for GraphEncoder")
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, graph_data: Data) -> torch.Tensor:
        x, edge_index = graph_data.x, graph_data.edge_index
        x = F.relu(self.node_embedding(x))
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        batch = getattr(graph_data, 'batch', None)
        x = global_mean_pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
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
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # --- FINAL FIX: Ensure output is always non-negative ---
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

class RAGKnowledgeBase:
    def __init__(self, knowledge_base_path: Optional[str] = None, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        if knowledge_base_path and Path(knowledge_base_path).exists():
            self.load_knowledge_base(knowledge_base_path)
        else:
            self._initialize_empty_kb()
    
    def _initialize_empty_kb(self):
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.cases = []
        print("Initialized empty knowledge base")
    
    def retrieve_similar_cases(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        if self.index is None or self.index.ntotal == 0:
            return {"similar_cases": []}
        try:
            query_vector = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
            similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            similar_cases = []
            for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.cases):
                    case = self.cases[idx].copy()
                    case["similarity"], case["rank"] = float(sim), i + 1
                    similar_cases.append(case)
            return {"similar_cases": similar_cases}
        except Exception as e:
            print(f"Error in knowledge base retrieval: {e}")
            return {"similar_cases": []}

    def add_case(self, embedding: np.ndarray, workflow_info: Dict[str, Any],
                actions: List[str], makespan: float):
        case = {"workflow_info": workflow_info, "actions": actions, "makespan": makespan}
        self.cases.append(case)
        if self.index is None: self._initialize_empty_kb()
        try:
            embedding_np = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            self.index.add(np.copy(embedding_np))
        except Exception as e:
            print(f"Error adding case to KB: {e}")
            self.cases.pop()

    def load_knowledge_base(self, path: str):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.index, self.cases = data.get("index"), data.get("cases", [])
            print(f"Loaded knowledge base with {len(self.cases)} cases")
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")
            self._initialize_empty_kb()
    
    def save_knowledge_base(self, path: str):
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({"index": self.index, "cases": self.cases}, f)
            print(f"Saved knowledge base with {len(self.cases)} cases to {path}")
        except Exception as e:
            print(f"Failed to save knowledge base: {e}")

def create_scheduler(method_name: str, **kwargs) -> BaseScheduler:
    if method_name == "WASS (Heuristic)":
        return WASSHeuristicScheduler()
    elif method_name == "WASS-DRL (w/o RAG)":
        return WASSSmartScheduler(kwargs.get("model_path"))
    elif method_name == "WASS-RAG":
        return WASSRAGScheduler(
            kwargs.get("model_path"), kwargs.get("knowledge_base_path")
        )
    else:
        raise ValueError(f"Unknown scheduler method: {method_name}")