import logging
import random
from typing import Dict, List

import numpy as np
import torch
import wrench.workflows as w
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from src.drl_agent import DQNAgent, SchedulingState
from src.encoding_constants import STATE_DIM, ACTION_DIM, CONTEXT_DIM
from src.interfaces import PredictedValue, Scheduler, SchedulingDecision
from src.performance_predictor import PerformancePredictor
from src.utils import get_logger

logger = get_logger(__name__, logging.INFO)


class WASSScheduler(Scheduler):
    """Abstract base class for WASS Schedulers."""

    def schedule(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> SchedulingDecision:
        raise NotImplementedError


class WASSDRLScheduler(WASSScheduler):
    """WASS-DRL Scheduler."""

    def __init__(self, drl_agent: DQNAgent, node_names: List[str]):
        self.drl_agent = drl_agent
        self.node_names = node_names
        self.node_map = {name: i for i, name in enumerate(node_names)}
        logger.info(f"Initialized WASSDRLScheduler with {len(node_names)} nodes.")

    def schedule(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> SchedulingDecision:
        if not ready_tasks:
            return {}

        task_to_schedule = ready_tasks[0]  # In the new sim, we schedule one by one
        state = self._extract_features(task_to_schedule, simulation)
        
        # Use the DRL agent to choose the best action (node)
        action_idx = self.drl_agent.act(state)
        chosen_node_name = self.node_names[action_idx]
        
        logger.debug(f"DRL Agent chose action {action_idx} -> Node {chosen_node_name} for task {task_to_schedule.id}")
        
        return {chosen_node_name: task_to_schedule}

    def _extract_features(self, task: w.Task, simulation: 'WassWrenchSimulator') -> SchedulingState:
        """Extracts features for a single task scheduling decision."""
        num_nodes = len(self.node_names)
        
        # 1. Task features
        task_features = [
            task.computation_size,
            len(task.parents),
            len(task.children),
            # Add more task-specific features if needed
        ]
        
        # 2. Node features (dynamic)
        node_features = []
        for node_name in self.node_names:
            node = simulation.platform.get_node(node_name)
            
            # Earliest time the node will be free
            earliest_finish_time = simulation.node_earliest_finish_time.get(node_name, 0.0)
            
            # Estimated data transfer time from all parents to this node
            total_d_time = 0.0
            for parent in task.parents:
                parent_node_id = simulation.task_to_node_map.get(parent)
                if parent_node_id and parent_node_id != node_name:
                    data_size = simulation.workflow.get_edge_data_size(parent.id, task.id)
                    # total_d_time += simulation.platform.get_communication_time(data_size, parent_node_id, node_name)
                    bandwidth = simulation.platform.get_bandwidth(parent_node_id, node_name)
                    total_d_time += data_size / bandwidth


            node_features.extend([
                node.speed,
                earliest_finish_time,
                total_d_time,
            ])

        # Combine and flatten
        state_vector = np.concatenate([task_features, node_features]).astype(np.float32)
        
        return SchedulingState(state_vector)

    # ---- Embedding Helpers (Placeholders) ----
    def _encode_state(self, simulation: 'WassWrenchSimulator'):
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        # simple aggregate placeholders
        vec[0] = len(simulation.workflow.tasks) if hasattr(simulation.workflow, 'tasks') else 0
        return torch.from_numpy(vec)

    def _encode_action(self, node_name: str, state) -> torch.Tensor:
        vec = np.zeros(ACTION_DIM, dtype=np.float32)
        idx = self.node_map.get(node_name, 0) % ACTION_DIM
        vec[idx] = 1.0
        return torch.from_numpy(vec)

    def _encode_context(self, retrieved_cases) -> torch.Tensor:
        # Average outcome placeholder
        vec = np.zeros(CONTEXT_DIM, dtype=np.float32)
        if retrieved_cases:
            avg = np.mean([c.get('outcome', 0.0) for c in retrieved_cases])
            vec[0] = avg
        return torch.from_numpy(vec)


class WASSRAGScheduler(WASSDRLScheduler):
    """WASS-RAG Scheduler."""

    def __init__(self, drl_agent: DQNAgent, node_names: List[str], predictor: PerformancePredictor):
        super().__init__(drl_agent, node_names)
        self.predictor = predictor
        self.scaler = StandardScaler()
        self.vectorizer = DictVectorizer()
        logger.info("Initialized WASSRAGScheduler.")

    def _get_estimated_makespans(self, task: w.Task, simulation: 'WassWrenchSimulator') -> Dict[str, PredictedValue]:
        """Gets estimated makespans for scheduling a task on each node."""
        predictions = {}
        for node_name in self.node_names:
            # Create a hypothetical future state for the predictor
            # This logic needs to be carefully designed. For simplicity, we use the same features as DRL
            # but in a production system, this could be much more complex.
            features = self._extract_rag_features(task, node_name, simulation)
            
            # Vectorize and scale features
            vectorized_features = self.vectorizer.transform([features])
            scaled_features = self.scaler.transform(vectorized_features)
            
            # Predict
            predicted_makespan = self.predictor.predict(scaled_features)[0]
            predictions[node_name] = PredictedValue(predicted_makespan)
            
        return predictions

    def schedule(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> SchedulingDecision:
        if not ready_tasks:
            return {}
        
        task_to_schedule = ready_tasks[0]
        
        # Get predictions from the "teacher" (performance predictor)
        estimated_makespans = self._get_estimated_makespans(task_to_schedule, simulation)
        
        # Find the best node according to the teacher
        best_node_from_teacher = min(estimated_makespans, key=lambda n: estimated_makespans[n].value)
        
        # Get the action from the DRL agent (the "student")
        state = self._extract_features(task_to_schedule, simulation)
        action_idx_from_student = self.drl_agent.act(state, use_teacher=True, teacher_action=self.node_map[best_node_from_teacher])
        
        chosen_node_name = self.node_names[action_idx_from_student]
        
        logger.debug(f"RAG Teacher chose {best_node_from_teacher}, Student chose {chosen_node_name} for {task_to_schedule.id}")
        
        return {chosen_node_name: task_to_schedule}

    def fit_scaler_vectorizer(self, features_list: List[Dict]):
        """Fits the scaler and vectorizer on a list of feature dictionaries."""
        if not features_list:
            logger.warning("Feature list is empty. Cannot fit scaler/vectorizer.")
            return
        self.scaler.fit(self.vectorizer.fit_transform(features_list))
        logger.info("Scaler and Vectorizer for RAG have been fitted.")

    def _extract_rag_features(self, task: w.Task, node_name: str, simulation: 'WassWrenchSimulator') -> Dict:
        """Extracts features for the RAG performance predictor."""
        # This should be consistent with the features used to train the predictor
        features = {
            "task_computation_size": task.computation_size,
            "num_parents": len(task.parents),
            "num_children": len(task.children),
        }
        
        node = simulation.platform.get_node(node_name)
        features.update({
            f"node_speed": node.speed,
            f"node_available_time": simulation.node_earliest_finish_time.get(node_name, 0.0),
        })

        # Data transfer features
        total_input_data_size = sum(simulation.workflow.get_edge_data_size(p.id, task.id) for p in task.parents)
        features["total_input_data_size"] = total_input_data_size

        return features


# -------- Factory API --------
def create_scheduler(name: str, node_names: List[str] = None, drl_agent: DQNAgent = None, predictor: PerformancePredictor = None, model_path: str = None, knowledge_base_path: str = None):
    """创建调度器的工厂函数"""
    if name == 'WASS-DRL (w/o RAG)':
        if drl_agent is None:
            # 如果没有提供drl_agent，创建一个简单的随机调度器
            from .simple_schedulers import SimpleRandomScheduler
            return SimpleRandomScheduler(name)
        return WASSDRLScheduler(drl_agent, node_names)
    elif name == 'WASS-RAG':
        if drl_agent is None or predictor is None:
            # 如果没有提供完整的组件，创建一个改进的启发式调度器
            from .simple_schedulers import ImprovedHeuristicScheduler  
            return ImprovedHeuristicScheduler(name)
        return WASSRAGScheduler(drl_agent, node_names, predictor)
    elif name == 'WASS (Heuristic)':
        from .simple_schedulers import HeuristicScheduler
        return HeuristicScheduler(name)
    else:
        raise ValueError(f'Unknown scheduler name: {name}')