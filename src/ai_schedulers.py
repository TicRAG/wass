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
        
        # 使用DRL agent选择动作，启用探索机制
        # 随着训练进行逐步降低epsilon值
        training_progress = simulation.current_step / max(simulation.total_steps, 1000)
        epsilon = max(0.1, 0.5 * (1 - training_progress))  # 从0.5递减到0.1
        
        action_idx = self.drl_agent.act(state, epsilon=epsilon)
        chosen_node_name = self.node_names[action_idx]
        
        logger.debug(f"DRL Agent chose action {action_idx} -> Node {chosen_node_name} for task {task_to_schedule.id} (epsilon={epsilon:.2f})")
        
        return {chosen_node_name: task_to_schedule}

    def _extract_features(self, task: w.Task, simulation: 'WassWrenchSimulator') -> SchedulingState:
        """Extracts enhanced features for better decision making."""
        num_nodes = len(self.node_names)
        
        # 1. Task features - 增强任务特征
        total_input_size = sum(simulation.workflow.get_edge_data_size(p.id, task.id) for p in task.parents)
        avg_parent_runtime = np.mean([p.computation_size for p in task.parents]) if task.parents else 0.0
        
        task_features = [
            task.computation_size,
            len(task.parents),
            len(task.children),
            total_input_size,
            avg_parent_runtime,
            task.computation_size / (total_input_size + 1e-6),  # 计算密度
        ]
        
        # 2. Node features - 增强节点特征
        node_features = []
        for node_name in self.node_names:
            node = simulation.platform.get_node(node_name)
            
            # 基础时间计算
            earliest_finish_time = simulation.node_earliest_finish_time.get(node_name, 0.0)
            computation_time = task.computation_size / node.speed
            
            # 数据传输时间
            total_d_time = 0.0
            max_d_time = 0.0
            parent_nodes = [simulation.task_to_node_map.get(parent) for parent in task.parents]
            
            for parent in task.parents:
                parent_node_id = simulation.task_to_node_map.get(parent)
                if parent_node_id and parent_node_id != node_name:
                    data_size = simulation.workflow.get_edge_data_size(parent.id, task.id)
                    bandwidth = simulation.platform.get_bandwidth(parent_node_id, node_name)
                    d_time = data_size / bandwidth
                    total_d_time += d_time
                    max_d_time = max(max_d_time, d_time)
            
            # 完成时间 = max(最早可用时间 + 数据传输时间) + 计算时间
            ready_time = max(earliest_finish_time, max_d_time)
            total_finish_time = ready_time + computation_time
            
            # 节点负载特征
            node_queue_length = len([t for t, n in simulation.task_to_node_map.items() if n == node_name])
            
            node_features.extend([
                node.speed,
                earliest_finish_time,
                total_d_time,
                max_d_time,
                total_finish_time,
                node_queue_length,
                computation_time,
                total_d_time / (computation_time + 1e-6),  # 通信计算比
            ])

        # 3. 全局特征 - 工作流上下文
        total_tasks = len(simulation.workflow.tasks)
        completed_tasks = len([t for t in simulation.workflow.tasks if t.get_state_as_string() == 'COMPLETED'])
        
        global_features = [
            completed_tasks / total_tasks,  # 完成进度
            len([t for t in simulation.workflow.tasks if t.get_state_as_string() == 'RUNNING']),  # 运行中任务
            np.mean([simulation.node_earliest_finish_time.get(n, 0.0) for n in self.node_names]),  # 平均节点负载
        ]
        
        # 标准化特征
        state_vector = np.concatenate([task_features, node_features, global_features]).astype(np.float32)
        
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
        teacher_makespan = estimated_makespans[best_node_from_teacher].value
        
        # Get the action from the DRL agent (the "student") - 不使用教师指导，让Agent独立决策
        state = self._extract_features(task_to_schedule, simulation)
        action_idx_from_student = self.drl_agent.act(state, use_teacher=False)
        chosen_node_name = self.node_names[action_idx_from_student]
        
        # 计算学生选择的预测makespan
        student_makespan = estimated_makespans[chosen_node_name].value
        
        # 实现R_RAG动态奖励机制
        rag_reward = teacher_makespan - student_makespan  # R_RAG = P_teacher - P_agent
        
        # 添加奖励规范化
        baseline = np.mean(list(estimated_makespans.values()))
        rag_reward = (rag_reward / (baseline + 1e-6)) * 10.0  # 标准化奖励范围
        
        # 带历史轨迹的学习
        self.drl_agent.store_transition(
            state=state,
            action=action_idx_from_student,
            reward=rag_reward,
            next_state=self._extract_features(task_to_schedule, simulation),
            done=len(simulation.workflow.tasks) == len(simulation.completed_tasks)
        )
        
        # 每100步批量学习
        if len(self.drl_agent.memory) % 100 == 0:
            self.drl_agent.replay()
        
        logger.debug(f"RAG: Teacher={best_node_from_teacher}(makespan={teacher_makespan:.2f}), "
                    f"Student={chosen_node_name}(makespan={student_makespan:.2f}), "
                    f"Reward={rag_reward:.2f}")
        
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