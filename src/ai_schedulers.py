import logging
import random
from typing import Dict, List

import numpy as np
import torch
import networkx as nx
import wrench.task as w
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
        """提取图结构特征，构建GNN输入"""
        
        # 构建当前工作流的DAG图
        dag = self._build_dag_graph(simulation.workflow)
        
        # 更新当前任务的状态
        if task.id in dag.nodes:
            dag.nodes[task.id]['is_submitted'] = True
        
        # 构建节点特征（包含当前任务和平台状态）
        node_features = {}
        for node_name in self.node_names:
            platform_node = simulation.platform.get_node(node_name)
            node_features[node_name] = {
                'speed': platform_node.speed,
                'available_time': simulation.node_earliest_finish_time.get(node_name, 0.0),
                'queue_length': len([t for t, n in simulation.task_to_node_map.items() if n == node_name])
            }
        
        # 使用GNN提取图特征
        graph_data = self.predictor.extract_graph_features(dag, node_features, focus_task_id=task.id)
        
        # 将图数据转换为特征向量（简化处理）
        # 这里我们使用图的全局特征作为状态表示
        node_embeddings = graph_data.x.numpy()
        
        # 计算图的全局统计特征
        graph_features = [
            np.mean(node_embeddings[:, 0]),  # 平均计算量
            np.std(node_embeddings[:, 0]),   # 计算量方差
            np.mean(node_embeddings[:, 5]),  # 平均节点速度
            np.mean(node_embeddings[:, 6]),  # 平均可用时间
            np.mean(node_embeddings[:, 7]),  # 平均队列长度
            np.mean(node_embeddings[:, 8]),  # 关键路径任务比例
            len(simulation.workflow.tasks),  # 总任务数
            len(simulation.completed_tasks) / len(simulation.workflow.tasks),  # 完成率
        ]
        
        # 任务特定特征
        task_specific_features = [
            task.computation_size,
            len(task.parents),
            len(task.children),
            sum(simulation.workflow.get_edge_data_size(p.id, task.id) for p in task.parents),
            sum(simulation.workflow.get_edge_data_size(task.id, c.id) for c in task.children),
        ]
        
        # 平台状态特征
        platform_features = []
        for node_name in self.node_names:
            platform_node = simulation.platform.get_node(node_name)
            current_tasks = [t for t, n in simulation.task_to_node_map.items() if n == node_name]
            total_computation = sum(t.computation_size for t in current_tasks)
            
            platform_features.extend([
                platform_node.speed,
                len(current_tasks),
                simulation.node_earliest_finish_time.get(node_name, 0.0),
            ])
        
        # 合并所有特征
        state_vector = np.concatenate([
            graph_features,
            task_specific_features,
            platform_features
        ]).astype(np.float32)
        
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

    def _build_dag_graph(self, workflow):
        """Convert workflow DAG to networkx DiGraph with task metadata"""
        dag = nx.DiGraph()
        for task in workflow.tasks:
            # 获取任务状态信息
            is_submitted = task.get_state_as_string() in ['SUBMITTED', 'RUNNING', 'COMPLETED']
            is_finished = task.get_state_as_string() == 'COMPLETED'
            assigned_host_id = -1  # 默认未分配
            
            dag.add_node(task.id, 
                computation_size=task.computation_size,
                parents=[p.id for p in task.parents],
                children=[c.id for c in task.children],
                is_submitted=is_submitted,
                is_finished=is_finished,
                assigned_host_id=assigned_host_id
            )
        for task in workflow.tasks:
            for child in task.children:
                data_size = workflow.get_edge_data_size(task.id, child.id)
                dag.add_edge(task.id, child.id, data_size=data_size)
        return dag

    def _get_estimated_makespans(self, task: w.Task, simulation: 'WassWrenchSimulator') -> Dict[str, PredictedValue]:
        """Estimate makespan for each possible node assignment using the GNN predictor.
        
        Returns:
            Dict mapping node names to predicted makespan values.
        """
        # 构建当前工作流的DAG图
        dag = self._build_dag_graph(simulation.workflow)
        
        # 更新当前任务的状态信息
        if task.id in dag.nodes:
            dag.nodes[task.id]['is_submitted'] = True
            
        # 构建节点特征
        node_features = {}
        for node_name in self.node_names:
            node = simulation.platform.get_node(node_name)
            node_features[node_name] = {
                'speed': node.speed,
                'available_time': simulation.node_earliest_finish_time.get(node_name, 0.0),
                'queue_length': len([t for t, n in simulation.task_to_node_map.items() if n == node_name])
            }
        
        # 为每个可能的节点分配计算预测的makespan
        estimated_makespans = {}
        for node_name in self.node_names:
            # 创建临时的节点分配
            temp_task_to_node_map = simulation.task_to_node_map.copy()
            temp_task_to_node_map[task] = node_name
            
            # 更新DAG中任务的分配信息
            temp_dag = dag.copy()
            if task.id in temp_dag.nodes:
                temp_dag.nodes[task.id]['assigned_host_id'] = self.node_names.index(node_name)
            
            # 提取图特征
            graph_data = self.predictor.extract_graph_features(temp_dag, node_features, focus_task_id=task.id)
            
            # 使用GNN预测器进行预测
            predicted_makespan = self.predictor.predict(graph_data)
            
            estimated_makespans[node_name] = PredictedValue(
                value=predicted_makespan,
                confidence=1.0  # 简化处理，实际应用中可以计算置信度
            )
        
        return estimated_makespans


class WASSRAGScheduler(WASSDRLScheduler):
    """WASS-RAG Scheduler with true R_RAG dynamic reward mechanism."""

    def __init__(self, drl_agent: DQNAgent, node_names: List[str], predictor: PerformancePredictor):
        super().__init__(drl_agent, node_names)
        self.predictor = predictor
        self.scaler = StandardScaler()
        self.vectorizer = DictVectorizer()
        self.reward_alpha = 0.8  # 奖励系数
        logger.info("Initialized WASSRAGScheduler with R_RAG dynamic reward mechanism.")

    def schedule(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> SchedulingDecision:
        if not ready_tasks:
            return {}
        
        task_to_schedule = ready_tasks[0]
        
        # 构建当前状态
        state = self._extract_features(task_to_schedule, simulation)
        
        # 让DRL Agent完全自主决策 - 彻底移除模仿学习
        # 使用动态epsilon值，随着训练进行逐步降低探索率
        training_progress = min(simulation.current_step / max(simulation.total_steps, 1000), 1.0)
        epsilon = max(0.05, 0.3 * (1 - training_progress))  # 从0.3递减到0.05
        
        action_idx_from_student = self.drl_agent.act(state, epsilon=epsilon)
        chosen_node_name = self.node_names[action_idx_from_student]
        
        # 获取"老师"（性能预测器）的建议 - 用于奖励计算，不用于决策
        estimated_makespans = self._get_estimated_makespans(task_to_schedule, simulation)
        best_node_from_teacher = min(estimated_makespans, key=lambda n: estimated_makespans[n].value)
        teacher_makespan = estimated_makespans[best_node_from_teacher].value
        student_makespan = estimated_makespans[chosen_node_name].value
        
        # 实现真正的R_RAG动态差分奖励机制
        # 奖励 = 老师最优选择的预测makespan - Agent自己探索的预测makespan
        # 这个差值明确告诉Agent它的探索方向是对是错
        raw_reward = teacher_makespan - student_makespan
        
        # 智能归一化：根据任务规模动态调整奖励范围
        task_scale = max(task_to_schedule.computation_size, 1.0)
        normalized_reward = raw_reward / task_scale
        
        # 使用sigmoid函数稳定奖励信号，避免极端值
        stable_reward = 2.0 / (1.0 + np.exp(-normalized_reward * 5.0)) - 1.0
        
        # 动态奖励缩放：根据训练进度调整奖励强度
        reward_scaling = 1.0 + training_progress * 2.0  # 训练后期加强学习信号
        scaled_reward = stable_reward * reward_scaling
        
        # 获取下一个状态用于学习
        next_state = self._extract_features(task_to_schedule, simulation)
        
        # 计算任务完成率作为辅助奖励信号
        completion_ratio = len(simulation.completed_tasks) / len(simulation.workflow.tasks)
        completion_bonus = 0.05 * completion_ratio if completion_ratio > 0.8 else 0.0
        
        # 紧急任务奖励：接近deadline时给予额外激励
        deadline_urgency = min(simulation.current_step / max(simulation.total_steps, 100), 1.0)
        urgency_bonus = 0.02 * deadline_urgency * completion_ratio
        
        # 探索奖励：鼓励探索新的调度策略
        exploration_bonus = 0.01 * np.random.random() if epsilon > 0.1 else 0.0
        
        # 最终奖励 = 主奖励 + 各种辅助奖励
        final_reward = scaled_reward + completion_bonus + urgency_bonus + exploration_bonus
        
        # 立即反馈给Agent进行强化学习
        self.drl_agent.store_transition(
            state=state,
            action=action_idx_from_student,
            reward=final_reward,
            next_state=next_state,
            done=len(simulation.workflow.tasks) == len(simulation.completed_tasks)
        )
        
        # 自适应学习频率：根据训练进度调整学习频率
        learn_frequency = max(5, int(20 * (1 - training_progress)))  # 从20递减到5
        if len(simulation.completed_tasks) % learn_frequency == 0:
            self.drl_agent.replay(batch_size=min(64, 16 + int(training_progress * 48)))
        
        # 记录详细的奖励信息用于调试和分析
        if len(simulation.completed_tasks) % 10 == 0:
            logger.info(f"R_RAG Decision: Task={task_to_schedule.id}, "
                       f"Teacher={best_node_from_teacher}(makespan={teacher_makespan:.2f}), "
                       f"Student={chosen_node_name}(makespan={student_makespan:.2f}), "
                       f"RawReward={raw_reward:.3f}, Scaled={scaled_reward:.3f}, "
                       f"FinalReward={final_reward:.3f}, Epsilon={epsilon:.2f}, "
                       f"Completion={completion_ratio:.2f}")
        
        return {chosen_node_name: task_to_schedule}

    def fit_scaler_vectorizer(self, features_list: List[Dict]):
        """Fits the scaler and vectorizer on a list of feature dictionaries."""
        if not features_list:
            logger.warning("Feature list is empty. Cannot fit scaler/vectorizer.")
            return
        self.scaler.fit(self.vectorizer.fit_transform(features_list))
        logger.info("Scaler and Vectorizer for RAG have been fitted.")


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