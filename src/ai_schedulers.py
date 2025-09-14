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
from src.reward_fix import RewardFix
from src.rag_fusion_fix import RAGFusionFix
from src.training_fix import TrainingFix

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
    """修复后的WASS-RAG调度器，整合了改进的奖励函数、RAG融合机制和训练策略"""

    def __init__(self, drl_agent: DQNAgent, node_names: List[str], predictor: PerformancePredictor):
        super().__init__(drl_agent, node_names)
        self.predictor = predictor
        self.scaler = StandardScaler()
        self.vectorizer = DictVectorizer()
        self.reward_alpha = 0.8  # 奖励系数
        
        # 初始化修复组件
        self.reward_fix = RewardFix()
        self.rag_fusion_fix = RAGFusionFix()
        self.training_fix = TrainingFix(
            initial_epsilon=1.0,
            min_epsilon=0.01,
            total_episodes=2000
        )
        
        # 训练统计
        self.episode_count = 0
        self.total_reward = 0.0
        self.makespan_history = []
        self.performance_history = []
        
        # 调试信息
        self.debug_info = {
            'reward_history': [],
            'fusion_history': [],
            'epsilon_history': []
        }
        
        logger.info("Initialized WASSRAGScheduler with R_RAG dynamic reward mechanism and fix components.")

    def schedule(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> SchedulingDecision:
        """
        修复后的调度方法，整合了改进的奖励函数、RAG融合机制和训练策略
        
        Args:
            ready_tasks: 就绪任务列表
            simulation: 仿真环境
        
        Returns:
            调度决策
        """
        if not ready_tasks:
            return {}
        
        task_to_schedule = ready_tasks[0]  # 只调度第一个任务
        
        # 1. 获取当前状态
        state = self._extract_features([task_to_schedule], simulation)
        
        # 2. 自适应epsilon探索
        recent_performance = self.performance_history[-1] if self.performance_history else None
        epsilon = self.training_fix.adaptive_epsilon(self.episode_count, recent_performance)
        self.debug_info['epsilon_history'].append(epsilon)
        
        # 3. 获取DRL Q值
        q_values = self.drl_agent.get_q_values(state)
        
        # 4. 获取RAG建议
        rag_suggestions = self._get_rag_suggestions([task_to_schedule], simulation)
        
        # 5. 增强RAG建议
        current_loads = self._get_current_loads(simulation)
        enhanced_rag = self.rag_fusion_fix.enhance_rag_suggestions(
            rag_suggestions, self.node_names, current_loads
        )
        
        # 6. 动态融合决策
        training_progress = min(1.0, self.episode_count / self.training_fix.total_episodes)
        fusion_result = self.rag_fusion_fix.dynamic_fusion(
            q_values, enhanced_rag, current_loads, training_progress
        )
        self.debug_info['fusion_history'].append(fusion_result)
        
        # 7. 选择动作
        if random.random() < epsilon:
            # 探索模式
            action_idx = random.randint(0, len(self.node_names) - 1)
        else:
            # 利用模式
            action_idx = fusion_result['index']
        
        chosen_node = self.node_names[action_idx]
        
        # 8. 计算奖励
        teacher_makespan = self._get_teacher_makespan([task_to_schedule], simulation)
        student_makespan = self._get_student_makespan([task_to_schedule], simulation, chosen_node)
        
        # 使用修复后的奖励函数
        reward = self.drl_agent.compute_reward(
            teacher_makespan, student_makespan,
            task_scale=task_to_schedule.computation_size,
            simulation=simulation,
            task=task_to_schedule,
            chosen_node=chosen_node
        )
        
        self.total_reward += reward
        self.debug_info['reward_history'].append(reward)
        
        # 9. 记录调试信息
        self.rag_fusion_fix.debug_fusion_info(task_to_schedule.id, fusion_result)
        
        # 10. 存储经验
        next_state = self._extract_features([task_to_schedule], simulation)  # 简化处理
        self.drl_agent.store_transition(state, action_idx, reward, next_state, False)
        
        # 11. 学习更新
        if self.episode_count % 5 == 0:  # 每5步学习一次
            loss = self.drl_agent.replay(training_progress)
            if loss is not None:
                logger.info(f"Episode {self.episode_count}, Loss: {loss:.4f}")
        
        # 12. 动态目标网络更新
        if self.training_fix.dynamic_target_update(self.episode_count):
            self.drl_agent.update_target_network()
            logger.info(f"Updated target network at episode {self.episode_count}")
        
        # 13. 记录训练调试信息
        lr = self.training_fix.adaptive_learning_rate(self.episode_count)
        should_update_target = self.training_fix.dynamic_target_update(self.episode_count)
        self.training_fix.debug_training_info(self.episode_count, epsilon, lr, should_update_target)
        
        # 返回调度决策（保持与原始接口一致）
        return {chosen_node: task_to_schedule}

    def _get_rag_suggestions(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> List[Dict]:
        """
        获取RAG建议
        
        Args:
            ready_tasks: 就绪任务列表
            simulation: 仿真环境
        
        Returns:
            RAG建议列表
        """
        if not ready_tasks:
            return []
        
        task = ready_tasks[0]
        
        # 构建查询
        query = {
            'task_id': task.id,
            'computation_size': task.computation_size,
            'memory_size': getattr(task, 'memory_size', 0),
            'workflow_type': getattr(simulation.workflow, 'type', 'unknown'),
            'current_step': simulation.current_step,
            'total_steps': simulation.total_steps
        }
        
        # 从RAG知识库获取建议
        suggestions = self.rag.query(query, k=5)  # 获取前5个建议
        
        # 转换为标准格式
        formatted_suggestions = []
        for suggestion in suggestions:
            formatted_suggestions.append({
                'node': suggestion.get('node', 'unknown'),
                'score': suggestion.get('score', 0.0),
                'similarity': suggestion.get('similarity', 0.0)
            })
        
        return formatted_suggestions
    
    def _get_current_loads(self, simulation: 'WassWrenchSimulator') -> Dict[str, float]:
        """
        获取当前节点负载情况
        
        Args:
            simulation: 仿真环境
        
        Returns:
            节点负载字典
        """
        loads = {}
        for node_name in self.node_names:
            node = simulation.platform.get_node(node_name)
            if hasattr(node, 'busy_time'):
                # 计算节点利用率
                total_time = simulation.current_step
                busy_time = node.busy_time
                loads[node_name] = busy_time / max(total_time, 1.0)
            else:
                loads[node_name] = 0.0  # 默认值
        
        return loads
    
    def _get_teacher_makespan(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> float:
        """
        获取老师（预测器）建议的makespan
        
        Args:
            ready_tasks: 就绪任务列表
            simulation: 仿真环境
        
        Returns:
            最优makespan
        """
        if not ready_tasks:
            return 0.0
        
        task = ready_tasks[0]
        
        # 获取所有节点的预测makespan
        estimated_makespans = self._get_estimated_makespans(task, simulation)
        
        # 找到最优makespan
        best_makespan = min(makespan.value for makespan in estimated_makespans.values())
        
        return best_makespan
    
    def _get_student_makespan(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator', chosen_node: str) -> float:
        """
        获取学生（Agent）选择的makespan
        
        Args:
            ready_tasks: 就绪任务列表
            simulation: 仿真环境
            chosen_node: 选择的节点
        
        Returns:
            选择的makespan
        """
        if not ready_tasks:
            return 0.0
        
        task = ready_tasks[0]
        
        # 获取所有节点的预测makespan
        estimated_makespans = self._get_estimated_makespans(task, simulation)
        
        # 获取选择节点的makespan
        chosen_makespan = estimated_makespans[chosen_node].value
        
        return chosen_makespan
    
    def end_episode(self, final_makespan: float):
        """
        结束一个回合，更新统计信息
        
        Args:
            final_makespan: 最终makespan
        """
        self.episode_count += 1
        self.makespan_history.append(final_makespan)
        
        # 计算性能指标（makespan越小越好）
        if len(self.makespan_history) > 1:
            baseline_makespan = self.makespan_history[0]
            performance_improvement = (baseline_makespan - final_makespan) / baseline_makespan
            self.performance_history.append(performance_improvement)
        
        # 记录回合统计
        avg_reward = self.total_reward / max(1, len(self.debug_info['reward_history']))
        logger.info(f"Episode {self.episode_count} completed. "
                   f"Final makespan: {final_makespan:.2f}, "
                   f"Average reward: {avg_reward:.4f}")
        
        # 重置回合统计
        self.total_reward = 0.0
        self.debug_info = {
            'reward_history': [],
            'fusion_history': [],
            'epsilon_history': []
        }
    
    def fit_scaler_vectorizer(self, features_list: List[Dict]):
        """Fits the scaler and vectorizer on a list of feature dictionaries."""
        if not features_list:
            logger.warning("Feature list is empty. Cannot fit scaler/vectorizer.")
            return
        
        # 提取特征值
        feature_values = []
        for features in features_list:
            values = list(features.values())
            feature_values.append(values)
        
        # 转换为numpy数组
        feature_array = np.array(feature_values)
        
        # 拟合标准化器
        self.scaler.fit(feature_array)
        
        # 拟合向量化器
        self.vectorizer.fit(features_list)
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