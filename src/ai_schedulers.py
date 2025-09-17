import logging
import random
import os
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
from src.knowledge_base.wrench_full_kb import WRENCHRAGKnowledgeBase, WRENCHKnowledgeCase

logger = get_logger(__name__, logging.WARNING)


class WASSScheduler(Scheduler):
    """Abstract base class for WASS Schedulers."""

    def schedule(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> SchedulingDecision:
        raise NotImplementedError


class WASSDRLScheduler(WASSScheduler):
    """WASS-DRL Scheduler."""

    def __init__(self, drl_agent: DQNAgent, node_names: List[str], predictor: PerformancePredictor = None):
        self.drl_agent = drl_agent
        self.node_names = node_names
        self.node_map = {name: i for i, name in enumerate(node_names)}
        self.predictor = predictor  # 添加predictor属性，即使为None
        self.sim = None
        self.compute_service = None
        self.hosts = None
        logger.info(f"Initialized WASSDRLScheduler with {len(node_names)} nodes.")

    def set_simulation_context(self, simulation, compute_service, hosts):
        """设置仿真上下文"""
        self.sim = simulation
        self.compute_service = compute_service
        self.hosts = hosts

    def schedule_ready_tasks(self, workflow, storage_service):
        """调度所有准备就绪的任务"""
        ready_tasks = workflow.get_ready_tasks()
        for task in ready_tasks:
            decision = self.schedule([task], workflow, storage_service)
            if decision:
                # 创建标准作业并提交
                task_list = list(decision.values())
                file_locations = {}
                for t in task_list:
                    for f in t.get_input_files():
                        file_locations[f] = storage_service
                    for f in t.get_output_files():
                        file_locations[f] = storage_service
                
                job = self.sim.create_standard_job([task], file_locations)
                self.compute_service.submit_standard_job(job)

    def schedule(self, ready_tasks: List[w.Task], workflow=None, storage_service=None) -> SchedulingDecision:
        """调度所有就绪任务"""
        if not ready_tasks:
            return {}

        scheduling_decisions = {}
        
        # 调度所有就绪任务，而不是只调度第一个
        for task in ready_tasks:
            state, _ = self._extract_features(task, workflow)
            
            # 使用DRL agent选择动作，启用探索机制
            # 随着训练进行逐步降低epsilon值
            epsilon = 0.1  # 固定epsilon值用于推理
            
            # Convert SchedulingState to numpy array before passing to DRL agent
            state_array = state.to_array()
            action_idx = self.drl_agent.act(state_array, epsilon=epsilon)
            chosen_node_name = self.node_names[action_idx]
            
            logger.debug(f"DRL Agent chose action {action_idx} -> Node {chosen_node_name} for task {task.get_name()} (epsilon={epsilon:.2f})")
            
            scheduling_decisions[chosen_node_name] = task
        
        return scheduling_decisions

    def _extract_features(self, task: w.Task, workflow=None) -> SchedulingState:
        """提取图结构特征，构建GNN输入"""
        
        # 简化特征提取，适应新的接口
        # 任务基本特征（3个维度）
        task_features = np.array([
            task.get_flops() if hasattr(task, 'get_flops') else 1000,  # 计算量
            len(task.get_parents()) if hasattr(task, 'get_parents') else 0,  # 父任务数
            len(task.get_children()) if hasattr(task, 'get_children') else 0,  # 子任务数
        ], dtype=np.float32)
        
        # 平台状态特征（2个维度 - 简化版）
        node_features = np.array([
            1.0,  # 节点速度（简化）
            0.0,  # 队列长度（简化）
        ], dtype=np.float32)
        
        # 创建空的图数据（简化处理）
        import torch
        graph_data = type('GraphData', (), {'x': torch.zeros((1, 10))})()
        
        return SchedulingState(task_features, node_features), graph_data

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
            
            # Use task name as identifier since wrench Task doesn't have id
            task_name = task.get_name()
            dag.add_node(task_name, 
                computation_size=getattr(task, 'computation_size', 0),
                parents=[p.get_name() for p in getattr(task, 'parents', [])],
                children=[c.get_name() for c in getattr(task, 'children', [])],
                is_submitted=is_submitted,
                is_finished=is_finished,
                assigned_host_id=assigned_host_id
            )
        for task in workflow.tasks:
            for child in getattr(task, 'children', []):
                data_size = workflow.get_edge_data_size(task.get_name(), child.get_name()) if hasattr(workflow, 'get_edge_data_size') else 0
                dag.add_edge(task.get_name(), child.get_name(), data_size=data_size)
        return dag

    def _get_estimated_makespans(self, task: w.Task, simulation: 'WassWrenchSimulator') -> Dict[str, PredictedValue]:
        """Estimate makespan for each possible node assignment using the GNN predictor.
        
        Returns:
            Dict mapping node names to predicted makespan values.
        """
        try:
            # 构建当前工作流的DAG图
            dag = self._build_dag_graph(simulation.workflow)
            
            # 更新当前任务的状态信息
            task_name = task.get_name()
            if task_name in dag.nodes:
                dag.nodes[task_name]['is_submitted'] = True
                
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
                try:
                    # 创建临时的节点分配
                    temp_task_to_node_map = simulation.task_to_node_map.copy()
                    temp_task_to_node_map[task] = node_name
                    
                    # 更新DAG中任务的分配信息
                    temp_dag = dag.copy()
                    if task_name in temp_dag.nodes:
                        temp_dag.nodes[task_name]['assigned_host_id'] = self.node_names.index(node_name)
                    
                    # 提取图特征
                    graph_data = self.predictor.extract_graph_features(temp_dag, node_features, focus_task_id=task_name)
                    
                    # 使用GNN预测器进行预测
                    predicted_makespan = self.predictor.predict(graph_data)
                    
                    estimated_makespans[node_name] = PredictedValue(
                        value=predicted_makespan,
                        confidence=1.0  # 简化处理，实际应用中可以计算置信度
                    )
                except Exception as e:
                    # 如果单个节点的预测失败，使用一个合理的默认值
                    print(f"[警告] 节点 {node_name} 的makespan预测失败: {e}")
                    # 使用基于节点速度的启发式值
                    node = simulation.platform.get_node(node_name)
                    default_makespan = 1000.0 / max(node.speed, 1.0)  # 避免除零
                    estimated_makespans[node_name] = PredictedValue(
                        value=default_makespan,
                        confidence=0.5
                    )
            
            return estimated_makespans
            
        except Exception as e:
            # 如果整个预测过程失败，为所有节点返回相同的默认值
            print(f"[错误] Makespan预测过程失败: {e}")
            estimated_makespans = {}
            default_makespan = 1000.0
            for node_name in self.node_names:
                estimated_makespans[node_name] = PredictedValue(
                    value=default_makespan,
                    confidence=0.1
                )
            return estimated_makespans


class WASSRAGScheduler(WASSDRLScheduler):
    """修复后的WASS-RAG调度器，整合了改进的奖励函数、RAG融合机制和训练策略"""

    def __init__(self, drl_agent: DQNAgent, node_names: List[str], predictor: PerformancePredictor, knowledge_base_path: str = None):
        print("DEBUG: WASSRAGScheduler __init__ called!")
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

        # 初始化RAG知识库
        self.rag = WRENCHRAGKnowledgeBase(embedding_dim=64)
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            print(f"DEBUG: Loading knowledge base from {knowledge_base_path}")
            self.rag.load_knowledge_base(knowledge_base_path)
            print(f"DEBUG: Loaded {len(self.rag.cases)} cases from knowledge base")
        else:
            print(f"DEBUG: No knowledge base file found at {knowledge_base_path}")
        
        # 训练统计
        self.episode_count = 0
        self.total_reward = 0.0
        self.makespan_history = []
        self.performance_history = []
        
        # 任务调度跟踪
        self.scheduled_tasks = {}  # task -> node mapping
        
        # 调试信息
        self.debug_info = {
            'reward_history': [],
            'fusion_history': [],
            'epsilon_history': []
        }
        
        logger.info("Initialized WASSRAGScheduler with R_RAG dynamic reward mechanism and fix components.")

    def schedule(self, ready_tasks: List[w.Task], workflow=None, storage_service=None) -> Dict[str, w.Task]:
        """
        优化后的调度方法，修复了训练稳定性问题并增强了RAG性能
        
        Args:
            ready_tasks: 就绪任务列表
            workflow: 工作流对象（可选）
            storage_service: 存储服务（可选）
        
        Returns:
            调度决策
        """
        if not ready_tasks:
            return {}
        
        # 调度所有就绪任务，而不是只调度第一个
        scheduling_decisions = {}
        
        # 获取图数据（用于RAG建议）
        graph_data = None
        if workflow and ready_tasks:
            try:
                state, graph_data = self._extract_features(ready_tasks[0], workflow)
            except:
                pass
        
        # 获取RAG建议（批量处理所有任务）
        rag_suggestions = self._get_rag_suggestions(ready_tasks, graph_data=graph_data, workflow=workflow)
        
        # 增强RAG建议（基于实际负载）
        current_loads = self._get_current_loads()
        enhanced_rag = self.rag_fusion_fix.enhance_rag_suggestions(
            rag_suggestions, self.node_names, current_loads
        )
        
        for task in ready_tasks:
            # 1. 获取当前状态
            state, graph_data = self._extract_features(task, workflow)
            
            # 2. 优化epsilon策略：使用更保守的探索
            base_epsilon = 0.1  # 降低基础探索率
            min_epsilon = 0.01
            decay_rate = 0.995
            epsilon = max(min_epsilon, base_epsilon * (decay_rate ** self.episode_count))
            
            # 3. 获取DRL Q值
            state_tensor = torch.FloatTensor(state.to_array()).unsqueeze(0)
            q_values = self.drl_agent.forward(state_tensor).detach().numpy()[0]
            
            # 4. 获取该任务的RAG建议
            task_rag_suggestions = [s for s in enhanced_rag if s.get('task_id') == task.get_name()]
            
            # 5. 选择动作 - 智能融合DRL和RAG
            action_idx = self._select_action_with_fusion(q_values, task_rag_suggestions, epsilon)
            
            chosen_node = self.node_names[action_idx]
            
            # 6. 记录调度决策
            scheduling_decisions[chosen_node] = task
            
            # 7. 更新任务跟踪和负载信息
            self.scheduled_tasks[task] = chosen_node
            current_loads[chosen_node] = current_loads.get(chosen_node, 0.0) + 1.0
        
        return scheduling_decisions
    
    def _select_action_with_fusion(self, q_values: np.ndarray, rag_suggestions: List[Dict], epsilon: float) -> int:
        """智能融合DRL和RAG建议选择动作"""
        # 增强RAG融合：即使不在探索模式，也考虑RAG建议
        if len(rag_suggestions) > 0:
            # 计算RAG建议的权重
            rag_weights = []
            node_scores = {node: 0.0 for node in self.node_names}
            
            for suggestion in rag_suggestions:
                node_name = suggestion['node']
                if node_name in node_scores:
                    # 使用confidence和similarity计算权重
                    weight = suggestion['confidence'] * suggestion.get('similarity', 0.5)
                    # 如果有makespan信息，优先选择makespan较小的节点
                    if 'makespan' in suggestion.get('metadata', {}):
                        makespan = suggestion['metadata']['makespan']
                        # makespan越小越好，所以用倒数
                        weight *= (1.0 / (makespan + 1.0))
                    node_scores[node_name] += weight
            
            # 如果RAG建议足够强，优先使用RAG建议
            max_rag_score = max(node_scores.values())
            if max_rag_score > 0.5:  # RAG建议置信度阈值
                # 找到RAG建议的最佳节点
                best_rag_node = max(node_scores.keys(), key=lambda x: node_scores[x])
                best_rag_idx = self.node_names.index(best_rag_node)
                
                # 以高概率选择RAG建议，小概率探索
                if random.random() < 0.9:  # 90%概率选择RAG建议
                    return best_rag_idx
                else:
                    # 10%概率随机探索
                    return random.randint(0, len(self.node_names) - 1)
        
        # 默认使用DRL决策
        if random.random() < epsilon:
            # 探索模式：随机选择
            return random.randint(0, len(self.node_names) - 1)
        else:
            # 利用模式：选择DRL最佳决策
            return np.argmax(q_values)

    def _get_rag_suggestions(self, ready_tasks: List[w.Task], graph_data=None, workflow=None) -> List[Dict]:
        """
        获取RAG建议（基于知识库）
        
        Args:
            ready_tasks: 就绪任务列表
            graph_data: 可选，已经提取好的图特征数据
            workflow: 工作流对象
        
        Returns:
            RAG建议列表
        """
        if not ready_tasks:
            return []
        
        suggestions = []
        
        for task in ready_tasks:
            task_name = task.get_name()
            
            # 获取任务特征
            task_features = self._extract_task_features(task, workflow)
            
            # 查询RAG知识库
            # 创建工作流嵌入（基于任务特征和工作流结构）
            workflow_embedding = self._create_workflow_embedding(task, workflow)
            task_features_array = np.array([
                task_features.get('flops', 0.0),
                task_features.get('input_files', 0),
                task_features.get('output_files', 0),
                len(task_features.get('task_name', '')),
                0.0  # 占位符
            ])
            
            rag_results = self.rag.retrieve_similar_cases(workflow_embedding, task_features_array, k=3)
            
            if rag_results:
                # 使用知识库中的建议
                for case, similarity in rag_results:
                    node_name = case.chosen_node
                    confidence = similarity
                    reason = f"rag_suggestion_makespan_{case.workflow_makespan:.2f}"
                    
                    suggestions.append({
                        'task_id': task_name,
                        'node': node_name,
                        'confidence': confidence,
                        'reason': reason,
                        'metadata': {
                            'case': case,
                            'similarity': similarity,
                            'makespan': case.workflow_makespan
                        }
                    })
                
                logger.info(f"RAG found {len(rag_results)} suggestions for task {task_name}")
            else:
                logger.warning(f"RAG found no suggestions for task {task_name}, using heuristic fallback")
                # 回退到启发式建议
                for node_name in self.node_names:
                    suggestions.append({
                        'task_id': task_name,
                        'node': node_name,
                        'confidence': 0.3,
                        'reason': 'heuristic_fallback',
                        'metadata': {}
                    })
        
        return suggestions
    
    def _create_workflow_embedding(self, task: w.Task, workflow=None) -> np.ndarray:
        """创建工作流嵌入向量（基于任务特征和工作流结构）"""
        embedding = np.zeros(64)
        
        if workflow:
            # 获取工作流的基本特征
            all_tasks = workflow.get_tasks()
            workflow_size = len(all_tasks)
            
            # 计算任务在工作流中的位置
            task_name = task.get_name()
            
            # 基本特征
            embedding[0] = float(workflow_size) / 20.0  # 归一化工作流大小
            embedding[1] = float(len(task.get_input_files())) / 10.0
            embedding[2] = float(len(task.get_output_files())) / 10.0
            embedding[3] = task.get_flops() / 10000.0  # 归一化FLOPS
            
            # 任务依赖特征
            try:
                # 计算任务的依赖数量
                dependencies = len(task.get_parents())
                children = len(task.get_children())
                embedding[4] = float(dependencies) / 5.0
                embedding[5] = float(children) / 5.0
            except:
                pass
            
            # 工作流结构特征
            try:
                # 计算工作流的关键路径长度（简化版）
                ready_tasks = len(workflow.get_ready_tasks())
                embedding[6] = float(ready_tasks) / float(workflow_size)
            except:
                pass
            
            # 任务名称特征（简单的哈希）
            task_name_hash = hash(task_name) % 1000 / 1000.0
            embedding[7] = task_name_hash
            
            # 平台特征（如果有的话）
            if hasattr(self, 'node_names'):
                embedding[8] = float(len(self.node_names)) / 10.0
        
        return embedding
    
    def _extract_task_features(self, task: w.Task, workflow=None) -> Dict:
        """提取任务特征用于RAG查询"""
        features = {
            'task_name': task.get_name(),
            'flops': task.get_flops(),
            'input_files': len(task.get_input_files()),
            'output_files': len(task.get_output_files())
        }
        
        if workflow:
            # 获取任务在工作流中的位置信息
            try:
                features['workflow_size'] = len(workflow.get_tasks())
                features['ready_tasks_count'] = len(workflow.get_ready_tasks())
            except:
                pass
        
        return features
    
    def _get_current_loads(self, simulation=None) -> Dict[str, float]:
        """获取当前节点负载（基于已调度任务）"""
        loads = {}
        
        # 基于已调度的任务计算负载
        if hasattr(self, 'scheduled_tasks'):
            for node_name in self.node_names:
                # 计算该节点上正在执行的任务数
                node_load = len([task for task, node in self.scheduled_tasks.items() if node == node_name])
                loads[node_name] = float(node_load)
        else:
            # 初始化负载为零
            for node_name in self.node_names:
                loads[node_name] = 0.0
        
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
    
    def _record_decision(self, task, node, score):
        """记录调度决策（简化版）"""
        logger.debug(f"Scheduled task {task.get_name()} to node {node} with score {score}")
    
    def set_simulation_context(self, sim, compute_services, hosts):
        """设置仿真上下文"""
        self.sim = sim
        self.compute_services = compute_services
        self.hosts = hosts
        self.node_names = hosts if isinstance(hosts, list) else list(hosts.keys())
    
    def schedule_ready_tasks(self, workflow, storage_service):
        """调度就绪任务"""
        ready_tasks = workflow.get_ready_tasks()
        if not ready_tasks:
            return
        
        # 调度每个就绪任务
        for task in ready_tasks:
            decision = self.schedule([task])
            for node_name, task_to_schedule in decision.items():
                # 创建标准作业
                file_locations = {}
                for f in task_to_schedule.get_input_files():
                    file_locations[f] = storage_service
                for f in task_to_schedule.get_output_files():
                    file_locations[f] = storage_service
                
                job = self.sim.create_standard_job([task_to_schedule], file_locations)
                
                # 提交作业到选定的主机对应的计算服务
                if node_name in self.compute_services:
                    self.compute_services[node_name].submit_standard_job(job)
                    logger.info(f"Submitted task {task_to_schedule.get_name()} to node {node_name}")
                else:
                    # 回退到第一个可用的计算服务
                    first_service = list(self.compute_services.values())[0]
                    first_service.submit_standard_job(job)
                    logger.warning(f"Node {node_name} not found in compute services, using fallback node")
    
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
        return WASSDRLScheduler(drl_agent, node_names, predictor)
    elif name == 'WASS-RAG':
        if drl_agent is None or predictor is None:
            # 如果没有提供完整的组件，创建一个改进的启发式调度器
            from .simple_schedulers import ImprovedHeuristicScheduler  
            return ImprovedHeuristicScheduler(name)
        return WASSRAGScheduler(drl_agent, node_names, predictor, knowledge_base_path)
    elif name == 'WASS (Heuristic)':
        from .simple_schedulers import HeuristicScheduler
        return HeuristicScheduler(name)
    else:
        raise ValueError(f'Unknown scheduler name: {name}')