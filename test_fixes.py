#!/usr/bin/env python3
"""
测试修复后的WASS-RAG系统
"""

import sys
import os
import numpy as np
import logging
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reward_fix():
    """测试修复后的奖励函数"""
    logger.info("Testing RewardFix...")
    
    from src.reward_fix import RewardFix
    
    # 创建奖励计算器
    reward_fix = RewardFix()
    
    # 测试归一化奖励
    teacher_makespan = 100.0  # 100秒
    student_makespan = 120.0  # 120秒
    task_scale = 10.0
    
    reward = reward_fix.calculate_normalized_reward(
        teacher_makespan, student_makespan, task_scale
    )
    
    logger.info(f"Normalized reward: {reward:.4f}")
    assert -1.0 <= reward <= 1.0, "Reward should be in [-1, 1] range"
    
    # 测试多目标奖励
    class MockSimulation:
        def __init__(self):
            self.completed_tasks = []
            self.workflow = MockWorkflow()
            self.platform = MockPlatform()
    
    class MockWorkflow:
        def __init__(self):
            self.tasks = [MockTask() for _ in range(10)]
    
    class MockTask:
        def __init__(self):
            self.computation_size = 10.0
    
    class MockPlatform:
        def get_node(self, name):
            return MockNode()
    
    class MockNode:
        def __init__(self):
            self.utilization = 0.5
    
    simulation = MockSimulation()
    task = MockTask()
    chosen_node = "node1"
    
    multi_reward = reward_fix.calculate_multi_objective_reward(
        simulation, task, chosen_node, teacher_makespan, student_makespan
    )
    
    logger.info(f"Multi-objective reward: {multi_reward:.4f}")
    
    logger.info("RewardFix test passed!")

def test_rag_fusion_fix():
    """测试修复后的RAG融合机制"""
    logger.info("Testing RAGFusionFix...")
    
    from src.rag_fusion_fix import RAGFusionFix
    
    # 创建RAG融合器
    rag_fusion = RAGFusionFix()
    
    # 测试RAG置信度计算
    rag_suggestions = [
        {'node': 'node1', 'score': 0.8},
        {'node': 'node2', 'score': 0.6},
        {'node': 'node3', 'score': 0.4}
    ]
    
    confidence = rag_fusion.calculate_rag_confidence(rag_suggestions)
    logger.info(f"RAG confidence: {confidence:.4f}")
    assert 0.0 <= confidence <= 1.0, "Confidence should be in [0, 1] range"
    
    # 测试RAG建议增强
    available_nodes = ['node1', 'node2', 'node3']
    current_loads = {'node1': 0.2, 'node2': 0.5, 'node3': 0.8}
    
    enhanced_suggestions = rag_fusion.enhance_rag_suggestions(
        rag_suggestions, available_nodes, current_loads
    )
    
    logger.info(f"Enhanced suggestions: {enhanced_suggestions}")
    
    # 测试全零向量处理
    zero_suggestions = [
        {'node': 'node1', 'score': 0.0},
        {'node': 'node2', 'score': 0.0},
        {'node': 'node3', 'score': 0.0}
    ]
    
    enhanced_zero = rag_fusion.enhance_rag_suggestions(
        zero_suggestions, available_nodes, current_loads
    )
    
    logger.info(f"Enhanced zero suggestions: {enhanced_zero}")
    
    # 测试动态融合
    q_values = [0.5, 0.3, 0.7]
    training_progress = 0.5
    
    fusion_result = rag_fusion.dynamic_fusion(
        q_values, rag_suggestions, current_loads, training_progress
    )
    
    logger.info(f"Fusion result: {fusion_result}")
    
    logger.info("RAGFusionFix test passed!")

def test_training_fix():
    """测试修复后的训练策略"""
    logger.info("Testing TrainingFix...")
    
    from src.training_fix import TrainingFix
    
    # 创建训练策略
    training_fix = TrainingFix(initial_epsilon=1.0, min_epsilon=0.01, total_episodes=2000)
    
    # 测试自适应epsilon
    episode = 500
    recent_performance = 0.8
    
    epsilon = training_fix.adaptive_epsilon(episode, recent_performance)
    logger.info(f"Adaptive epsilon at episode {episode}: {epsilon:.4f}")
    assert training_fix.min_epsilon <= epsilon <= training_fix.initial_epsilon, "Epsilon should be in valid range"
    
    # 测试课程学习
    task_complexity = 1.0
    adjusted_complexity = training_fix.curriculum_learning(episode, task_complexity)
    logger.info(f"Adjusted complexity at episode {episode}: {adjusted_complexity:.4f}")
    
    # 测试自适应学习率
    base_lr = 0.001
    lr = training_fix.adaptive_learning_rate(episode, base_lr)
    logger.info(f"Adaptive learning rate at episode {episode}: {lr:.6f}")
    
    # 测试动态目标网络更新
    should_update = training_fix.dynamic_target_update(episode)
    logger.info(f"Should update target network at episode {episode}: {should_update}")
    
    logger.info("TrainingFix test passed!")

def test_drl_agent_fix():
    """测试修复后的DRL代理"""
    logger.info("Testing DQNAgent with fixes...")
    
    from src.drl_agent import DQNAgent, SchedulingState
    
    # 创建DRL代理
    state_dim = 10
    action_dim = 3
    agent = DQNAgent(state_dim, action_dim)
    
    # 测试动作选择
    state = SchedulingState(np.random.rand(state_dim))
    action = agent.act(state, epsilon=0.1)
    logger.info(f"Selected action: {action}")
    assert 0 <= action < action_dim, "Action should be in valid range"
    
    # 测试奖励计算
    teacher_makespan = 100.0
    student_makespan = 120.0
    task_scale = 10.0
    
    reward = agent.compute_reward(teacher_makespan, student_makespan, task_scale)
    logger.info(f"Computed reward: {reward:.4f}")
    
    # 测试经验存储和回放
    next_state = SchedulingState(np.random.rand(state_dim))
    agent.store_transition(state, action, reward, next_state, False)
    
    # 填充一些经验
    for _ in range(50):
        s = SchedulingState(np.random.rand(state_dim))
        a = np.random.randint(0, action_dim)
        r = np.random.rand() * 2 - 1  # [-1, 1]
        ns = SchedulingState(np.random.rand(state_dim))
        agent.store_transition(s, a, r, ns, False)
    
    # 测试经验回放
    loss = agent.replay(training_progress=0.5)
    logger.info(f"Replay loss: {loss:.4f}")
    
    # 测试目标网络更新
    agent.update_target_network()
    logger.info("Target network updated")
    
    logger.info("DQNAgent test passed!")

def main():
    """运行所有测试"""
    logger.info("Starting WASS-RAG system tests...")
    
    try:
        test_reward_fix()
        test_rag_fusion_fix()
        test_training_fix()
        test_drl_agent_fix()
        
        logger.info("All tests passed! The WASS-RAG system has been successfully fixed.")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())