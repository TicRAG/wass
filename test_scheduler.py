#!/usr/bin/env python3
"""
测试修复后的WASS-RAG调度器
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

def create_mock_environment():
    """创建模拟环境"""
    
    class MockTask:
        def __init__(self, task_id, computation_size=10.0):
            self.id = task_id
            self.computation_size = computation_size
    
    class MockNode:
        def __init__(self, name):
            self.name = name
            self.busy_time = 0
            self.utilization = 0.5
    
    class MockPlatform:
        def __init__(self, node_names):
            self.nodes = {name: MockNode(name) for name in node_names}
        
        def get_node(self, name):
            return self.nodes.get(name)
    
    class MockWorkflow:
        def __init__(self, num_tasks=10):
            self.tasks = [MockTask(f"task_{i}", np.random.randint(5, 20)) for i in range(num_tasks)]
            self.type = "scientific"
    
    class MockSimulation:
        def __init__(self, num_nodes=3, num_tasks=10):
            self.platform = MockPlatform([f"node_{i}" for i in range(num_nodes)])
            self.workflow = MockWorkflow(num_tasks)
            self.current_step = 0
            self.total_steps = 100
            self.completed_tasks = []
            self.ready_tasks = self.workflow.tasks.copy()
    
    return MockSimulation()

def test_wass_rag_scheduler():
    """测试修复后的WASS-RAG调度器"""
    logger.info("Testing WASS-RAG Scheduler...")
    
    # 创建模拟环境
    simulation = create_mock_environment()
    
    # 创建配置
    config = {
        'initial_epsilon': 1.0,
        'min_epsilon': 0.01,
        'total_episodes': 2000
    }
    
    # 创建模拟的预测器和RAG知识库
    class MockPredictor:
        @staticmethod
        def load(path):
            return MockPredictor()
        
        def predict(self, features):
            # 返回随机预测值
            return np.random.rand() * 100 + 50
    
    class MockRAGKnowledgeBase:
        @staticmethod
        def load(path):
            return MockRAGKnowledgeBase()
        
        def query(self, query, k=5):
            # 返回随机建议
            suggestions = []
            for i in range(k):
                node_name = f"node_{i % 3}"
                score = np.random.rand()
                suggestions.append({
                    'node': node_name,
                    'score': score,
                    'similarity': np.random.rand()
                })
            return suggestions
    
    # 替换原始类
    sys.modules['src.performance_predictor'] = type('module', (), {
        'PerformancePredictor': MockPredictor
    })()
    
    sys.modules['src.rag_knowledge_base'] = type('module', (), {
        'RAGKnowledgeBase': MockRAGKnowledgeBase
    })()
    
    # 创建模拟的get_logger函数
def mock_get_logger(*args, **kwargs):
    name = args[0] if args else __name__
    if not isinstance(name, str):
        name = str(name)
    return logging.getLogger(name)
    
    sys.modules['src.utils'] = type('module', (), {
        'get_logger': mock_get_logger
    })()
    
    # 导入修复后的调度器
    from src.ai_schedulers import WASSRAGScheduler
    
    # 创建调度器
    node_names = [f"node_{i}" for i in range(3)]
    predictor = MockPredictor()
    rag = MockRAGKnowledgeBase()
    
    # 创建模拟的DRL代理
    from src.drl_agent import DQNAgent
    
    drl_agent = DQNAgent(state_dim=10, action_dim=3)
    
    # 创建调度器
    scheduler = WASSRAGScheduler(drl_agent, node_names, predictor)
    
    # 测试调度
    for episode in range(5):
        logger.info(f"Episode {episode + 1}")
        
        # 重置模拟环境
        simulation = create_mock_environment()
        
        # 运行调度
        step = 0
        while simulation.ready_tasks and step < 20:
            # 获取就绪任务
            ready_tasks = simulation.ready_tasks[:1]  # 每次只调度一个任务
            
            if ready_tasks:
                # 调度任务
                decision = scheduler.schedule(ready_tasks, simulation)
                
                # 更新模拟环境
                for node, task in decision.items():
                    simulation.ready_tasks.remove(task)
                    simulation.completed_tasks.append(task)
                    simulation.platform.get_node(node).busy_time += task.computation_size
                
                simulation.current_step += 1
                step += 1
        
        # 结束回合
        final_makespan = simulation.current_step * 10  # 简化的makespan计算
        scheduler.end_episode(final_makespan)
        
        logger.info(f"Episode {episode + 1} completed with makespan: {final_makespan}")
    
    logger.info("WASS-RAG Scheduler test passed!")

def main():
    """运行测试"""
    logger.info("Starting WASS-RAG Scheduler experiment...")
    
    try:
        test_wass_rag_scheduler()
        logger.info("Experiment completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())