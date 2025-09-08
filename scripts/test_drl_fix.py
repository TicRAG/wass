#!/usr/bin/env python3
"""
测试修复后的DRL调度器功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_mock_state(task_count: int, node_count: int, current_task: str):
    """创建模拟的调度状态"""
    try:
        from src.ai_schedulers import SchedulingState
        
        # 创建模拟工作流图
        tasks = []
        for i in range(task_count):
            task = {
                "id": f"task_{i}",
                "flops": 1e9 + i * 1e8,  # 1-2 GFlops
                "memory": 1e9 + i * 1e8,  # 1-2 GB
                "dependencies": [f"task_{j}" for j in range(max(0, i-2), i)] if i > 0 else []
            }
            tasks.append(task)
        
        workflow_graph = {
            "tasks": tasks,
            "name": "test_workflow"
        }
        
        # 创建模拟集群状态
        cluster_state = {
            "nodes": {
                f"node_{i}": {
                    "cpu_capacity": 10.0,
                    "memory_capacity": 16.0,
                    "current_load": 0.3 + (i * 0.1),
                    "available": True
                }
                for i in range(node_count)
            }
        }
        
        # 创建调度状态
        state = SchedulingState(
            workflow_graph=workflow_graph,
            cluster_state=cluster_state,
            pending_tasks=[f"task_{i}" for i in range(task_count) if f"task_{i}" != current_task],
            current_task=current_task,
            available_nodes=[f"node_{i}" for i in range(node_count)],
            timestamp=1234567890.0
        )
        
        return state
        
    except ImportError:
        # 如果没有AI模块，返回简单的mock对象
        class MockState:
            def __init__(self):
                self.workflow_graph = {"tasks": [], "name": "mock"}
                self.cluster_state = {"nodes": {}}
                self.pending_tasks = []
                self.current_task = current_task
                self.available_nodes = [f"node_{i}" for i in range(node_count)]
                self.timestamp = 1234567890.0
        
        return MockState()

def test_drl_schedulers():
    """测试DRL调度器的修复"""
    
    try:
        from src.ai_schedulers import WASSSmartScheduler, WASSRAGScheduler
        
        print("=== DRL调度器修复测试 ===")
        
        # 1. 测试WASSSmartScheduler
        print("\n1. 测试WASS-DRL (w/o RAG)调度器...")
        
        smart_scheduler = WASSSmartScheduler("models/wass_models.pth")
        print(f"   ✓ 成功创建{smart_scheduler.name}调度器")
        
        # 创建模拟状态
        mock_state = create_mock_state(
            task_count=10,
            node_count=4,
            current_task="task_0"
        )
        
        # 测试决策制定
        action = smart_scheduler.make_decision(mock_state)
        print(f"   ✓ 成功制定决策: {action.task_id} -> {action.target_node}")
        print(f"     置信度: {action.confidence:.3f}")
        print(f"     推理: {action.reasoning}")
        
        # 2. 测试WASSRAGScheduler  
        print("\n2. 测试WASS-RAG调度器...")
        
        rag_scheduler = WASSRAGScheduler(
            model_path="models/wass_models.pth",
            knowledge_base_path="data/wass_knowledge_base.pkl"
        )
        print(f"   ✓ 成功创建{rag_scheduler.name}调度器")
        
        # 测试决策制定
        action = rag_scheduler.make_decision(mock_state)
        print(f"   ✓ 成功制定决策: {action.task_id} -> {action.target_node}")
        print(f"     置信度: {action.confidence:.3f}")
        print(f"     推理: {action.reasoning}")
        
        # 3. 测试多个决策
        print("\n3. 测试连续决策制定...")
        
        for i in range(3):
            test_state = create_mock_state(
                task_count=10,
                node_count=4,
                current_task=f"task_{i+1}"
            )
            
            smart_action = smart_scheduler.make_decision(test_state)
            rag_action = rag_scheduler.make_decision(test_state)
            
            print(f"   任务{i+1}:")
            print(f"     DRL: {smart_action.target_node} (置信度: {smart_action.confidence:.3f})")
            print(f"     RAG: {rag_action.target_node} (置信度: {rag_action.confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_drl_schedulers()
    
    if success:
        print("\n🎉 DRL调度器修复测试成功!")
        print("现在可以运行完整的实验框架")
    else:
        print("\n❌ 仍有问题需要解决")
