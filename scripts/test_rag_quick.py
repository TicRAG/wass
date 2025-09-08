#!/usr/bin/env python3
"""
快速测试RAG调度器修复
运行一个简单的调度决策来查看调试输出
"""

import os
import sys
import json

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    from src.ai_schedulers import create_scheduler, SchedulingState
    print("✓ Successfully imported schedulers")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def create_simple_test_state():
    """创建简单的测试状态"""
    
    # 简单的工作流图
    workflow_graph = {
        "tasks": ["task_0"],
        "dependencies": {},
        "task_requirements": {
            "task_0": {"cpu": 2.0, "memory": 4.0, "duration": 5.0}
        }
    }
    
    # 简单的集群状态
    cluster_state = {
        "nodes": {
            "node_0": {"cpu_capacity": 10.0, "memory_capacity": 16.0, "current_load": 0.3},
            "node_1": {"cpu_capacity": 10.0, "memory_capacity": 16.0, "current_load": 0.5},
            "node_2": {"cpu_capacity": 10.0, "memory_capacity": 16.0, "current_load": 0.7},
            "node_3": {"cpu_capacity": 10.0, "memory_capacity": 16.0, "current_load": 0.2}
        }
    }
    
    return SchedulingState(
        workflow_graph=workflow_graph,
        cluster_state=cluster_state,
        pending_tasks=[],
        current_task="task_0",
        available_nodes=["node_0", "node_1", "node_2", "node_3"],
        timestamp=1725782400.0
    )

def test_rag_decision():
    """测试RAG调度器决策过程"""
    
    print("=== Testing RAG Scheduler Decision ===")
    
    try:
        # 创建调度器
        rag_scheduler = create_scheduler(
            "WASS-RAG",
            model_path="models/wass_models.pth",
            knowledge_base_path="data/knowledge_base.pkl"
        )
        print("✓ RAG scheduler created successfully")
        
        # 创建测试状态
        state = create_simple_test_state()
        print("✓ Test state created")
        
        print(f"\nMaking scheduling decision...")
        print(f"Available nodes: {state.available_nodes}")
        print(f"Current task: {state.current_task}")
        
        # 进行决策（这会打印调试信息）
        action = rag_scheduler.make_decision(state)
        
        print(f"\n=== Decision Result ===")
        print(f"Selected node: {action.target_node}")
        print(f"Confidence: {action.confidence:.3f}")
        if action.reasoning:
            print(f"Reasoning: {action.reasoning}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test RAG decision: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("RAG Scheduler Quick Test")
    print("This will show debug output from the scheduling decision")
    print("="*60)
    
    success = test_rag_decision()
    
    print(f"\n{'='*60}")
    if success:
        print("✅ Test completed successfully!")
        print("   Check the debug output above to see if predictions are diverse")
        print("   If you see 'DEGRADATION' warnings, the issue persists")
        print("   If you see diverse node scores, the fix worked!")
    else:
        print("❌ Test failed!")
        print("   Check the error output above")
    print(f"{'='*60}")
