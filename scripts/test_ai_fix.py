#!/usr/bin/env python3
"""
测试AI调度器模块的时间戳修复
"""

import sys
import os
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

def test_timestamp_fix():
    """测试时间戳修复"""
    
    print("Testing timestamp fix...")
    
    try:
        # 测试导入
        from src.ai_schedulers import RAGKnowledgeBase
        print("✓ Successfully imported RAGKnowledgeBase")
        
        # 测试知识库创建
        kb = RAGKnowledgeBase()
        print("✓ Successfully created empty knowledge base")
        
        # 测试添加案例（这里之前出错）
        import numpy as np
        
        embedding = np.random.rand(32).astype('float32')
        workflow_info = {"task_count": 10, "type": "test"}
        actions = ["node_0", "node_1"]
        makespan = 100.0
        
        kb.add_case(embedding, workflow_info, actions, makespan)
        print("✓ Successfully added case to knowledge base")
        
        # 验证案例确实被添加了
        print(f"✓ Knowledge base now has {len(kb.cases)} cases")
        
        # 检查时间戳格式
        if kb.cases:
            timestamp = kb.cases[0]["timestamp"]
            print(f"✓ Timestamp format: {timestamp}")
            
        print("\n🎉 All tests passed! The timestamp fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_scheduler():
    """测试基础调度器功能"""
    
    print("\nTesting basic scheduler functionality...")
    
    try:
        from src.ai_schedulers import WASSHeuristicScheduler, SchedulingState
        
        # 创建启发式调度器
        scheduler = WASSHeuristicScheduler()
        print("✓ Successfully created heuristic scheduler")
        
        # 创建测试状态
        workflow = {
            "tasks": [
                {
                    "id": "task_0",
                    "flops": 1e9,
                    "memory": 1e9,
                    "dependencies": []
                }
            ]
        }
        
        cluster_state = {
            "nodes": {
                "node_0": {"cpu_capacity": 10.0, "memory_capacity": 16.0, "current_load": 0.3},
                "node_1": {"cpu_capacity": 10.0, "memory_capacity": 16.0, "current_load": 0.5}
            }
        }
        
        state = SchedulingState(
            workflow_graph=workflow,
            cluster_state=cluster_state,
            pending_tasks=["task_0"],
            current_task="task_0",
            available_nodes=["node_0", "node_1"],
            timestamp=1234567890.0
        )
        
        # 测试决策
        decision = scheduler.make_decision(state)
        print(f"✓ Successfully made scheduling decision: {decision.target_node}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning}")
        
        return True
        
    except Exception as e:
        print(f"✗ Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== WASS-RAG AI Scheduler Test ===")
    
    # 测试时间戳修复
    timestamp_ok = test_timestamp_fix()
    
    # 测试基础调度器
    scheduler_ok = test_basic_scheduler()
    
    if timestamp_ok and scheduler_ok:
        print("\n🎉 All tests passed! The AI scheduler module is working correctly.")
        print("\nYou can now safely run:")
        print("  python scripts/initialize_ai_models.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
