#!/usr/bin/env python3
"""
快速测试WASS-RAG预测是否合理
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from ai_schedulers import WASSRAGScheduler, SchedulingState
    print("✅ Successfully imported AI schedulers")
    
    # 创建调度器
    scheduler = WASSRAGScheduler("models/wass_models.pth", "data/knowledge_base.pkl")
    print("✅ Successfully initialized WASS-RAG scheduler")
    
    # 创建测试状态
    test_workflow = {
        "tasks": [
            {"id": "task_0", "flops": 5e9, "memory": 2.0},
            {"id": "task_1", "flops": 3e9, "memory": 1.5}
        ]
    }
    
    test_cluster = {
        "nodes": {
            "node_0": {"cpu_capacity": 2.0, "memory_capacity": 16.0, "current_load": 0.3},
            "node_1": {"cpu_capacity": 3.0, "memory_capacity": 32.0, "current_load": 0.5},
            "node_2": {"cpu_capacity": 1.5, "memory_capacity": 12.0, "current_load": 0.7}
        }
    }
    
    state = SchedulingState(
        workflow_graph=test_workflow,
        cluster_state=test_cluster,
        pending_tasks=["task_0", "task_1"],
        current_task="task_0",
        available_nodes=["node_0", "node_1", "node_2"],
        timestamp=0.0
    )
    
    # 测试调度决策
    print("\n🧪 Testing WASS-RAG predictions:")
    action = scheduler.schedule(state)
    print(f"✅ Scheduling decision: {action.task_id} -> {action.target_node}")
    print(f"   Confidence: {action.confidence:.3f}")
    if action.reasoning:
        print(f"   Reasoning: {action.reasoning}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
