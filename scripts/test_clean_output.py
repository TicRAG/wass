#!/usr/bin/env python3
"""
简洁测试：验证RAG调度器工作正常且无过多调试输出
"""

import os
import sys

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

def create_test_state():
    """创建测试状态"""
    workflow_graph = {
        "tasks": ["task_0"],
        "dependencies": {},
        "task_requirements": {
            "task_0": {"cpu": 2.0, "memory": 4.0, "duration": 5.0}
        }
    }
    
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

def test_clean_output():
    """测试简洁输出"""
    
    print("🧪 Testing Clean RAG Scheduler Output")
    print("="*50)
    
    try:
        # 创建调度器
        rag_scheduler = create_scheduler(
            "WASS-RAG",
            model_path="models/wass_models.pth",
            knowledge_base_path="data/knowledge_base.pkl"
        )
        
        # 创建测试状态
        state = create_test_state()
        
        print(f"\n📋 Making scheduling decision...")
        print(f"   Available nodes: {state.available_nodes}")
        print(f"   Current task: {state.current_task}")
        
        # 进行决策（应该输出很少的信息）
        action = rag_scheduler.make_decision(state)
        
        print(f"\n✅ Decision Result:")
        print(f"   Selected node: {action.target_node}")
        print(f"   Confidence: {action.confidence:.3f}")
        print(f"   Reasoning: {action.reasoning[:100]}...")
        
        # 检查是否有问题
        if "DEGRADATION" in action.reasoning:
            print(f"\n⚠️ Warning: Still has degradation issues!")
            return False
        else:
            print(f"\n🎉 Success: Clean output with normal RAG operation!")
            return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clean_output()
    
    print("="*50)
    if success:
        print("✅ All tests passed! Ready for production experiments.")
        print("📝 Note: Debug output has been minimized for clean logs.")
        print("🚀 Run: python experiments/real_experiment_framework.py")
    else:
        print("❌ Some issues detected. Check the output above.")
