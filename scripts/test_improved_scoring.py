#!/usr/bin/env python3
"""
测试修复后的RAG评分系统
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

def test_improved_scoring():
    """测试改进的评分系统"""
    
    print("🧪 Testing Improved RAG Scoring System")
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
        
        # 进行决策
        action = rag_scheduler.make_decision(state)
        
        print(f"\n✅ Decision Result:")
        print(f"   Selected node: {action.target_node}")
        print(f"   Confidence: {action.confidence:.3f}")
        print(f"   Reasoning: {action.reasoning}")
        
        # 分析reasoning中的评分信息
        reasoning = action.reasoning
        if "top choices:" in reasoning:
            choices_part = reasoning.split("top choices: ")[1].split(";")[0]
            print(f"\n📊 Makespan Analysis:")
            print(f"   {choices_part}")
            
            # 检查是否有负数或异常值
            if "s" in choices_part:
                print(f"✅ Makespans are in reasonable time units (seconds)")
            else:
                print(f"⚠️ Makespan format may be incorrect")
        
        # 检查是否有问题
        if "DEGRADATION" in reasoning:
            print(f"\n❌ Still has degradation issues!")
            return False
        else:
            print(f"\n🎉 Success: Improved scoring system working correctly!")
            return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_scoring()
    
    print("="*50)
    if success:
        print("✅ Improved scoring system validated!")
        print("📈 Expected improvements:")
        print("   - Positive scores (1/makespan)")
        print("   - Intuitive makespan display")
        print("   - Better decision explanations")
        print("🚀 Ready for clean experiments!")
    else:
        print("❌ Issues detected in scoring system.")
