#!/usr/bin/env python3
"""
详细调试RAG调度器的数据结构问题
"""

import os
import sys
import json
import traceback

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
        "tasks": ["task_0"],  # 字符串格式的任务列表
        "dependencies": {},   # 依赖关系
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
    
    state = SchedulingState(
        workflow_graph=workflow_graph,
        cluster_state=cluster_state,
        pending_tasks=[],
        current_task="task_0",
        available_nodes=["node_0", "node_1", "node_2", "node_3"],
        timestamp=1725782400.0
    )
    
    print("📋 Test State Created:")
    print(f"  - Workflow tasks: {workflow_graph['tasks']}")
    print(f"  - Current task: {state.current_task}")
    print(f"  - Available nodes: {state.available_nodes}")
    print(f"  - Task requirements: {workflow_graph['task_requirements']}")
    
    return state

def test_feature_extraction():
    """测试特征提取过程"""
    print("\n=== Testing Feature Extraction ===")
    
    try:
        # 创建基础调度器来测试特征提取
        from src.ai_schedulers import WASSSmartScheduler
        
        scheduler = WASSSmartScheduler()
        state = create_simple_test_state()
        
        print("🔍 Testing _extract_simple_features...")
        features = scheduler._extract_simple_features(state)
        print(f"✓ Features extracted successfully: shape={features.shape}")
        print(f"  First few features: {features[:10].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        traceback.print_exc()
        return False

def test_graph_construction():
    """测试图构建过程"""
    print("\n=== Testing Graph Construction ===")
    
    try:
        from src.ai_schedulers import WASSSmartScheduler
        
        scheduler = WASSSmartScheduler()
        state = create_simple_test_state()
        
        print("🔍 Testing _build_graph_data...")
        graph_data = scheduler._build_graph_data(state)
        
        if graph_data is None:
            print("⚠️ Graph data is None (expected if no torch_geometric)")
        else:
            print(f"✓ Graph data created successfully")
            print(f"  Node features shape: {graph_data.x.shape}")
            print(f"  Edge index shape: {graph_data.edge_index.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Graph construction failed: {e}")
        traceback.print_exc()
        return False

def test_rag_components():
    """测试RAG组件"""
    print("\n=== Testing RAG Components ===")
    
    try:
        from src.ai_schedulers import WASSRAGScheduler
        
        print("🔍 Creating RAG scheduler...")
        rag_scheduler = WASSRAGScheduler(
            model_path="models/wass_models.pth",
            knowledge_base_path="data/knowledge_base.pkl"
        )
        print("✓ RAG scheduler created")
        
        state = create_simple_test_state()
        
        print("🔍 Testing action encoding...")
        action_embedding = rag_scheduler._encode_action("node_0", state)
        print(f"✓ Action encoded: shape={action_embedding.shape}")
        
        print("🔍 Testing state feature extraction...")
        state_features = rag_scheduler.base_scheduler._extract_simple_features(state)
        print(f"✓ State features: shape={state_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ RAG component test failed: {e}")
        traceback.print_exc()
        return False

def test_full_decision():
    """测试完整的决策过程"""
    print("\n=== Testing Full Decision Process ===")
    
    try:
        rag_scheduler = create_scheduler(
            "WASS-RAG",
            model_path="models/wass_models.pth",
            knowledge_base_path="data/knowledge_base.pkl"
        )
        
        state = create_simple_test_state()
        
        print("🔍 Making full decision...")
        action = rag_scheduler.make_decision(state)
        
        print(f"✓ Decision made successfully:")
        print(f"  Selected node: {action.target_node}")
        print(f"  Confidence: {action.confidence:.3f}")
        print(f"  Reasoning: {action.reasoning}")
        
        return True
        
    except Exception as e:
        print(f"✗ Full decision test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 RAG Scheduler Debug Test")
    print("=" * 60)
    
    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Graph Construction", test_graph_construction), 
        ("RAG Components", test_rag_components),
        ("Full Decision", test_full_decision)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*60}")
    print("📊 Test Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print(f"\n🎉 All tests passed! RAG scheduler should work correctly.")
    else:
        print(f"\n⚠️ Some tests failed. Check the output above for details.")
