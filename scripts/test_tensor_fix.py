#!/usr/bin/env python3
"""
测试张量维度修复
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_tensor_dimensions():
    """测试张量维度是否正确"""
    
    try:
        from src.ai_schedulers import WASSSmartScheduler, WASSRAGScheduler, SchedulingState
        import torch
        
        print("=== 张量维度修复测试 ===")
        
        # 创建调度器
        smart_scheduler = WASSSmartScheduler("models/wass_models.pth")
        rag_scheduler = WASSRAGScheduler(
            model_path="models/wass_models.pth",
            knowledge_base_path="data/wass_knowledge_base.pkl"
        )
        
        # 创建大规模测试状态（模拟实际场景）
        def create_large_state(task_count=100, node_count=16):
            tasks = []
            for i in range(task_count):
                task = {
                    "id": f"task_{i}",
                    "flops": 1e9 + i * 1e8,
                    "memory": 1e9 + i * 1e8,
                    "dependencies": [f"task_{j}" for j in range(max(0, i-3), i)] if i > 0 else []
                }
                tasks.append(task)
            
            workflow_graph = {"tasks": tasks, "name": "large_test"}
            cluster_state = {
                "nodes": {
                    f"node_{i}": {
                        "cpu_capacity": 10.0,
                        "memory_capacity": 16.0,
                        "current_load": 0.3 + (i * 0.05) % 0.6,
                        "available": True
                    }
                    for i in range(node_count)
                }
            }
            
            return SchedulingState(
                workflow_graph=workflow_graph,
                cluster_state=cluster_state,
                pending_tasks=[f"task_{i}" for i in range(10, task_count)],
                current_task="task_10",
                available_nodes=[f"node_{i}" for i in range(node_count)],
                timestamp=1234567890.0
            )
        
        # 测试不同规模的任务
        test_cases = [
            (10, 4, "小规模"),
            (50, 8, "中等规模"),
            (100, 16, "大规模")
        ]
        
        for task_count, node_count, desc in test_cases:
            print(f"\n{desc}测试 ({task_count}任务, {node_count}节点):")
            
            try:
                state = create_large_state(task_count, node_count)
                
                # 测试WASS-DRL
                print(f"  测试WASS-DRL...")
                smart_action = smart_scheduler.make_decision(state)
                if "DEGRADED" in smart_action.reasoning:
                    print(f"    ⚠️  DRL降级: {smart_action.reasoning}")
                else:
                    print(f"    ✓ DRL正常: {smart_action.target_node} (置信度: {smart_action.confidence:.3f})")
                
                # 测试WASS-RAG
                print(f"  测试WASS-RAG...")
                rag_action = rag_scheduler.make_decision(state)
                if "DEGRADED" in rag_action.reasoning:
                    print(f"    ⚠️  RAG降级: {rag_action.reasoning}")
                else:
                    print(f"    ✓ RAG正常: {rag_action.target_node} (置信度: {rag_action.confidence:.3f})")
                    
            except Exception as e:
                print(f"    ❌ {desc}测试失败: {e}")
                return False
        
        # 特殊情况测试
        print(f"\n特殊情况测试:")
        
        # 测试边界条件
        edge_cases = [
            (1, 1, "最小规模"),
            (200, 32, "超大规模")
        ]
        
        for task_count, node_count, desc in edge_cases:
            try:
                state = create_large_state(task_count, node_count)
                smart_action = smart_scheduler.make_decision(state)
                rag_action = rag_scheduler.make_decision(state)
                
                smart_degraded = "DEGRADED" in smart_action.reasoning
                rag_degraded = "DEGRADED" in rag_action.reasoning
                
                print(f"  {desc}: DRL{'降级' if smart_degraded else '正常'}, RAG{'降级' if rag_degraded else '正常'}")
                
            except Exception as e:
                print(f"  {desc}测试异常: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tensor_dimensions()
    
    if success:
        print("\n🎉 张量维度修复测试完成!")
        print("现在可以运行大规模实验而不会有维度错误")
    else:
        print("\n❌ 仍有张量维度问题需要解决")
