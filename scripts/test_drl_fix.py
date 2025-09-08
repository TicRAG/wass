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

def test_drl_schedulers():
    """测试DRL调度器的修复"""
    
    try:
        from src.ai_schedulers import WASSSmartScheduler, WASSRAGScheduler
        from experiments.real_experiment_framework import create_mock_state
        
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
