#!/usr/bin/env python3
"""
测试降级日志输出
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_degradation_logging():
    """测试降级日志的输出"""
    
    try:
        from src.ai_schedulers import WASSSmartScheduler, WASSRAGScheduler
        from experiments.real_experiment_framework import create_mock_state
        
        print("=== 降级日志测试 ===")
        
        # 创建有问题的模拟状态来触发降级
        print("\n1. 测试正常情况（不应该有降级）...")
        
        smart_scheduler = WASSSmartScheduler("models/wass_models.pth")
        mock_state = create_mock_state(
            task_count=5,
            node_count=4,
            current_task="task_0"
        )
        
        action = smart_scheduler.make_decision(mock_state)
        print(f"   决策结果: {action.task_id} -> {action.target_node}")
        print(f"   推理信息: {action.reasoning}")
        
        # 检查是否有降级标记
        if "DEGRADED" in action.reasoning:
            print("   ⚠️  检测到降级标记!")
        else:
            print("   ✓ 正常决策，无降级")
            
        print("\n2. 测试RAG调度器...")
        
        rag_scheduler = WASSRAGScheduler(
            model_path="models/wass_models.pth",
            knowledge_base_path="data/wass_knowledge_base.pkl"
        )
        
        action = rag_scheduler.make_decision(mock_state)
        print(f"   决策结果: {action.task_id} -> {action.target_node}")
        print(f"   推理信息: {action.reasoning}")
        
        # 检查是否有降级标记
        if "DEGRADED" in action.reasoning:
            print("   ⚠️  检测到降级标记!")
        else:
            print("   ✓ 正常决策，无降级")
        
        print("\n3. 测试多个任务（监控降级模式）...")
        
        degradation_count = 0
        total_decisions = 0
        
        for i in range(5):
            test_state = create_mock_state(
                task_count=10,
                node_count=4,
                current_task=f"task_{i}"
            )
            
            smart_action = smart_scheduler.make_decision(test_state)
            rag_action = rag_scheduler.make_decision(test_state)
            
            total_decisions += 2
            
            if "DEGRADED" in smart_action.reasoning:
                degradation_count += 1
                print(f"   任务{i} DRL降级: {smart_action.reasoning}")
                
            if "DEGRADED" in rag_action.reasoning:
                degradation_count += 1
                print(f"   任务{i} RAG降级: {rag_action.reasoning}")
        
        print(f"\n降级统计:")
        print(f"   总决策数: {total_decisions}")
        print(f"   降级次数: {degradation_count}")
        print(f"   降级率: {degradation_count/total_decisions*100:.1f}%")
        
        if degradation_count == 0:
            print("   ✓ 所有决策都正常，无降级发生")
        else:
            print("   ⚠️  检测到降级，请检查日志详情")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_degradation_logging()
    
    if success:
        print("\n🎉 降级日志测试完成!")
        print("现在实验中的任何降级都会有明显的⚠️标记")
    else:
        print("\n❌ 仍有问题需要解决")
