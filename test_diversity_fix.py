#!/usr/bin/env python3
"""
测试WASS-RAG决策多样性修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai_schedulers import WASSRAGScheduler, SchedulingState, NodeInfo
import numpy as np
from collections import Counter

def test_wass_rag_diversity():
    """测试WASS-RAG决策多样性"""
    
    print("🧪 测试WASS-RAG决策多样性修复...")
    
    # 初始化调度器
    scheduler = WASSRAGScheduler()
    
    # 创建测试环境
    nodes = [
        NodeInfo(id="node_0", cpu_cores=8, memory_gb=16, available_cpu=6, available_memory=12),
        NodeInfo(id="node_1", cpu_cores=8, memory_gb=16, available_cpu=7, available_memory=14),
        NodeInfo(id="node_2", cpu_cores=8, memory_gb=16, available_cpu=5, available_memory=10),
        NodeInfo(id="node_3", cpu_cores=8, memory_gb=16, available_cpu=8, available_memory=15),
    ]
    
    # 测试决策多样性
    decisions = []
    confidences = []
    
    print("\n📊 测试20个不同任务的调度决策...")
    
    for i in range(20):
        state = SchedulingState(
            current_task=f"task_{i}",
            available_nodes=nodes,
            task_queue=[f"task_{j}" for j in range(i+1, i+5)],
            load_balance=0.5,
            system_health=0.9
        )
        
        action = scheduler.make_decision(state)
        decisions.append(action.target_node)
        confidences.append(action.confidence)
        
        print(f"Task {i:2d}: {action.target_node} (confidence: {action.confidence:.3f})")
    
    # 分析结果
    print("\n📈 决策多样性分析:")
    node_counts = Counter(decisions)
    total_decisions = len(decisions)
    
    for node, count in sorted(node_counts.items()):
        percentage = (count / total_decisions) * 100
        print(f"  {node}: {count:2d} 次 ({percentage:5.1f}%)")
    
    # 评估多样性
    unique_nodes = len(node_counts)
    max_count = max(node_counts.values())
    balance_score = 1 - (max_count / total_decisions)
    
    print(f"\n🎯 多样性指标:")
    print(f"  使用的不同节点数: {unique_nodes}/4")
    print(f"  平衡性评分: {balance_score:.3f} (越接近1越均衡)")
    
    # 置信度分析
    conf_mean = np.mean(confidences)
    conf_std = np.std(confidences)
    print(f"  平均置信度: {conf_mean:.3f} ± {conf_std:.3f}")
    
    # 判断修复是否成功
    success = unique_nodes >= 3 and balance_score > 0.3 and conf_mean < 0.6
    
    if success:
        print("\n✅ 修复成功！WASS-RAG现在展现良好的决策多样性")
    else:
        print("\n❌ 修复效果有限，需要进一步调整")
        
    return success

def test_degradation_logging():
    """测试降级日志输出"""
    
    print("\n🔍 测试降级日志输出...")
    
    # 该测试应该触发降级日志
    scheduler = WASSRAGScheduler()
    
    nodes = [
        NodeInfo(id="node_0", cpu_cores=4, memory_gb=8, available_cpu=2, available_memory=4),
        NodeInfo(id="node_1", cpu_cores=4, memory_gb=8, available_cpu=3, available_memory=6),
    ]
    
    state = SchedulingState(
        current_task="test_task",
        available_nodes=nodes,
        task_queue=["task_1", "task_2"],
        load_balance=0.7,
        system_health=0.8
    )
    
    print("期望看到降级日志（Performance predictor appears untrained）...")
    action = scheduler.make_decision(state)
    print(f"Decision: {action.target_node} (confidence: {action.confidence:.3f})")

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 WASS-RAG决策多样性修复验证")
    print("=" * 60)
    
    try:
        # 测试决策多样性
        diversity_success = test_wass_rag_diversity()
        
        # 测试降级日志
        test_degradation_logging()
        
        print("\n" + "=" * 60)
        if diversity_success:
            print("🎉 整体修复验证成功！")
        else:
            print("⚠️  修复部分成功，但仍需优化")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
