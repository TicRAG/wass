#!/usr/bin/env python3
"""
测试修复后的RAG系统
验证模型和知识库的一致性
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    from src.ai_schedulers import WASSRAGScheduler, PerformancePredictor, RAGKnowledgeBase
    HAS_AI_MODULES = True
except ImportError as e:
    print(f"Error: Required AI modules not available: {e}")
    sys.exit(1)

def test_performance_predictor():
    """测试性能预测器的预测质量"""
    print("🔍 Testing Performance Predictor...")
    
    # 加载模型
    try:
        checkpoint = torch.load("models/wass_models.pth", map_location="cpu")
        model = PerformancePredictor(input_dim=96, hidden_dim=128)
        model.load_state_dict(checkpoint["performance_predictor"])
        model.eval()
        
        # 加载归一化参数
        metadata = checkpoint.get("metadata", {}).get("performance_predictor", {})
        y_mean = metadata.get("y_mean", 0.0)
        y_std = metadata.get("y_std", 1.0)
        
        print(f"✅ Model loaded successfully")
        print(f"   Normalization: mean={y_mean:.2f}, std={y_std:.2f}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # 测试预测
    print("\n🧪 Testing predictions...")
    
    # 生成测试样本
    test_samples = []
    expected_ranges = []
    
    for i in range(10):
        # 生成合理的特征向量
        state_features = np.random.uniform(0.1, 0.9, 32)  # 状态嵌入
        action_features = np.random.uniform(0.1, 0.9, 32)  # 动作嵌入
        context_features = np.random.uniform(0.1, 0.9, 32)  # 上下文嵌入
        
        features = np.concatenate([state_features, action_features, context_features])
        test_samples.append(features)
        
        # 根据特征预估合理的makespan范围
        task_complexity = state_features[1]  # 假设这代表任务复杂度
        node_efficiency = action_features[1]  # 假设这代表节点效率
        expected_makespan = (0.5 + task_complexity * 5) / (0.1 + node_efficiency)
        expected_ranges.append(expected_makespan)
    
    # 进行预测
    predictions = []
    with torch.no_grad():
        for features in test_samples:
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            pred_normalized = model(input_tensor).item()
            pred_denormalized = pred_normalized * y_std + y_mean
            predictions.append(pred_denormalized)
    
    # 分析预测结果
    predictions = np.array(predictions)
    expected_ranges = np.array(expected_ranges)
    
    negative_count = np.sum(predictions < 0)
    reasonable_count = np.sum((predictions >= 0.1) & (predictions <= 100))
    
    print(f"   Predictions range: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]")
    print(f"   Negative predictions: {negative_count}/10")
    print(f"   Reasonable predictions (0.1-100s): {reasonable_count}/10")
    print(f"   Prediction std: {np.std(predictions):.2f}")
    
    if negative_count == 0 and reasonable_count >= 8:
        print("✅ Performance predictor looks healthy!")
        return True
    else:
        print("⚠️  Performance predictor may have issues")
        return False

def test_knowledge_base():
    """测试知识库的质量"""
    print("\n🔍 Testing Knowledge Base...")
    
    try:
        kb = RAGKnowledgeBase(embedding_dim=32)
        kb.load_knowledge_base("data/knowledge_base.pkl")
        
        print(f"✅ Knowledge base loaded successfully")
        print(f"   Total cases: {len(kb.cases)}")
        
        # 分析知识库中的makespan分布
        makespans = [case['performance'] for case in kb.cases]
        makespans = np.array(makespans)
        
        print(f"   Makespan range: [{np.min(makespans):.2f}, {np.max(makespans):.2f}]")
        print(f"   Makespan mean: {np.mean(makespans):.2f}")
        print(f"   Makespan std: {np.std(makespans):.2f}")
        
        negative_kb_count = np.sum(makespans < 0)
        reasonable_kb_count = np.sum((makespans >= 0.1) & (makespans <= 500))
        
        print(f"   Negative makespans: {negative_kb_count}/{len(makespans)}")
        print(f"   Reasonable makespans: {reasonable_kb_count}/{len(makespans)}")
        
        if negative_kb_count == 0 and reasonable_kb_count >= len(makespans) * 0.9:
            print("✅ Knowledge base looks healthy!")
            return True
        else:
            print("⚠️  Knowledge base may have issues")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load knowledge base: {e}")
        return False

def test_rag_scheduler():
    """测试完整的RAG调度器"""
    print("\n🔍 Testing RAG Scheduler...")
    
    try:
        scheduler = WASSRAGScheduler()
        print("✅ RAG scheduler initialized successfully")
        
        # 创建测试工作流
        workflow = {
            'tasks': [{'id': f'task_{i}', 'flops': 1e9, 'memory': 1.0} for i in range(5)],
            'dependencies': []
        }
        
        # 创建测试集群
        cluster = {
            'nodes': [
                {'id': f'node_{i}', 'cpu': 16.0, 'memory': 32.0, 'current_load': 0.1} 
                for i in range(3)
            ]
        }
        
        print("🧪 Testing scheduling decision...")
        
        # 测试调度决策
        task = workflow['tasks'][0]
        decision = scheduler.schedule_task(task, cluster, workflow)
        
        print(f"   Decision: {decision}")
        
        if decision and 'node_id' in decision:
            print("✅ RAG scheduler produced valid decision!")
            return True
        else:
            print("⚠️  RAG scheduler decision may be invalid")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test RAG scheduler: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 Testing Fixed RAG System")
    print("=" * 50)
    
    # 测试各个组件
    predictor_ok = test_performance_predictor()
    kb_ok = test_knowledge_base()
    rag_ok = test_rag_scheduler()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Performance Predictor: {'✅ PASS' if predictor_ok else '❌ FAIL'}")
    print(f"   Knowledge Base: {'✅ PASS' if kb_ok else '❌ FAIL'}")
    print(f"   RAG Scheduler: {'✅ PASS' if rag_ok else '❌ FAIL'}")
    
    if predictor_ok and kb_ok and rag_ok:
        print("\n🎉 All tests passed! The system should work correctly now.")
        print("   You can run: python experiments/real_experiment_framework.py")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    return predictor_ok and kb_ok and rag_ok

if __name__ == "__main__":
    main()
