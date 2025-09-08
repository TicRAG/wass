#!/usr/bin/env python3
"""
诊断特征匹配问题
"""

import sys
import os
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from ai_schedulers import WASSRAGScheduler, SchedulingState
    
    # 创建调度器
    scheduler = WASSRAGScheduler("models/wass_models.pth", "data/knowledge_base.pkl")
    
    # 创建测试状态
    test_workflow = {
        "tasks": [
            {"id": "task_0", "flops": 5e9, "memory": 2.0}
        ]
    }
    
    test_cluster = {
        "nodes": {
            "node_0": {"cpu_capacity": 2.0, "memory_capacity": 16.0, "current_load": 0.3}
        }
    }
    
    state = SchedulingState(
        workflow_graph=test_workflow,
        cluster_state=test_cluster,
        pending_tasks=["task_0"],
        current_task="task_0",
        available_nodes=["node_0"],
        timestamp=0.0
    )
    
    print("🔍 Debugging feature encoding...")
    
    # 1. 检查状态编码
    if scheduler.base_scheduler:
        state_embedding = scheduler.base_scheduler._extract_simple_features(state)
    else:
        state_embedding = scheduler._extract_simple_features_fallback(state)
    
    print(f"State embedding shape: {state_embedding.shape}")
    print(f"State embedding range: [{state_embedding.min().item():.3f}, {state_embedding.max().item():.3f}]")
    print(f"State embedding: {state_embedding[:10].tolist()}")  # 前10个特征
    
    # 2. 检查动作编码
    action_embedding = scheduler._encode_action("node_0", state)
    print(f"Action embedding shape: {action_embedding.shape}")
    print(f"Action embedding range: [{action_embedding.min().item():.3f}, {action_embedding.max().item():.3f}]")
    print(f"Action embedding: {action_embedding[:10].tolist()}")
    
    # 3. 检查上下文编码
    retrieved_context = scheduler.knowledge_base.retrieve_similar_cases(
        state_embedding.cpu().numpy(), top_k=5
    )
    context_embedding = scheduler._encode_context(retrieved_context)
    print(f"Context embedding shape: {context_embedding.shape}")
    print(f"Context embedding range: [{context_embedding.min().item():.3f}, {context_embedding.max().item():.3f}]")
    print(f"Context embedding: {context_embedding[:10].tolist()}")
    
    # 4. 检查合并特征
    state_flat = state_embedding.flatten()[:32]
    action_flat = action_embedding.flatten()[:32]
    context_flat = context_embedding.flatten()[:32]
    
    # 填充到32维
    def pad_to_32(tensor):
        if len(tensor) < 32:
            padding = torch.zeros(32 - len(tensor), device=tensor.device)
            return torch.cat([tensor, padding])
        return tensor[:32]
    
    combined_features = torch.cat([
        pad_to_32(state_flat),
        pad_to_32(action_flat),
        pad_to_32(context_flat)
    ])
    
    print(f"Combined features shape: {combined_features.shape}")
    print(f"Combined features range: [{combined_features.min().item():.3f}, {combined_features.max().item():.3f}]")
    print(f"Combined features mean: {combined_features.mean().item():.3f}")
    print(f"Combined features std: {combined_features.std().item():.3f}")
    
    # 5. 检查模型预测
    with torch.no_grad():
        raw_prediction = scheduler.performance_predictor(combined_features).item()
        print(f"Raw model prediction: {raw_prediction:.3f}")
        
        if hasattr(scheduler, '_y_mean') and hasattr(scheduler, '_y_std'):
            denormalized = raw_prediction * scheduler._y_std + scheduler._y_mean
            print(f"Denormalized prediction: {denormalized:.3f}")
            print(f"Normalization params: mean={scheduler._y_mean:.3f}, std={scheduler._y_std:.3f}")
    
    # 6. 检查知识库中的案例样本
    print(f"\n📊 Knowledge base stats:")
    print(f"Total cases: {len(scheduler.knowledge_base.cases)}")
    if len(scheduler.knowledge_base.cases) > 0:
        first_case = scheduler.knowledge_base.cases[0]
        print(f"First case keys: {list(first_case.keys())}")
        if 'makespan' in first_case:
            makespans = [case.get('makespan', 0) for case in scheduler.knowledge_base.cases[:100]]
            print(f"Sample makespans: mean={np.mean(makespans):.2f}, std={np.std(makespans):.2f}")
            print(f"Sample makespans range: [{np.min(makespans):.2f}, {np.max(makespans):.2f}]")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
