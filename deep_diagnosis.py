#!/usr/bin/env python3
"""
深度诊断脚本：检查训练数据与预测时特征分布的一致性
"""

import sys
import os
import torch
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from ai_schedulers import WASSRAGScheduler, SchedulingState
    
    def deep_diagnosis():
        """
        深度诊断特征分布问题
        """
        print("=== Deep Feature Distribution Diagnosis ===")

        # 1. 初始化调度器
        try:
            scheduler = WASSRAGScheduler("models/wass_models.pth", "data/knowledge_base.pkl")
            print("✓ Scheduler initialized with trained model")
        except Exception as e:
            print(f"✗ Failed to initialize scheduler: {e}")
            return

        # 2. 创建多个不同的测试场景
        scenarios = [
            {
                "name": "Easy Task",
                "task": {"id": "task_easy", "flops": 1e9, "memory": 2.0},
                "nodes": {
                    "node_0": {"cpu_capacity": 2.0, "memory_capacity": 16.0, "current_load": 0.2},
                    "node_1": {"cpu_capacity": 4.0, "memory_capacity": 32.0, "current_load": 0.1},
                }
            },
            {
                "name": "Hard Task", 
                "task": {"id": "task_hard", "flops": 8e9, "memory": 12.0},
                "nodes": {
                    "node_0": {"cpu_capacity": 2.0, "memory_capacity": 16.0, "current_load": 0.8},
                    "node_1": {"cpu_capacity": 4.0, "memory_capacity": 32.0, "current_load": 0.3},
                }
            },
            {
                "name": "Balanced Task",
                "task": {"id": "task_balanced", "flops": 3e9, "memory": 6.0},
                "nodes": {
                    "node_0": {"cpu_capacity": 3.0, "memory_capacity": 24.0, "current_load": 0.4},
                    "node_1": {"cpu_capacity": 2.5, "memory_capacity": 20.0, "current_load": 0.6},
                }
            }
        ]

        for scenario in scenarios:
            print(f"\n--- Scenario: {scenario['name']} ---")
            
            state = SchedulingState(
                workflow_graph={"tasks": [scenario["task"]]},
                cluster_state={"nodes": scenario["nodes"]},
                pending_tasks=[scenario["task"]["id"]],
                current_task=scenario["task"]["id"],
                available_nodes=list(scenario["nodes"].keys()),
                timestamp=0.0
            )
            
            print(f"Task: {scenario['task']['flops']/1e9:.1f} GFlops, {scenario['task']['memory']:.1f} GB")
            
            # 对每个节点进行详细分析
            for node_name in state.available_nodes:
                node_info = scenario["nodes"][node_name]
                print(f"\nAnalyzing {node_name}: CPU={node_info['cpu_capacity']:.1f}GF, Load={node_info['current_load']:.1f}")
                
                try:
                    # 1. 生成特征
                    if scheduler.base_scheduler:
                        state_embedding = scheduler.base_scheduler._extract_simple_features(state)
                    else:
                        state_embedding = scheduler._extract_simple_features_fallback(state)
                    
                    action_embedding = scheduler._encode_action(node_name, state)
                    
                    retrieved_context = scheduler.knowledge_base.retrieve_similar_cases(
                        state_embedding.cpu().numpy(), top_k=5
                    )
                    context_embedding = scheduler._encode_context(retrieved_context)
                    
                    # 2. 合并特征
                    state_flat = state_embedding.flatten()[:32]
                    action_flat = action_embedding.flatten()[:32]
                    context_flat = context_embedding.flatten()[:32]
                    
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
                    
                    # 3. 检查特征质量
                    print(f"  Action embedding key features:")
                    print(f"    CPU_fit: {action_embedding[0].item():.3f}")
                    print(f"    Mem_fit: {action_embedding[1].item():.3f}")
                    print(f"    Perf_match: {action_embedding[2].item():.3f}")
                    
                    print(f"  Combined features stats:")
                    print(f"    Range: [{combined_features.min().item():.3f}, {combined_features.max().item():.3f}]")
                    print(f"    Mean: {combined_features.mean().item():.3f}")
                    print(f"    Std: {combined_features.std().item():.3f}")
                    
                    # 4. 模型预测
                    with torch.no_grad():
                        raw_prediction = scheduler.performance_predictor(combined_features).item()
                        denormalized = raw_prediction * scheduler._y_std + scheduler._y_mean
                        
                        print(f"  Model prediction:")
                        print(f"    Raw (normalized): {raw_prediction:.3f}")
                        print(f"    Denormalized: {denormalized:.3f}s")
                        
                        # 5. 完整的预测流程
                        final_prediction = scheduler._predict_performance(
                            state_embedding, action_embedding, retrieved_context
                        )
                        print(f"    Final (after constraints): {final_prediction:.3f}s")
                
                except Exception as e:
                    print(f"  ✗ Error analyzing {node_name}: {e}")
                    import traceback
                    traceback.print_exc()

        # 3. 检查知识库样本
        print(f"\n--- Knowledge Base Analysis ---")
        if len(scheduler.knowledge_base.cases) > 0:
            sample_cases = scheduler.knowledge_base.cases[:10]
            makespans = [case.get('makespan', 0) for case in sample_cases]
            print(f"Sample makespans from KB: {[f'{x:.2f}' for x in makespans]}")
            
            # 检查第一个案例的特征
            if 'actions' in sample_cases[0]:
                actions = sample_cases[0]['actions']
                print(f"Sample action features: {actions[:10] if len(actions) > 10 else actions}")

    if __name__ == "__main__":
        deep_diagnosis()

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
