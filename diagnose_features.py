#!/usr/bin/env python3
"""
特征工程诊断脚本
用于检查action_embedding的差异性
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
    
    def diagnose_features():
        """
        一个简单的诊断脚本，用于检查action_embedding的差异性。
        """
        print("=== Feature Engineering Diagnosis ===")

        # 1. 初始化一个WASS-RAG调度器 (无需加载模型)
        try:
            scheduler = WASSRAGScheduler()
            print("✓ Scheduler initialized")
        except Exception as e:
            print(f"✗ Failed to initialize scheduler: {e}")
            return

        # 2. 创建一个模拟的调度状态
        state = SchedulingState(
            workflow_graph={
                "tasks": [
                    {"id": "task_1", "flops": 4e9, "memory": 8.0, "dependencies": ["task_0"]}
                ]
            },
            cluster_state={
                "nodes": {
                    "node_0": {"cpu_capacity": 2.0, "memory_capacity": 16.0, "current_load": 0.8},
                    "node_1": {"cpu_capacity": 4.0, "memory_capacity": 32.0, "current_load": 0.2},
                    "node_2": {"cpu_capacity": 3.0, "memory_capacity": 24.0, "current_load": 0.5},
                }
            },
            pending_tasks=["task_1"],
            current_task="task_1",
            available_nodes=["node_0", "node_1", "node_2"],
            timestamp=0.0
        )
        print("✓ Created mock scheduling state")
        print(f"\nScheduling Task: {state.current_task} (CPU req: 4.0 GFlops, Mem req: 8.0 GB)")
        print("Available Nodes:")
        for name, info in state.cluster_state["nodes"].items():
            cpu_cap = info['cpu_capacity']
            mem_cap = info['memory_capacity']
            load = info['current_load']
            available_cpu = cpu_cap * (1.0 - load)
            print(f"  - {name}: CPU={cpu_cap:.1f}GF (avail: {available_cpu:.1f}), Mem={mem_cap}GB, Load={load:.1f}")

        # 3. 为每个可用节点生成并打印action_embedding
        print("\n--- Generated Action Embeddings (first 14 features) ---")
        feature_names = [
            "CPU_fit", "Mem_fit", "Perf_match", "Data_locality", "Load_balance",
            "Node_ID", "Current_load", "Free_capacity", "CPU_cap_norm", "Mem_cap_norm",
            "Task_CPU_norm", "Task_Mem_norm", "Compute_intensity", "Available_nodes"
        ]
        
        embeddings = []
        for node in state.available_nodes:
            try:
                embedding = scheduler._encode_action(node, state)
                embeddings.append(embedding)
                # 打印前14个最重要的特征，以便比较
                feature_values = [f"{x:.3f}" for x in embedding[:14].tolist()]
                print(f"Node '{node}': {dict(zip(feature_names, feature_values))}")
            except Exception as e:
                print(f"✗ Error encoding action for node {node}: {e}")

        # 4. 检查差异性
        if not embeddings:
            print("\n✗ No embeddings generated.")
            return
            
        print("\n--- Analysis ---")
        try:
            # 检查所有向量是否都相同
            first_embedding = embeddings[0]
            all_same = True
            for emb in embeddings[1:]:
                if not torch.allclose(first_embedding, emb, atol=1e-6):
                    all_same = False
                    break
            
            if all_same:
                print("✗ CRITICAL: All action embeddings are identical! The model cannot learn.")
            else:
                print("✅ SUCCESS: Action embeddings are different for each node.")
                
            # 计算特征的标准差，看其变化幅度
            features_stacked = torch.stack(embeddings)
            stds = torch.std(features_stacked, dim=0)
            print(f"\nFeature standard deviations (higher = more variance):")
            for i, (name, std_val) in enumerate(zip(feature_names, stds[:14].tolist())):
                print(f"  {name:<15}: {std_val:.4f}")
            
            total_variance = torch.sum(stds).item()
            print(f"\nTotal variance: {total_variance:.4f}")
            if total_variance > 0.5:
                print("✅ EXCELLENT: Features show very high variance across nodes.")
            elif total_variance > 0.1:
                print("✅ SUCCESS: Features show significant variance across nodes.")
            else:
                print("⚠️ WARNING: Features show very little variance. The model may still struggle to learn.")

            # 检查关键的交互特征
            print(f"\n--- Key Interaction Features Analysis ---")
            cpu_fits = [emb[0].item() for emb in embeddings]
            mem_fits = [emb[1].item() for emb in embeddings]
            perf_matches = [emb[2].item() for emb in embeddings]
            
            print(f"CPU Fit scores: {[f'{x:.3f}' for x in cpu_fits]}")
            print(f"Mem Fit scores: {[f'{x:.3f}' for x in mem_fits]}")
            print(f"Perf Match scores: {[f'{x:.3f}' for x in perf_matches]}")
            
            if len(set([f"{x:.2f}" for x in cpu_fits])) > 1:
                print("✅ CPU fit varies across nodes - GOOD!")
            else:
                print("❌ CPU fit is same for all nodes - BAD!")

        except Exception as e:
            print(f"\n✗ Error during analysis: {e}")
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":
        diagnose_features()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the project root directory and all dependencies are installed.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
