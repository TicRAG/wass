#!/usr/bin/env python3
"""
重新训练性能预测器模型，修复负值预测问题
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import List, Dict, Any

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    from torch.utils.data import TensorDataset, DataLoader
    from src.ai_schedulers import PerformancePredictor, RAGKnowledgeBase
    HAS_AI_MODULES = True
except ImportError as e:
    print(f"Error: Required AI modules not available: {e}")
    sys.exit(1)

def create_improved_training_data(num_scenarios: int = 5000) -> List[Dict[str, Any]]:
    """
    生成高质量的合成训练数据（V5 - 最终修复版）
    确保特征生成逻辑一致，并为知识库保留必要字段。
    """
    print(f"🔧 Generating {num_scenarios} scenarios for training data...")
    from src.ai_schedulers import WASSRAGScheduler, SchedulingState
    temp_scheduler = WASSRAGScheduler()
    training_data = []
    makespan_values = []

    for i in range(num_scenarios):
        num_nodes = np.random.randint(2, 21)
        nodes = {f"node_{j}": {
            "cpu_capacity": round(np.random.uniform(2.0, 8.0), 2),
            "memory_capacity": round(np.random.uniform(8.0, 64.0), 2),
            "current_load": round(np.random.uniform(0.1, 0.9), 2),
        } for j in range(num_nodes)}

        task_info = {
            "id": f"task_{i}", "flops": float(np.random.uniform(0.5e9, 15e9)),
            "memory": round(np.random.uniform(1.0, 16.0), 2),
            "dependencies": [f"task_{k}" for k in range(np.random.randint(0, 4))]
        }

        state = SchedulingState(
            workflow_graph={"tasks": [task_info], "task_requirements": {f"task_{i}": task_info}},
            cluster_state={"nodes": nodes}, pending_tasks=[f"task_{i}"], current_task=f"task_{i}",
            available_nodes=list(nodes.keys()), timestamp=0.0
        )

        state_embedding = temp_scheduler._extract_simple_features_fallback(state)
        context_embedding = torch.randn(32, device=temp_scheduler.device)

        for node_name, node_details in nodes.items():
            action_embedding = temp_scheduler._encode_action(node_name, state)
            combined_features = torch.cat([
                state_embedding, action_embedding, context_embedding
            ]).cpu().numpy()
            
            task_cpu_gflops = task_info["flops"] / 1e9
            available_cpu = node_details["cpu_capacity"] * (1.0 - node_details["current_load"])
            base_time = task_cpu_gflops / max(available_cpu, 0.1)
            mem_penalty = max(0, task_info["memory"] - node_details["memory_capacity"]) * 0.5
            load_penalty = node_details["current_load"] * 2.0
            random_noise = np.random.uniform(-0.5, 0.5)
            execution_time = max(1.0, min(180.0, base_time + mem_penalty + load_penalty + random_noise))
            makespan_values.append(execution_time)
            
            # --- 关键修改：将 state_embedding 和其他元数据也存起来 ---
            training_data.append({
                "features": combined_features.tolist(),
                "makespan": execution_time,
                "state_embedding": state_embedding.cpu().numpy().tolist(), # 添加 state_embedding
                "workflow_features": { # 添加知识库需要的元数据
                    "task_count": 1,
                    "avg_task_flops": task_info["flops"],
                    "avg_memory": task_info["memory"]
                }
            })
            
    makespan_array = np.array(makespan_values)
    print(f"📊 Single task execution time distribution:")
    print(f"   Mean: {np.mean(makespan_array):.2f}s, Std: {np.std(makespan_array):.2f}s, "
          f"Min: {np.min(makespan_array):.2f}s, Max: {np.max(makespan_array):.2f}s")
    
    return training_data

def train_improved_performance_predictor(training_data: List[Dict[str, Any]], epochs: int = 200, batch_size: int = 64):
    """训练改进的性能预测器"""
    
    print(f"🚀 Training improved performance predictor...")
    print(f"   Training samples: {len(training_data)}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    # 准备训练数据
    X = np.array([sample["features"] for sample in training_data])
    y = np.array([sample["makespan"] for sample in training_data])
    
    # 数据归一化（重要！）
    y_mean, y_std = np.mean(y), np.std(y)
    y_normalized = (y - y_mean) / (y_std + 1e-8)
    
    print(f"📈 Training data statistics:")
    print(f"   Original y: mean={y_mean:.2f}, std={y_std:.2f}")
    print(f"   Normalized y: mean={np.mean(y_normalized):.6f}, std={np.std(y_normalized):.6f}")
    print(f"   Normalized range: [{np.min(y_normalized):.3f}, {np.max(y_normalized):.3f}]")
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y_normalized).to(device)
    
    # 创建数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = PerformancePredictor(input_dim=96, hidden_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
        
        if patience_counter >= 50:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        all_predictions = model(X_tensor).squeeze().cpu().numpy()
        all_predictions_denorm = all_predictions * y_std + y_mean
        
        mse = np.mean((all_predictions_denorm - y) ** 2)
        mae = np.mean(np.abs(all_predictions_denorm - y))
        r2 = 1 - np.sum((y - all_predictions_denorm) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        pred_std = np.std(all_predictions_denorm)
        pred_range = np.max(all_predictions_denorm) - np.min(all_predictions_denorm)
        
        print(f"\n✅ Training completed!")
        print(f"   Final Loss: {best_loss:.6f}")
        print(f"   MSE: {mse:.2f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   R²: {r2:.4f}")
        print(f"   Prediction std: {pred_std:.2f}")
        print(f"   Prediction range: {pred_range:.2f}")
    
    return model, y_mean, y_std, {
        "mse": mse, "mae": mae, "r2": r2,
        "pred_std": pred_std, "pred_range": pred_range
    }

def regenerate_knowledge_base(training_data: List[Dict[str, Any]]) -> RAGKnowledgeBase:
    """根据新的训练数据重新生成知识库（V2 - 修复版）"""
    print(f"\n🔄 Regenerating knowledge base with {len(training_data)} cases...")
    
    kb = RAGKnowledgeBase(embedding_dim=32)
    
    for data in training_data:
        # --- 关键修改：现在可以安全地读取 "state_embedding" ---
        embedding = np.array(data["state_embedding"], dtype=np.float32)
        
        # 构建一个简化的 workflow_info
        workflow_info = data.get("workflow_features", {"type": "retrained_synthetic"})
        
        # 使用实际的makespan
        makespan = data["makespan"]
        
        # 添加到知识库
        kb.add_case(embedding, workflow_info, actions=[], makespan=makespan)
    
    print(f"✅ Knowledge base regenerated with {len(kb.cases)} cases")
    return kb

def main():
    print("🔧 Retraining Performance Predictor with Improved Data")
    print("=" * 60)
    
    # 生成改进的训练数据
    # --- 请修改下面这一行 ---
    # 将 num_samples 修改为 num_scenarios
    training_data = create_improved_training_data(num_scenarios=5000)
    
    # 训练模型
    model, y_mean, y_std, metrics = train_improved_performance_predictor(training_data)
    
    # 重新生成知识库（使用相同的训练数据）
    kb = regenerate_knowledge_base(training_data)
    
    # 保存模型
    model_path = "models/wass_models.pth"
    print(f"\n💾 Saving retrained model to {model_path}...")
    
    # 加载现有checkpoint（如果存在）
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        print("   Loaded existing checkpoint")
    except:
        checkpoint = {}
        print("   Creating new checkpoint")
    
    # 更新性能预测器（保留其他组件）
    checkpoint["performance_predictor"] = model.state_dict()
    
    # 确保其他必要组件存在，如果不存在则创建默认值
    if "policy_network" not in checkpoint:
        from src.ai_schedulers import PolicyNetwork
        checkpoint["policy_network"] = PolicyNetwork(
            state_dim=32, action_dim=1, hidden_dim=128
        ).state_dict()
        print("   Added default PolicyNetwork")
    
    if "gnn_encoder" not in checkpoint:
        try:
            from src.ai_schedulers import GraphEncoder
            checkpoint["gnn_encoder"] = GraphEncoder(
                node_feature_dim=8, edge_feature_dim=4, 
                hidden_dim=64, output_dim=32
            ).state_dict()
            print("   Added default GraphEncoder")
        except Exception as e:
            print(f"   Skipping GraphEncoder: {e}")
    
    # 更新元数据
    if "metadata" not in checkpoint:
        checkpoint["metadata"] = {}
    
    checkpoint["metadata"]["performance_predictor"] = {
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "training_samples": len(training_data),
        "retrained_at": "2025-09-08",
        "validation_results": metrics
    }
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(checkpoint, model_path)
    
    # 保存知识库
    kb_path = "data/knowledge_base.pkl"
    print(f"\n💾 Saving regenerated knowledge base to {kb_path}...")
    os.makedirs("data", exist_ok=True)
    kb.save_knowledge_base(kb_path)
    
    print(f"✅ Model and knowledge base retrained and saved successfully!")
    print(f"   New normalization: mean={y_mean:.2f}, std={y_std:.2f}")
    print(f"   Performance metrics: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.2f}")
    print(f"   Knowledge base cases: {len(kb.cases)}")
    print(f"\n🎉 Ready for testing! Run: python experiments/real_experiment_framework.py")

if __name__ == "__main__":
    main()
