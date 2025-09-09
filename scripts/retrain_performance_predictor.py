#!/usr/bin/env python3
"""
[最终修复版] 重新训练性能预测器模型，修复负值预测和类型错误问题
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
    from src.ai_schedulers import PerformancePredictor, RAGKnowledgeBase, WASSRAGScheduler, SchedulingState
    HAS_AI_MODULES = True
except ImportError as e:
    print(f"Error: Required AI modules not available: {e}")
    sys.exit(1)

def create_improved_training_data(num_scenarios: int = 5000) -> List[Dict[str, Any]]:
    """
    [最终修复版] 生成高质量的、与推理路径完全一致的合成训练数据。
    """
    print(f"🔧 Generating {num_scenarios} scenarios for training data...")
    
    temp_scheduler = WASSRAGScheduler()
    temp_kb = RAGKnowledgeBase()
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
            "id": f"task_{i}", "flops": float(np.random.uniform(0.5e9, 20e9)),
            "memory": round(np.random.uniform(1.0, 16.0), 2),
            "dependencies": [f"task_{k}" for k in range(np.random.randint(0, 4))]
        }

        state = SchedulingState(
            workflow_graph={"tasks": [task_info]},
            cluster_state={"nodes": nodes}, pending_tasks=[f"task_{i}"], current_task=f"task_{i}",
            available_nodes=list(nodes.keys()), timestamp=0.0
        )

        state_embedding = temp_scheduler._extract_simple_features_fallback(state)
        retrieved_context = temp_kb.retrieve_similar_cases(state_embedding.cpu().numpy())
        context_embedding = temp_scheduler._encode_context(retrieved_context)

        for node_name, node_details in nodes.items():
            action_embedding = temp_scheduler._encode_action(node_name, state)
            combined_features = torch.cat([
                state_embedding, action_embedding, context_embedding
            ]).cpu().numpy()
            
            task_cpu_gflops = task_info["flops"] / 1e9
            available_cpu = node_details["cpu_capacity"] * (1.0 - node_details["current_load"])
            
            base_time = task_cpu_gflops / max(available_cpu, 0.1)
            mem_penalty = max(0, task_info["memory"] - node_details["memory_capacity"]) * 2.0
            
            task_ratio = task_cpu_gflops / max(1.0, task_info["memory"])
            node_ratio = available_cpu / max(1.0, node_details["memory_capacity"])
            mismatch_penalty = abs(task_ratio - node_ratio) * 0.5

            random_noise = np.random.uniform(0.95, 1.05)
            execution_time = (base_time + mem_penalty + mismatch_penalty) * random_noise
            execution_time = max(0.1, min(200.0, execution_time))
            makespan_values.append(execution_time)
            
            new_case = {
                "features": combined_features.tolist(),
                "makespan": execution_time,
                "state_embedding": state_embedding.cpu().numpy().tolist(), # 存储为list
                "workflow_features": {"task_count": 1}
            }
            training_data.append(new_case)
            
            # --- 最终修正：在调用 add_case 前，将 list 转换回 numpy array ---
            if i % 10 == 0:
                embedding_array = np.array(new_case["state_embedding"], dtype=np.float32)
                temp_kb.add_case(embedding_array, new_case["workflow_features"], [], new_case["makespan"])
            
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
    
    X = np.array([sample["features"] for sample in training_data])
    y = np.array([sample["makespan"] for sample in training_data])
    
    y_mean, y_std = np.mean(y), np.std(y)
    y_normalized = (y - y_mean) / (y_std + 1e-8)
    
    print(f"📈 Training data statistics:")
    print(f"   Original y: mean={y_mean:.2f}, std={y_std:.2f}")
    
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y_normalized).view(-1, 1).to(device) # 修正形状以匹配模型输出
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PerformancePredictor(input_dim=96, hidden_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, verbose=True)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
        
        if patience_counter >= 40:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    model.eval()
    with torch.no_grad():
        all_predictions = model(X_tensor).squeeze().cpu().numpy()
        all_predictions_denorm = all_predictions * y_std + y_mean
        
        mse = np.mean((all_predictions_denorm - y) ** 2)
        mae = np.mean(np.abs(all_predictions_denorm - y))
        r2 = 1 - (np.sum((y - all_predictions_denorm) ** 2) / np.sum((y - y_mean) ** 2))
        
        print(f"\n✅ Training completed!")
        print(f"   Final Loss: {best_loss:.6f}")
        print(f"   MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    return model, y_mean, y_std, {"mse": mse, "mae": mae, "r2": r2}

def regenerate_knowledge_base(training_data: List[Dict[str, Any]]) -> RAGKnowledgeBase:
    """根据新的训练数据重新生成知识库"""
    print(f"\n🔄 Regenerating knowledge base with {len(training_data)} cases...")
    
    kb = RAGKnowledgeBase(embedding_dim=32)
    
    for data in training_data:
        embedding = np.array(data["state_embedding"], dtype=np.float32)
        workflow_info = data.get("workflow_features", {"type": "retrained_synthetic"})
        makespan = data["makespan"]
        kb.add_case(embedding, workflow_info, actions=[], makespan=makespan)
    
    print(f"✅ Knowledge base regenerated with {len(kb.cases)} cases")
    return kb

def main():
    print("🔧 Retraining Performance Predictor with Improved Data (Final Version)")
    print("=" * 60)
    
    training_data = create_improved_training_data(num_scenarios=5000)
    model, y_mean, y_std, metrics = train_improved_performance_predictor(training_data)
    kb = regenerate_knowledge_base(training_data)
    
    model_path = "models/wass_models.pth"
    print(f"\n💾 Saving retrained model to {model_path}...")
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except FileNotFoundError:
        checkpoint = {}
    
    checkpoint["performance_predictor"] = model.state_dict()
    
    if "policy_network" not in checkpoint:
        policy_net = PolicyNetwork(state_dim=64, hidden_dim=128)
        checkpoint["policy_network"] = policy_net.state_dict()
    
    if "gnn_encoder" not in checkpoint and HAS_AI_MODULES:
        gnn_encoder = GraphEncoder(node_feature_dim=8, edge_feature_dim=4, hidden_dim=64, output_dim=32)
        checkpoint["gnn_encoder"] = gnn_encoder.state_dict()
    
    checkpoint["metadata"] = {
        "performance_predictor": {
            "y_mean": float(y_mean), "y_std": float(y_std),
            "training_samples": len(training_data), "retrained_at": datetime.now().isoformat(),
            "validation_results": metrics
        }
    }
    
    os.makedirs("models", exist_ok=True)
    torch.save(checkpoint, model_path)
    
    kb_path = "data/knowledge_base.pkl"
    print(f"\n💾 Saving regenerated knowledge base to {kb_path}...")
    os.makedirs("data", exist_ok=True)
    kb.save_knowledge_base(kb_path)
    
    print(f"\n✅ Model and knowledge base retrained and saved successfully!")
    print(f"   New normalization: mean={y_mean:.2f}, std={y_std:.2f}")
    print(f"   Performance metrics: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.2f}")

if __name__ == "__main__":
    main()