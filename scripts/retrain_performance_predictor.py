#!/usr/bin/env python3
"""
[最终修复版 V2] 重新训练性能预测器模型，与时序感知的仿真框架完全对齐
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import List, Dict, Any
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    from torch.utils.data import TensorDataset, DataLoader
    from src.ai_schedulers import WASSRAGScheduler, SchedulingState, PerformancePredictor, RAGKnowledgeBase
    HAS_AI_MODULES = True
except ImportError as e:
    print(f"Error: Required AI modules not available: {e}")
    sys.exit(1)

def create_time_aware_training_data(num_scenarios: int = 5000) -> List[Dict[str, Any]]:
    """
    [最终版] 生成高质量的、与时序感知推理路径完全一致的合成训练数据。
    """
    print(f"🔧 Generating {num_scenarios} time-aware scenarios for training data...")
    
    # 我们需要一个调度器实例来调用其内部的编码方法
    temp_scheduler = WASSRAGScheduler()
    training_data = []
    
    for i in range(num_scenarios):
        # 1. 创建一个随机的场景
        num_nodes = np.random.randint(2, 21)
        nodes = {f"node_{j}": {
            "cpu_capacity": round(np.random.uniform(2.0, 8.0), 2),
            "memory_capacity": round(np.random.uniform(8.0, 64.0), 2),
            "current_load": round(np.random.uniform(0.1, 0.9), 2),
        } for j in range(num_nodes)}

        task_info = {
            "id": f"task_{i}", "flops": float(np.random.uniform(0.5e9, 20e9)),
            "memory": round(np.random.uniform(1.0, 16.0), 2),
            "dependencies": [] # 在这个微型场景中，我们只关心单任务决策
        }
        
        # 2. 模拟一个微型的调度状态，这是关键！
        current_time = np.random.uniform(0, 100) # 模拟一个随机的当前时间
        
        # 模拟随机的节点可用时间
        node_available_times = {name: current_time + np.random.uniform(0, 20) for name in nodes.keys()}
        
        # 模拟 earliest_start_times，这正是新特征所需要的
        earliest_start_times = {name: max(current_time, avail_time) for name, avail_time in node_available_times.items()}

        state = SchedulingState(
            workflow_graph={"tasks": [task_info]},
            cluster_state={"nodes": nodes, "earliest_start_times": earliest_start_times},
            pending_tasks=[f"task_{i}"],
            current_task=f"task_{i}",
            available_nodes=list(nodes.keys()),
            timestamp=current_time
        )

        # 3. 为场景中的每个可能的决策（将任务分配给每个节点）生成一条训练数据
        state_embedding = temp_scheduler._extract_simple_features_fallback(state)
        retrieved_context = temp_scheduler.knowledge_base.retrieve_similar_cases(state_embedding.cpu().numpy())
        context_embedding = temp_scheduler._encode_context(retrieved_context)

        for node_name in nodes.keys():
            # 使用时序感知编码器生成 action embedding
            action_embedding = temp_scheduler._encode_action(node_name, state)
            
            combined_features = torch.cat([
                state_embedding, action_embedding, context_embedding
            ]).cpu().numpy()
            
            # 4. 计算真实的目标值 (ground truth makespan)
            task_cpu_gflops = task_info["flops"] / 1e9
            node_cpu_gflops = nodes[node_name]["cpu_capacity"]
            
            # 真实执行时间
            execution_time = task_cpu_gflops / max(0.1, node_cpu_gflops)
            
            # 真实完成时间（完工时间）= 节点可用时间 + 执行时间
            finish_time = earliest_start_times[node_name] + execution_time
            
            # 添加随机噪声
            finish_time *= np.random.uniform(0.95, 1.05)

            training_data.append({
                "features": combined_features.tolist(),
                "makespan": max(0.1, finish_time), # 我们的模型预测的是完工时间
                "state_embedding": state_embedding.cpu().numpy().tolist()
            })
            
    print(f"📊 Generated {len(training_data)} training samples.")
    return training_data

# ... train_improved_performance_predictor 和 regenerate_knowledge_base 函数保持不变 ...
def train_improved_performance_predictor(training_data: List[Dict[str, Any]], epochs: int = 50, batch_size: int = 256):
    print(f"🚀 Training improved performance predictor...")
    # (此函数无需修改)
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
    print(f"   Original y (makespan): mean={y_mean:.2f}, std={y_std:.2f}")
    
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y_normalized).view(-1, 1).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PerformancePredictor(input_dim=96, hidden_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_loss = float('inf')
    
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

        if avg_loss < best_loss: best_loss = avg_loss
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
            
    model.eval()
    with torch.no_grad():
        all_predictions = model(X_tensor).squeeze().cpu().numpy()
        all_predictions_denorm = all_predictions * y_std + y_mean
        
        mse = np.mean((all_predictions_denorm - y) ** 2)
        r2 = 1 - (np.sum((y - all_predictions_denorm) ** 2) / np.sum((y - y_mean) ** 2))
        
        print(f"\n✅ Training completed! Final Loss: {best_loss:.6f}, R²: {r2:.4f}")
    
    return model, y_mean, y_std, {"r2": r2}

def regenerate_knowledge_base(training_data: List[Dict[str, Any]]) -> RAGKnowledgeBase:
    print(f"\n🔄 Regenerating knowledge base...")
    # (此函数无需修改)
    kb = RAGKnowledgeBase(embedding_dim=32)
    for data in training_data:
        embedding = np.array(data["state_embedding"], dtype=np.float32)
        kb.add_case(embedding, {}, [], data["makespan"])
    print(f"✅ Knowledge base regenerated with {len(kb.cases)} cases")
    return kb

def main():
    print("🔧 Retraining Performance Predictor with Time-Aware Data (Final Version)")
    print("=" * 60)
    
    training_data = create_time_aware_training_data(num_scenarios=5000)
    model, y_mean, y_std, metrics = train_improved_performance_predictor(training_data)
    kb = regenerate_knowledge_base(training_data)
    
    model_path = "models/wass_models.pth"
    print(f"\n💾 Saving retrained model to {model_path}...")
    
    checkpoint = {"performance_predictor": model.state_dict()}
    # ... (其余部分保持不变)
    try:
        # 尝试加载旧模型以保留策略网络等其他部分
        old_checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        checkpoint['policy_network'] = old_checkpoint.get('policy_network')
        checkpoint['gnn_encoder'] = old_checkpoint.get('gnn_encoder')
    except Exception:
        pass # 如果没有旧模型，就只保存新训练的预测器

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

if __name__ == "__main__":
    main()