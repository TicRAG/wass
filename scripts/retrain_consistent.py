#!/usr/bin/env python3
"""
完全重写的性能预测器训练脚本
确保训练和预测时使用完全相同的特征生成逻辑
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Any
import pickle

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)  # 添加项目根目录
sys.path.insert(0, os.path.join(project_root, 'src'))  # 添加src目录

try:
    from ai_schedulers import WASSRAGScheduler, SchedulingState
    from interfaces import PerformancePredictor
    print("✅ Successfully imported all required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_consistent_training_data(num_samples: int = 5000) -> List[Dict[str, Any]]:
    """
    生成与ai_schedulers.py完全一致的训练数据
    """
    print(f"🔧 Generating {num_samples} training samples with consistent features...")

    # 创建临时调度器实例来复用特征编码逻辑
    temp_scheduler = WASSRAGScheduler()
    training_data = []
    makespan_values = []

    for i in range(num_samples):
        # 1. 创建随机调度场景
        num_nodes = np.random.randint(2, 11)  # 2-10个节点
        
        nodes = {f"node_{j}": {
            "cpu_capacity": round(np.random.uniform(2.0, 8.0), 2),
            "memory_capacity": round(np.random.uniform(8.0, 64.0), 2), 
            "current_load": round(np.random.uniform(0.1, 0.9), 2),
        } for j in range(num_nodes)}

        # 任务信息
        task_info = {
            "id": f"task_{i}",
            "flops": float(np.random.uniform(0.5e9, 15e9)),  # 0.5-15 GFlops
            "memory": round(np.random.uniform(1.0, 16.0), 2),
            "dependencies": []
        }

        # 创建调度状态
        state = SchedulingState(
            workflow_graph={"tasks": [task_info], "task_requirements": {f"task_{i}": task_info}},
            cluster_state={"nodes": nodes},
            pending_tasks=[f"task_{i}"],
            current_task=f"task_{i}",
            available_nodes=list(nodes.keys()),
            timestamp=0.0
        )

        # 2. 使用调度器的方法生成状态嵌入
        state_embedding = temp_scheduler._extract_simple_features_fallback(state)
        
        # 3. 为每个节点生成训练样本
        for node_name, node_details in nodes.items():
            # 关键：使用与预测时完全相同的_encode_action函数
            action_embedding = temp_scheduler._encode_action(node_name, state)
            
            # 模拟上下文嵌入
            context_embedding = torch.randn(32, device=temp_scheduler.device)
            
            # 拼接96维特征
            combined_features = torch.cat([
                state_embedding,
                action_embedding,
                context_embedding
            ]).cpu().numpy()
            
            # 4. 计算真实执行时间标签
            task_cpu_gflops = task_info["flops"] / 1e9
            node_cpu_cap = node_details["cpu_capacity"]
            node_load = node_details["current_load"]
            
            available_cpu = node_cpu_cap * (1.0 - node_load)
            
            # 基础执行时间
            base_time = task_cpu_gflops / max(available_cpu, 0.1)
            
            # 添加各种现实因素
            mem_penalty = max(0, task_info["memory"] - node_details["memory_capacity"]) * 0.5
            load_penalty = node_load * 2.0
            random_noise = np.random.uniform(-0.5, 0.5)
            
            execution_time = base_time + mem_penalty + load_penalty + random_noise
            execution_time = max(1.0, min(180.0, execution_time))  # 约束范围
            
            makespan_values.append(execution_time)
            
            training_data.append({
                "features": combined_features.tolist(),
                "makespan": execution_time
            })
    
    # 打印统计
    makespan_array = np.array(makespan_values)
    print(f"📊 Execution time distribution:")
    print(f"   Mean: {np.mean(makespan_array):.2f}s")
    print(f"   Std:  {np.std(makespan_array):.2f}s")
    print(f"   Range: [{np.min(makespan_array):.2f}, {np.max(makespan_array):.2f}]s")
    
    return training_data

def train_consistent_model(training_data: List[Dict[str, Any]], epochs: int = 200):
    """
    训练性能预测器
    """
    print(f"🚀 Training model with {len(training_data)} samples...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # 准备数据
    X = np.array([sample["features"] for sample in training_data])
    y = np.array([sample["makespan"] for sample in training_data])
    
    # 归一化标签
    y_mean, y_std = np.mean(y), np.std(y)
    y_normalized = (y - y_mean) / (y_std + 1e-8)
    
    print(f"📈 Training statistics:")
    print(f"   Original y: mean={y_mean:.2f}, std={y_std:.2f}")
    print(f"   Normalized y range: [{np.min(y_normalized):.3f}, {np.max(y_normalized):.3f}]")
    
    # 转换为PyTorch
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y_normalized).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 模型和优化器
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
        
        # 早停和日志
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch:3d}: loss={avg_loss:.6f}, best={best_loss:.6f}")
        
        if patience_counter >= 50:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    print(f"✅ Training completed. Final loss: {best_loss:.6f}")
    
    # 保存模型和归一化参数
    model_path = os.path.join(project_root, "models", "wass_models.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 加载已有模型组件（如果存在）
    checkpoint = {}
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            print("📦 Loaded existing model components")
        except:
            print("📦 Creating new model checkpoint")
    
    # 保存所有组件
    checkpoint.update({
        'performance_predictor_state_dict': model.state_dict(),
        'y_mean': y_mean,
        'y_std': y_std,
        'training_samples': len(training_data),
        'final_loss': best_loss
    })
    
    torch.save(checkpoint, model_path)
    print(f"💾 Model saved to {model_path}")
    print(f"📊 Normalization: mean={y_mean:.2f}, std={y_std:.2f}")

def main():
    """主函数"""
    print("🔄 Retraining Performance Predictor with Consistent Features")
    print("=" * 60)
    
    # 1. 生成一致的训练数据
    training_data = create_consistent_training_data(num_samples=5000)
    
    # 2. 训练模型
    train_consistent_model(training_data, epochs=200)
    
    print("\n✅ Retraining completed successfully!")
    print("🧪 Run 'python test_predictions.py' to validate improvements")

if __name__ == "__main__":
    main()
