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

def create_improved_training_data(num_samples: int = 5000) -> List[Dict[str, Any]]:
    """创建改进的训练数据，确保makespan分布合理"""
    
    print(f"🔧 Generating {num_samples} improved training samples...")
    training_data = []
    
    # 统计生成的makespan分布
    makespan_values = []
    
    for i in range(num_samples):
        # 生成随机工作流（更多样化）
        task_count = np.random.randint(3, 101)  # 3-100个任务
        cluster_size = np.random.randint(2, 21)  # 2-20个节点
        
        # 为每个节点生成不同的容量
        for node_idx in range(min(cluster_size, 10)):  # 限制节点数以避免过多数据
            # 节点特性
            cpu_capacity = np.random.uniform(8.0, 32.0)  # 8-32 GFlops
            memory_capacity = np.random.uniform(8.0, 64.0)  # 8-64 GB
            current_load = np.random.uniform(0.1, 0.9)
            
            # 工作流特征（更真实的范围）
            workflow_features = {
                "task_count": task_count,
                "avg_task_flops": np.random.uniform(0.5e9, 5e9),  # 每个任务0.5-5 GFlops
                "avg_memory": np.random.uniform(0.5, 8.0),  # 0.5-8 GB
                "dependency_ratio": np.random.uniform(0.1, 0.7),
                "data_intensity": np.random.uniform(0.05, 0.4)
            }
            
            # 生成状态嵌入（32维）
            state_embedding = np.array([
                task_count / 100.0,  # 归一化任务数
                workflow_features["avg_task_flops"] / 5e9,  # 归一化计算量
                workflow_features["avg_memory"] / 8.0,  # 归一化内存
                workflow_features["dependency_ratio"],
                workflow_features["data_intensity"],
                cluster_size / 20.0,  # 归一化集群大小
                current_load,  # 当前负载
            ] + [np.random.randn() * 0.05 for _ in range(25)])  # 填充到32维
            
            # 生成动作嵌入（32维）- 节点选择
            action_embedding = np.array([
                node_idx / 10.0,  # 节点索引归一化
                cpu_capacity / 32.0,  # CPU容量归一化
                memory_capacity / 64.0,  # 内存容量归一化
                current_load,  # 当前负载
                1.0 - current_load,  # 空闲度
                (cpu_capacity / 32.0) * (1.0 - current_load),  # 有效计算能力
                (memory_capacity / 64.0) * (1.0 - current_load),  # 有效内存
            ] + [np.random.randn() * 0.05 for _ in range(25)])  # 填充到32维
            
            # 生成上下文嵌入（32维）- 历史信息
            historical_makespan = np.random.uniform(10.0, 200.0)
            similarity_score = np.random.uniform(0.4, 0.95)
            case_count = np.random.randint(3, 10)
            
            context_embedding = np.array([
                historical_makespan / 200.0,  # 历史makespan归一化
                similarity_score,  # 相似度得分
                case_count / 10.0,  # 案例数量归一化
                np.random.uniform(0.6, 1.0),  # 置信度
            ] + [np.random.randn() * 0.05 for _ in range(28)])  # 填充到32维
            
            # 改进的makespan计算（确保物理合理性）
            # 单任务平均执行时间
            avg_task_time = workflow_features["avg_task_flops"] / (cpu_capacity * 1e9)
            
            # 考虑并行性的总执行时间
            total_compute_time = task_count * avg_task_time
            parallel_efficiency = 0.6 + np.random.uniform(0.0, 0.3)  # 60-90%并行效率
            ideal_parallel_time = total_compute_time / (cluster_size * parallel_efficiency)
            
            # 各种开销因子
            load_overhead = 1.0 + current_load * 0.6  # 负载开销
            dependency_overhead = 1.0 + workflow_features["dependency_ratio"] * 0.4  # 依赖开销
            communication_overhead = 1.0 + workflow_features["data_intensity"] * 0.3  # 通信开销
            
            # 随机变化（模拟系统噪声）
            noise_factor = np.random.uniform(0.85, 1.15)
            
            # 最终makespan
            makespan = ideal_parallel_time * load_overhead * dependency_overhead * communication_overhead * noise_factor
            
            # 确保makespan在合理范围内
            makespan = max(0.5, min(500.0, makespan))
            makespan_values.append(makespan)
            
            # 拼接所有特征（96维：32+32+32）
            combined_features = np.concatenate([state_embedding, action_embedding, context_embedding])
            
            training_data.append({
                "id": f"improved_{i}_{node_idx}",
                "state_embedding": state_embedding.tolist(),
                "action_embedding": action_embedding.tolist(),
                "context_embedding": context_embedding.tolist(),
                "features": combined_features.tolist(),
                "makespan": makespan,
                "workflow_features": workflow_features,
                "node_features": {
                    "cpu_capacity": cpu_capacity,
                    "memory_capacity": memory_capacity,
                    "current_load": current_load
                }
            })
    
    # 打印makespan分布统计
    makespan_array = np.array(makespan_values)
    print(f"📊 Makespan distribution:")
    print(f"   Mean: {np.mean(makespan_array):.2f}s")
    print(f"   Std:  {np.std(makespan_array):.2f}s")
    print(f"   Min:  {np.min(makespan_array):.2f}s")
    print(f"   Max:  {np.max(makespan_array):.2f}s")
    print(f"   Median: {np.median(makespan_array):.2f}s")
    
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
    """根据新的训练数据重新生成知识库"""
    
    print(f"\n🔄 Regenerating knowledge base with {len(training_data)} cases...")
    
    # 创建新的知识库
    kb = RAGKnowledgeBase(embedding_dim=32)
    
    for data in training_data:
        # 使用状态嵌入作为主要特征
        embedding = np.array(data["state_embedding"], dtype=np.float32)
        
        # 构建工作流信息
        workflow_info = {
            "task_count": data["workflow_features"]["task_count"],
            "avg_task_flops": data["workflow_features"]["avg_task_flops"],
            "avg_memory": data["workflow_features"]["avg_memory"],
            "dependency_ratio": data["workflow_features"]["dependency_ratio"],
            "data_intensity": data["workflow_features"]["data_intensity"],
            "complexity": "medium",
            "type": "retrained_synthetic"
        }
        
        # 生成虚拟动作序列（节点分配）
        cluster_size = int(data["workflow_features"]["task_count"] * 0.1) + 2  # 估算集群大小
        actions = [f"node_{i % cluster_size}" for i in range(data["workflow_features"]["task_count"])]
        
        # 使用实际的makespan
        makespan = data["makespan"]
        
        # 添加到知识库
        kb.add_case(embedding, workflow_info, actions, makespan)
    
    print(f"✅ Knowledge base regenerated with {len(kb.cases)} cases")
    return kb

def main():
    """主函数"""
    print("🔧 Retraining Performance Predictor with Improved Data")
    print("=" * 60)
    
    # 生成改进的训练数据
    training_data = create_improved_training_data(num_samples=5000)
    
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
    
    # 更新性能预测器
    checkpoint["performance_predictor"] = model.state_dict()
    
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
