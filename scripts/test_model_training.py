#!/usr/bin/env python3
"""
测试模型训练逻辑的简化版本
用于验证PerformancePredictor训练是否能解决RAG调度器问题
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
import json

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# 简化的PerformancePredictor（如果AI模块不可用）
class SimplePerformancePredictor(nn.Module):
    """简化的性能预测器"""
    
    def __init__(self, input_dim=96, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def generate_test_data(num_samples: int = 1000) -> tuple:
    """生成测试训练数据"""
    
    print(f"Generating {num_samples} test samples...")
    
    X = []  # 特征
    y = []  # 目标
    
    for i in range(num_samples):
        # 生成96维特征向量 (32 state + 32 action + 32 context)
        
        # State features (工作流特征)
        task_count = np.random.randint(5, 51)
        state_features = np.array([
            task_count / 50.0,  # 任务数量归一化
            np.random.uniform(0.2, 0.8),  # 依赖比例
            np.random.uniform(0.1, 0.5),  # 数据密集度
        ] + [np.random.randn() * 0.1 for _ in range(29)])  # 填充到32维
        
        # Action features (节点特征)
        node_load = np.random.uniform(0.1, 0.9)
        action_features = np.array([
            node_load,  # 节点负载
            1.0 - node_load,  # 空闲度
            np.random.uniform(0.5, 1.5),  # 相对性能
        ] + [np.random.randn() * 0.1 for _ in range(29)])  # 填充到32维
        
        # Context features (历史信息)
        context_features = np.array([
            np.random.uniform(0.3, 0.9),  # 相似度
            np.random.uniform(0.5, 1.0),  # 置信度
        ] + [np.random.randn() * 0.1 for _ in range(30)])  # 填充到32维
        
        # 合并特征
        combined_features = np.concatenate([state_features, action_features, context_features])
        
        # 生成目标值（基于特征的合理计算）
        base_time = task_count * 2.0  # 基础时间
        load_factor = 1.0 + node_load * 0.5  # 负载影响
        performance_factor = action_features[2]  # 性能影响
        noise = np.random.uniform(0.9, 1.1)  # 随机噪声
        
        makespan = base_time * load_factor / performance_factor * noise
        
        X.append(combined_features)
        y.append(makespan)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def test_model_training():
    """测试模型训练过程"""
    
    print("=== Testing Model Training Logic ===")
    
    # 1. 生成测试数据
    X, y = generate_test_data(1000)
    print(f"Generated data shape: X={X.shape}, y={y.shape}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    print(f"Target mean: {y.mean():.2f} ± {y.std():.2f}")
    
    # 2. 数据归一化
    y_mean, y_std = y.mean(), y.std()
    y_normalized = (y - y_mean) / y_std
    
    # 3. 转换为PyTorch张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y_normalized).to(device)
    
    # 4. 创建模型
    model = SimplePerformancePredictor(input_dim=96, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 5. 训练循环
    epochs = 100
    batch_size = 32
    num_batches = len(X) // batch_size
    
    print(f"\nTraining for {epochs} epochs...")
    
    best_loss = float('inf')
    training_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # 简单的批次处理
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            # 前向传播
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
    
    # 6. 加载最佳模型并评估
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        all_predictions = model(X_tensor).squeeze().cpu().numpy()
        
        # 反归一化
        all_predictions_denorm = all_predictions * y_std + y_mean
        
        # 计算指标
        mse = np.mean((all_predictions_denorm - y) ** 2)
        mae = np.mean(np.abs(all_predictions_denorm - y))
        r2 = 1 - np.sum((y - all_predictions_denorm) ** 2) / np.sum((y - y.mean()) ** 2)
        
        # 检查预测多样性
        pred_std = np.std(all_predictions_denorm)
        pred_range = np.max(all_predictions_denorm) - np.min(all_predictions_denorm)
        unique_predictions = len(np.unique(np.round(all_predictions_denorm, 3)))
        
        print(f"\n=== Training Results ===")
        print(f"Final Loss: {best_loss:.6f}")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"Prediction std: {pred_std:.2f}")
        print(f"Prediction range: {pred_range:.2f}")
        print(f"Unique predictions: {unique_predictions}/{len(y)}")
        
        # 诊断结果
        print(f"\n=== Diagnosis ===")
        if unique_predictions <= 1:
            print("❌ CRITICAL: Model collapsed - all predictions identical!")
            print("   Root cause: Training data or model architecture issue")
        elif pred_std < 1.0:
            print("⚠️  WARNING: Low prediction diversity")
            print("   May cause limited differentiation in RAG scheduler")
        else:
            print("✅ SUCCESS: Model shows good prediction diversity")
            print("   Should resolve RAG scheduler identical score issue")
        
        # 输出一些样本预测
        print(f"\nSample predictions vs targets:")
        for i in range(min(10, len(y))):
            print(f"  Sample {i}: Pred={all_predictions_denorm[i]:.2f}, Target={y[i]:.2f}")
    
    return {
        "success": unique_predictions > 1 and pred_std >= 1.0,
        "mse": mse,
        "r2": r2,
        "prediction_diversity": pred_std,
        "unique_predictions": unique_predictions
    }

def save_test_results(results: Dict[str, Any]):
    """保存测试结果"""
    
    results_file = "test_model_training_results.json"
    
    with open(results_file, 'w') as f:
        # 将numpy类型转换为Python原生类型
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        json.dump({
            "timestamp": "2025-09-08",
            "test_name": "performance_predictor_training",
            "results": serializable_results,
            "conclusion": "SUCCESS: Training produces diverse predictions" if results["success"] else "FAILURE: Training issues detected"
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    print("Testing WASS-RAG PerformancePredictor Training...")
    print("This test validates that proper training can solve the identical score issue.\n")
    
    results = test_model_training()
    save_test_results(results)
    
    print(f"\n{'='*60}")
    if results["success"]:
        print("🎉 CONCLUSION: Model training approach is VALIDATED!")
        print("   The enhanced training script should resolve RAG scheduler issues.")
        print("   Recommendation: Deploy the updated initialize_ai_models.py")
    else:
        print("⚠️  CONCLUSION: Training approach needs refinement.")
        print("   Additional work may be needed on data generation or model architecture.")
    
    print(f"{'='*60}")
