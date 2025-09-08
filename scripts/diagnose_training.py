#!/usr/bin/env python3
"""
诊断训练数据和模型问题
"""

import json
import numpy as np
import torch
import torch.nn as nn

def analyze_training_data():
    """分析训练数据质量"""
    
    print("=== Training Data Analysis ===")
    
    try:
        with open("data/synthetic_training_data.json", 'r') as f:
            training_data = json.load(f)
        print(f"✓ Loaded {len(training_data)} training samples")
    except Exception as e:
        print(f"✗ Failed to load training data: {e}")
        return
    
    # 提取特征和目标
    X = np.array([data["combined_features"] for data in training_data], dtype=np.float32)
    y = np.array([data["makespan"] for data in training_data], dtype=np.float32)
    
    print(f"\n--- Feature Analysis ---")
    print(f"Feature shape: {X.shape}")
    print(f"Feature mean: {np.mean(X):.6f}")
    print(f"Feature std: {np.std(X):.6f}")
    print(f"Feature range: [{np.min(X):.6f}, {np.max(X):.6f}]")
    
    # 检查特征中的NaN或inf
    nan_count = np.sum(np.isnan(X))
    inf_count = np.sum(np.isinf(X))
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")
    
    # 检查特征多样性
    unique_rows = len(np.unique(X, axis=0))
    print(f"Unique feature vectors: {unique_rows}/{len(X)} ({unique_rows/len(X)*100:.1f}%)")
    
    print(f"\n--- Target Analysis ---")
    print(f"Target shape: {y.shape}")
    print(f"Target mean: {np.mean(y):.2f}")
    print(f"Target std: {np.std(y):.2f}")
    print(f"Target range: [{np.min(y):.2f}, {np.max(y):.2f}]")
    
    unique_targets = len(np.unique(y))
    print(f"Unique target values: {unique_targets}/{len(y)} ({unique_targets/len(y)*100:.1f}%)")
    
    # 检查目标中的NaN或inf
    y_nan_count = np.sum(np.isnan(y))
    y_inf_count = np.sum(np.isinf(y))
    print(f"Target NaN count: {y_nan_count}")
    print(f"Target Inf count: {y_inf_count}")
    
    # 检查线性相关性
    try:
        # 简单的线性回归测试
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        r2 = r2_score(y, y_pred)
        print(f"\nLinear regression R²: {r2:.4f}")
        
        if r2 < 0.1:
            print("⚠️  Very low linear correlation - features may not predict targets well")
        
    except ImportError:
        print("sklearn not available for correlation test")
    
    # 采样显示几个样本
    print(f"\n--- Sample Data ---")
    for i in range(min(3, len(training_data))):
        sample = training_data[i]
        print(f"Sample {i}:")
        print(f"  ID: {sample['id']}")
        print(f"  Makespan: {sample['makespan']:.2f}")
        print(f"  Features[0:5]: {sample['combined_features'][:5]}")
        print(f"  Features[32:37]: {sample['combined_features'][32:37]}")
        print(f"  Features[64:69]: {sample['combined_features'][64:69]}")

def test_simple_model():
    """测试简单模型是否能学习"""
    
    print(f"\n=== Simple Model Test ===")
    
    # 生成简单的测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成96维特征，其中前几维对目标有影响
    X = np.random.randn(n_samples, 96).astype(np.float32)
    
    # 目标函数：基于前几个特征的简单线性组合 + 噪声
    y = (X[:, 0] * 10 + X[:, 1] * 5 + X[:, 2] * 3 + 
         X[:, 32] * 2 + X[:, 64] * 1 + 
         np.random.randn(n_samples) * 0.1 + 50).astype(np.float32)
    
    print(f"Generated test data: X={X.shape}, y={y.shape}")
    print(f"Target range: [{np.min(y):.2f}, {np.max(y):.2f}]")
    
    # 数据归一化
    y_mean, y_std = np.mean(y), np.std(y)
    y_normalized = (y - y_mean) / y_std
    
    # 转换为PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y_normalized).to(device)
    
    # 定义简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(96, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.network(x)
    
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        predictions = model(X_tensor).squeeze()
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    # 评估
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_tensor).squeeze()
        final_loss = criterion(final_predictions, y_tensor)
        
        # 反归一化
        pred_denorm = final_predictions.cpu().numpy() * y_std + y_mean
        r2 = 1 - np.sum((y - pred_denorm) ** 2) / np.sum((y - np.mean(y)) ** 2)
        pred_std = np.std(pred_denorm)
        
        print(f"\nSimple model results:")
        print(f"Final loss: {final_loss.item():.6f}")
        print(f"R²: {r2:.4f}")
        print(f"Prediction std: {pred_std:.2f}")
        
        if r2 > 0.8 and pred_std > 1.0:
            print("✅ Simple model works - issue is with training data")
        else:
            print("❌ Simple model also fails - may be architecture or PyTorch issue")

if __name__ == "__main__":
    analyze_training_data()
    test_simple_model()
