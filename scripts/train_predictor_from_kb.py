#!/usr/bin/env python3
"""
WASS-RAG 阶段二：性能预测器训练脚本 (最终API修正版)

该脚本加载由 `generate_kb_dataset.py` 生成的高质量数据集，
并使用这些数据来训练 Performance Predictor 模型。
这解决了“训练-仿真偏差”问题，确保模型能够准确预测真实仿真环境下的性能。
"""

import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# 导入 AI 调度器中定义的模型结构
from src.ai_schedulers import PerformancePredictor

def load_training_dataset() -> List[Dict[str, Any]]:
    """加载知识库播种阶段生成的数据集"""
    dataset_path = Path("data/kb_training_dataset.json")
    if not dataset_path.exists():
        print(f"❌ 错误：数据集文件未找到于 {dataset_path}")
        print("   请先运行 `scripts/generate_kb_dataset.py` 来生成数据。")
        sys.exit(1)
        
    print(f"📚 Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    print(f"   Successfully loaded {len(dataset)} samples.")
    return dataset

def train_predictor(training_data: List[Dict[str, Any]], epochs: int = 100, batch_size: int = 512, learning_rate: float = 0.001):
    """
    使用加载的数据集训练 PerformancePredictor 模型。
    """
    print(f"\n🚀 Starting Performance Predictor training...")
    print(f"   Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # 1. 准备数据
    X = np.array([
        s['state_features'] + s['action_features'] + s['context_features'] 
        for s in training_data
    ])
    y = np.array([s['achieved_finish_time'] for s in training_data])
    
    y_mean, y_std = np.mean(y), np.std(y)
    if y_std < 1e-8: y_std = 1.0
    y_normalized = (y - y_mean) / y_std
    
    print(f"📈 Target (achieved_finish_time) stats: mean={y_mean:.2f}, std={y_std:.2f}")
    
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y_normalized).view(-1, 1).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. 初始化模型、损失函数和优化器
    model = PerformancePredictor(input_dim=X.shape[1], hidden_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 3. 训练循环
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "temp_best_predictor.pth")

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    # 4. 评估和验证
    print("\n✅ Training complete. Evaluating on the full dataset...")
    model.load_state_dict(torch.load("temp_best_predictor.pth"))
    os.remove("temp_best_predictor.pth")

    model.eval()
    with torch.no_grad():
        all_predictions_norm = model(X_tensor).squeeze().cpu().numpy()
        all_predictions = all_predictions_norm * y_std + y_mean
        
        mse = np.mean((all_predictions - y) ** 2)
        r2 = 1 - (np.sum((y - all_predictions) ** 2) / np.sum((y - y_mean) ** 2))
        
        print(f"   Validation Results: MSE = {mse:.4f}, R² = {r2:.4f}")
        
    return model, y_mean, y_std, {"r2": r2, "mse": mse}

def save_model(model: PerformancePredictor, y_mean: float, y_std: float, metrics: Dict):
    """将训练好的模型和元数据保存到 WASS 模型文件中"""
    model_path = Path("models/wass_models.pth")
    model_path.parent.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving trained model and metadata to {model_path}...")
    
    try:
        # --- API 修正处 ---
        # 添加 weights_only=False 以允许加载包含元数据的完整 checkpoint 文件
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        # --- 修正结束 ---
        print("   Found existing model file. Updating Performance Predictor weights.")
    except (FileNotFoundError, EOFError): # Also handle empty/corrupt files
        checkpoint = {}
        print("   No existing model file found or file is invalid. Creating a new checkpoint.")

    checkpoint["performance_predictor"] = model.state_dict()
    
    checkpoint["metadata"] = checkpoint.get("metadata", {})
    checkpoint["metadata"]["performance_predictor"] = {
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "training_samples": metrics['total_samples'],
        "retrained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_source": "kb_training_dataset.json",
        "validation_results": {
            "r2": metrics["r2"],
            "mse": metrics["mse"]
        }
    }
    
    torch.save(checkpoint, model_path)
    print(f"✅ Model saved successfully.")

def main():
    """主函数"""
    dataset = load_training_dataset()
    
    model, y_mean, y_std, metrics = train_predictor(dataset)
    
    metrics['total_samples'] = len(dataset)
    save_model(model, y_mean, y_std, metrics)

if __name__ == "__main__":
    main()