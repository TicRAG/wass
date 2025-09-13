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
from src.performance_predictor import PerformancePredictor, SimplePerformancePredictor

def load_training_dataset() -> List[Dict[str, Any]]:
    """加载知识库播种阶段生成的数据集"""
    # Align with JSONL knowledge base generation output name if needed
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
    修复：添加数据去重和正确的训练/验证集分割
    """
    print(f"\n🚀 Starting Performance Predictor training...")
    print(f"   Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # 1. 数据去重处理
    print(f"   Original samples: {len(training_data)}")
    
    # 使用特征组合作为键来去除重复样本
    unique_data = {}
    for sample in training_data:
        key = (tuple(sample['state_features']), tuple(sample['action_features']), tuple(sample['context_features']))
        if key not in unique_data:
            unique_data[key] = sample
    
    unique_samples = list(unique_data.values())
    print(f"   Unique samples after deduplication: {len(unique_samples)}")
    
    # 2. 准备数据
    X = np.array([
        s['state_features'] + s['action_features'] + s['context_features'] 
        for s in unique_samples
    ])
    y = np.array([s['achieved_finish_time'] for s in unique_samples])
    
    y_mean, y_std = np.mean(y), np.std(y)
    if y_std < 1e-8: y_std = 1.0
    
    print(f"📈 Target (achieved_finish_time) stats: mean={y_mean:.2f}, std={y_std:.2f}")
    
    # 3. 训练/验证集分割 (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"   Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 标准化数据
    y_train_normalized = (y_train - y_mean) / y_std
    y_val_normalized = (y_val - y_mean) / y_std
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train_normalized).view(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val_normalized).view(-1, 1).to(device)
    
    # 创建训练数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 4. 初始化模型、损失函数和优化器
    model = SimplePerformancePredictor(input_dim=X.shape[1], hidden_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 5. 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        # 早停检查和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "temp_best_predictor.pth")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停
        if patience_counter >= max_patience:
            print(f"   Early stopping at epoch {epoch+1} due to no improvement")
            break
    
    # 6. 加载最佳模型并最终评估
    print("\n✅ Training complete. Evaluating on validation set...")
    model.load_state_dict(torch.load("temp_best_predictor.pth"))
    os.remove("temp_best_predictor.pth")

    model.eval()
    with torch.no_grad():
        val_predictions_norm = model(X_val_tensor).cpu().numpy().flatten()
        val_predictions = val_predictions_norm * y_std + y_mean
        
        mse = np.mean((val_predictions - y_val) ** 2)
        mae = np.mean(np.abs(val_predictions - y_val))
        r2 = 1 - (np.sum((y_val - val_predictions) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))
        
        print(f"   Validation Results: MSE = {mse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")
        
        # 健全性检查和警告
        if r2 > 0.98:
            print(f"   ⚠️  警告: R²值过高 ({r2:.4f})，可能存在轻微过拟合，但仍可接受")
        elif r2 < 0.3:
            print(f"   ⚠️  警告: R²值较低 ({r2:.4f})，模型性能可能不佳")
        else:
            print(f"   ✅ R²值正常 ({r2:.4f})，模型训练成功")
        
    return model, y_mean, y_std, {"r2": r2, "mse": mse, "mae": mae}

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
        "unique_samples": metrics['unique_samples'],
        "retrained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_source": "kb_training_dataset.json",
        "validation_results": {
            "r2": metrics["r2"],
            "mse": metrics["mse"],
            "mae": metrics.get("mae", 0.0)
        },
        "training_improvements": "Added data deduplication and proper train/val split"
    }
    
    torch.save(checkpoint, model_path)
    print(f"✅ Model saved successfully.")

def main():
    """主函数"""
    dataset = load_training_dataset()
    
    model, y_mean, y_std, metrics = train_predictor(dataset)
    
    # 添加额外的统计信息
    metrics['total_samples'] = len(dataset)
    
    # 计算去重后的样本数量
    unique_data = {}
    for sample in dataset:
        key = (tuple(sample['state_features']), tuple(sample['action_features']), tuple(sample['context_features']))
        if key not in unique_data:
            unique_data[key] = sample
    metrics['unique_samples'] = len(unique_data)
    
    save_model(model, y_mean, y_std, metrics)

if __name__ == "__main__":
    main()