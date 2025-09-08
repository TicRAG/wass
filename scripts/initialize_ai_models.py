#!/usr/bin/env python3
"""
WASS-RAG AI模型训练和初始化脚本
为实验框架准备预训练的AI模型和知识库
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import pickle

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    from src.ai_schedulers import (
        GraphEncoder, PolicyNetwork, PerformancePredictor, 
        RAGKnowledgeBase, WASSHeuristicScheduler, create_scheduler
    )
    HAS_AI_MODULES = True
except ImportError as e:
    print(f"Warning: AI modules not available: {e}")
    HAS_AI_MODULES = False

class PerformancePredictorDataset(Dataset):
    """性能预测器训练数据集"""
    
    def __init__(self, training_data: List[Dict[str, Any]]):
        self.data = training_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 拼接 state + action + context (32 + 32 + 32 = 96维)
        state = torch.tensor(sample["state_embedding"], dtype=torch.float32)
        action = torch.tensor(sample["action_embedding"], dtype=torch.float32)
        context = torch.tensor(sample["context_embedding"], dtype=torch.float32)
        
        features = torch.cat([state, action, context])  # 96维特征向量
        target = torch.tensor(sample["makespan"], dtype=torch.float32)
        
        return features, target

def train_performance_predictor(training_data: List[Dict[str, Any]], 
                              epochs: int = 100, 
                              batch_size: int = 32,
                              learning_rate: float = 0.001) -> PerformancePredictor:
    """训练性能预测器模型"""
    
    print(f"Training PerformancePredictor with {len(training_data)} samples...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建数据集和数据加载器
    dataset = PerformancePredictorDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = PerformancePredictor(input_dim=96, hidden_dim=128).to(device)
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练循环
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            
            # 前向传播
            predictions = model(features).squeeze()
            loss = criterion(predictions, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # 打印训练进度
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    print(f"Training completed. Final loss: {best_loss:.6f}")
    return model

def validate_trained_model(model_path: str, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """验证训练好的模型性能"""
    
    if not HAS_AI_MODULES:
        print("Cannot validate model: AI modules not available")
        return {}
    
    print(f"\n=== Validating Trained Model ===")
    
    # 加载模型
    try:
        models = torch.load(model_path, map_location="cpu")
        
        # 创建PerformancePredictor实例并加载权重
        model = PerformancePredictor(input_dim=96, hidden_dim=128)
        model.load_state_dict(models["performance_predictor"])
        model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return {}
    
    # 准备测试数据（使用部分训练数据进行验证）
    test_samples = training_data[:100]  # 使用前100个样本
    X_test = np.array([data["combined_features"] for data in test_samples], dtype=np.float32)
    y_test = np.array([data["makespan"] for data in test_samples], dtype=np.float32)
    
    # 获取训练时的归一化参数
    y_mean = models["metadata"]["performance_predictor"]["y_mean"]
    y_std = models["metadata"]["performance_predictor"]["y_std"]
    
    # 进行预测
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        predictions_normalized = model(X_tensor).squeeze().cpu().numpy()
        
        # 反归一化预测结果
        predictions = predictions_normalized * y_std + y_mean
    
    # 计算评估指标
    mse = float(np.mean((predictions - y_test) ** 2))
    mae = float(np.mean(np.abs(predictions - y_test)))
    
    # 检查预测多样性
    pred_std = float(np.std(predictions))
    pred_range = float(np.max(predictions) - np.min(predictions))
    
    # 检查是否所有预测都相同（RAG问题的核心）
    unique_predictions = len(np.unique(np.round(predictions, 6)))
    
    results = {
        "mse": mse,
        "mae": mae,
        "prediction_std": pred_std,
        "prediction_range": pred_range,
        "unique_predictions": unique_predictions,
        "total_samples": len(test_samples)
    }
    
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Prediction diversity (std): {pred_std:.2f}")
    print(f"Prediction range: {pred_range:.2f}")
    print(f"Unique predictions: {unique_predictions}/{len(test_samples)}")
    
    # 判断模型质量
    if unique_predictions <= 1:
        print("❌ CRITICAL: Model produces identical predictions!")
        print("   This will cause RAG scheduler to output same scores for all nodes.")
    elif pred_std < 1.0:
        print("⚠️  WARNING: Low prediction diversity")
        print("   RAG scheduler may have limited differentiation capability.")
    else:
        print("✅ GOOD: Model shows healthy prediction diversity")
        print("   RAG scheduler should be able to differentiate between nodes.")
    
    return results

def create_synthetic_training_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """创建合成的训练数据 - 增强版，包含完整的 (state, action, context) 特征"""
    
    training_data = []
    
    for i in range(num_samples):
        # 生成随机工作流
        task_count = np.random.randint(5, 51)  # 5-50个任务
        cluster_size = np.random.randint(2, 17)  # 2-16个节点
        
        # 生成工作流特征
        workflow_features = {
            "task_count": task_count,
            "avg_flops": np.random.uniform(1e9, 10e9),
            "avg_memory": np.random.uniform(0.5e9, 4e9),
            "dependency_ratio": np.random.uniform(0.2, 0.8),
            "data_intensity": np.random.uniform(0.1, 0.5)
        }
        
        # 生成集群特征
        cluster_features = {
            "node_count": cluster_size,
            "avg_cpu_capacity": 10.0,  # GFlops
            "avg_memory_capacity": 16.0,  # GB
            "avg_load": np.random.uniform(0.2, 0.8)
        }
        
        # 生成状态嵌入（32维）
        state_embedding = np.array([
            workflow_features["task_count"] / 100.0,
            workflow_features["avg_flops"] / 10e9,
            workflow_features["avg_memory"] / 4e9,
            workflow_features["dependency_ratio"],
            workflow_features["data_intensity"],
            cluster_features["node_count"] / 16.0,
            cluster_features["avg_load"],
        ] + [np.random.randn() * 0.1 for _ in range(25)])  # 填充到32维，减小随机噪声
        
        # 为多个节点生成样本（模拟真实调度场景）
        for node_idx in range(min(cluster_size, 8)):  # 限制为8个节点样本
            # 生成动作嵌入（32维）- 节点特征
            current_load = np.random.uniform(0.1, 0.9)
            cpu_capacity = np.random.uniform(8.0, 12.0)
            memory_capacity = np.random.uniform(12.0, 20.0)
            
            action_embedding = np.array([
                node_idx / 16.0,  # 节点ID归一化
                cluster_size / 16.0,  # 可用节点数
                1.0 if node_idx == 0 else 0.0,  # 是否首选节点
                current_load,  # 当前负载
                1.0 - current_load,  # 空闲度
                cpu_capacity / 16.0,  # CPU容量归一化
                memory_capacity / 32.0,  # 内存容量归一化
            ] + [np.random.randn() * 0.05 for _ in range(25)])  # 填充到32维
            
            # 生成上下文嵌入（32维）- 历史信息
            historical_makespan = np.random.uniform(50.0, 200.0)
            similarity_score = np.random.uniform(0.3, 0.9)
            case_count = np.random.randint(3, 8)
            
            context_embedding = np.array([
                historical_makespan / 200.0,  # 历史makespan归一化
                similarity_score,  # 相似度得分
                case_count / 10.0,  # 案例数量归一化
                np.random.uniform(0.5, 1.0),  # 置信度
            ] + [np.random.randn() * 0.05 for _ in range(28)])  # 填充到32维
            
            # 计算真实makespan（改进的物理意义合理的公式）
            # 基础计算时间：任务数量 * 平均任务时间 / 并行度
            avg_task_compute_time = workflow_features["avg_flops"] / (cpu_capacity * 1e9)  # 秒
            sequential_time = task_count * avg_task_compute_time
            parallel_efficiency = 0.7 + np.random.uniform(-0.1, 0.1)  # 70%±10% 并行效率
            base_makespan = sequential_time / (cluster_size * parallel_efficiency)
            
            # 影响因子
            load_factor = 1.0 + current_load * 0.8  # 负载影响更显著
            dependency_factor = 1.0 + workflow_features["dependency_ratio"] * 0.5  # 依赖影响
            data_factor = 1.0 + workflow_features["data_intensity"] * 0.3  # 数据传输影响
            
            # 添加合理的随机变化
            noise_factor = np.random.uniform(0.8, 1.2)
            makespan = base_makespan * load_factor * dependency_factor * data_factor * noise_factor
            
            # 确保makespan在合理范围内 (1秒到300秒)
            makespan = max(1.0, min(300.0, makespan))
            
            # 拼接所有特征（96维：32+32+32）
            combined_features = np.concatenate([state_embedding, action_embedding, context_embedding])
            
            training_data.append({
                "id": f"synthetic_{i}_{node_idx}",
                "workflow_features": workflow_features,
                "cluster_features": cluster_features,
                "node_features": {
                    "node_id": node_idx,
                    "current_load": current_load,
                    "cpu_capacity": cpu_capacity,
                    "memory_capacity": memory_capacity
                },
                "state_embedding": state_embedding.tolist(),
                "action_embedding": action_embedding.tolist(),
                "context_embedding": context_embedding.tolist(),
                "combined_features": combined_features.tolist(),  # 96维输入特征
                "makespan": float(makespan),  # 目标值
                "timestamp": time.time(),
                "method": "synthetic_heuristic"
            })
    
    print(f"Generated {len(training_data)} training samples from {num_samples} scenarios")
    return training_data

def train_performance_predictor(training_data: List[Dict[str, Any]], 
                               epochs: int = 100, 
                               batch_size: int = 32,
                               learning_rate: float = 0.0001) -> Dict[str, Any]:  # 降低学习率
    """训练性能预测器模型"""
    
    if not HAS_AI_MODULES:
        print("Cannot train model: AI modules not available")
        return None
        
    print(f"\n=== Training PerformancePredictor ===")
    print(f"Training samples: {len(training_data)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 准备数据
    X = np.array([data["combined_features"] for data in training_data], dtype=np.float32)
    y = np.array([data["makespan"] for data in training_data], dtype=np.float32)
    
    # 检查数据质量
    print(f"Feature stats: mean={np.mean(X):.6f}, std={np.std(X):.6f}")
    print(f"Target stats: mean={np.mean(y):.2f}, std={np.std(y):.2f}")
    
    # 检查数据多样性
    unique_features = len(np.unique(X.round(6), axis=0))
    unique_targets = len(np.unique(y.round(3)))
    print(f"Unique features: {unique_features}/{len(X)} ({unique_features/len(X)*100:.1f}%)")
    print(f"Unique targets: {unique_targets}/{len(y)} ({unique_targets/len(y)*100:.1f}%)")
    
    if unique_features < len(X) * 0.5:
        print("⚠️  Warning: Low feature diversity detected!")
    if unique_targets < len(y) * 0.1:
        print("⚠️  Warning: Low target diversity detected!")
    
    # 数据归一化
    y_mean, y_std = np.mean(y), np.std(y)
    y_normalized = (y - y_mean) / (y_std + 1e-8)  # 添加小值避免除零
    
    print(f"Normalized target stats: mean={np.mean(y_normalized):.6f}, std={np.std(y_normalized):.6f}")
    print(f"Target range: [{np.min(y_normalized):.3f}, {np.max(y_normalized):.3f}]")
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y_normalized).to(device)
    
    # 创建数据集和数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = PerformancePredictor(input_dim=96, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    
    # 训练循环
    training_losses = []
    best_loss = float('inf')
    
    print(f"\nStarting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)
        
        # 记录最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
        
        # 每10个epoch打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        all_predictions = model(X_tensor).squeeze().cpu().numpy()
        
        # 反归一化预测结果
        all_predictions_denorm = all_predictions * y_std + y_mean
        
        # 计算评估指标
        mse = np.mean((all_predictions_denorm - y) ** 2)
        mae = np.mean(np.abs(all_predictions_denorm - y))
        r2 = 1 - np.sum((y - all_predictions_denorm) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        print(f"\n=== Training Results ===")
        print(f"Final Loss: {best_loss:.6f}")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        
        # 检查预测多样性
        pred_std = np.std(all_predictions_denorm)
        pred_range = np.max(all_predictions_denorm) - np.min(all_predictions_denorm)
        print(f"Prediction std: {pred_std:.2f}")
        print(f"Prediction range: {pred_range:.2f}")
        
        if pred_std < 1.0:
            print("⚠️  Warning: Low prediction diversity detected!")
        else:
            print("✓ Good prediction diversity achieved")
    
    # 返回训练好的模型和元数据
    return {
        "model_state_dict": best_model_state,
        "training_metadata": {
            "epochs": epochs,
            "best_loss": float(best_loss),
            "final_mse": float(mse),
            "final_mae": float(mae),
            "final_r2": float(r2),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "prediction_std": float(pred_std),
            "prediction_range": float(pred_range),
            "training_samples": len(training_data),
            "device": str(device)
        }
    }

def create_trained_models(training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """创建经过训练的模型"""
    
    print(f"\n=== Creating Trained Models ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    
    # 1. 训练性能预测器（核心）
    print(f"\n1. Training PerformancePredictor...")
    perf_predictor_result = train_performance_predictor(
        training_data, 
        epochs=200,  # 增加训练轮数
        batch_size=64,  # 增加批次大小
        learning_rate=0.0001  # 降低学习率
    )
    
    if perf_predictor_result:
        models["performance_predictor"] = perf_predictor_result["model_state_dict"]
        training_metadata = perf_predictor_result["training_metadata"]
    else:
        print("Failed to train PerformancePredictor, using random initialization")
        models["performance_predictor"] = PerformancePredictor(input_dim=96, hidden_dim=128).state_dict()
        training_metadata = {"error": "training_failed"}
    
    # 2. 创建策略网络（DRL用，暂时随机初始化）
    print(f"\n2. Creating PolicyNetwork...")
    models["policy_network"] = PolicyNetwork(
        state_dim=32,
        action_dim=1,
        hidden_dim=128
    ).state_dict()
    
    # 3. 创建GNN编码器（如果可用）
    print(f"\n3. Creating GraphEncoder...")
    try:
        models["gnn_encoder"] = GraphEncoder(
            node_feature_dim=8,
            edge_feature_dim=4,
            hidden_dim=64,
            output_dim=32
        ).state_dict()
        print("   ✓ GraphEncoder created successfully")
    except Exception as e:
        print(f"   ⚠️  Skipping GraphEncoder: {e}")
    
    # 4. 添加训练元数据
    models["metadata"] = {
        "model_version": "2.0.0",
        "created_at": time.time(),
        "performance_predictor": training_metadata,
        "notes": "PerformancePredictor trained on synthetic data, others randomly initialized"
    }
    
    return models

def create_dummy_knowledge_base(training_data: List[Dict[str, Any]]) -> RAGKnowledgeBase:
    """创建知识库并填充合成数据"""
    
    if not HAS_AI_MODULES:
        print("Cannot create knowledge base: AI modules not available")
        return None
    
    kb = RAGKnowledgeBase(embedding_dim=32)
    
    print(f"Adding {len(training_data)} cases to knowledge base...")
    
    for data in training_data:
        # 提取特征
        embedding = np.array(data["state_embedding"], dtype=np.float32)
        
        workflow_info = {
            "task_count": data["workflow_features"]["task_count"],
            "complexity": "medium",
            "type": "synthetic"
        }
        
        actions = [f"node_{i % data['cluster_features']['node_count']}" 
                  for i in range(data["workflow_features"]["task_count"])]
        
        makespan = data["makespan"]
        
        # 添加到知识库
        kb.add_case(embedding, workflow_info, actions, makespan)
    
    print(f"Knowledge base created with {len(kb.cases)} cases")
    return kb

def initialize_ai_models_and_kb():
    """初始化AI模型和知识库"""
    
    print("=== WASS-RAG AI Model and Knowledge Base Initialization ===")
    
    # 创建目录
    model_dir = Path("models")
    data_dir = Path("data")
    
    model_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    # 1. 生成训练数据
    print("\n1. Generating synthetic training data...")
    training_data = create_synthetic_training_data(2000)
    
    # 保存训练数据
    training_data_path = data_dir / "synthetic_training_data.json"
    with open(training_data_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"   Saved training data to: {training_data_path}")
    
    # 2. 创建经过训练的模型
    print("\n2. Creating and training models...")
    models = create_trained_models(training_data)
    
    model_path = model_dir / "wass_models.pth"
    torch.save(models, model_path)
    print(f"   Saved trained models to: {model_path}")
    
    # 打印训练结果摘要
    if "metadata" in models and "performance_predictor" in models["metadata"]:
        perf_meta = models["metadata"]["performance_predictor"]
        if isinstance(perf_meta, dict) and "final_r2" in perf_meta:
            print(f"   PerformancePredictor R²: {perf_meta['final_r2']:.4f}")
            print(f"   Training samples: {perf_meta['training_samples']}")
            print(f"   Prediction diversity: {perf_meta['prediction_std']:.2f}")
    
    # 3. 验证训练好的模型
    print("\n3. Validating trained model...")
    validation_results = validate_trained_model(str(model_path), training_data)
    
    if validation_results:
        # 将验证结果添加到模型元数据中
        if "metadata" in models:
            models["metadata"]["validation_results"] = validation_results
            # 重新保存包含验证结果的模型
            torch.save(models, model_path)
            print(f"   Updated model file with validation results")
    
    # 4. 创建知识库
    print("\n3. Creating knowledge base...")
    if HAS_AI_MODULES:
        kb = create_dummy_knowledge_base(training_data)
        
        if kb is not None:
            kb_path = data_dir / "knowledge_base.pkl"
            kb.save_knowledge_base(kb_path)
            print(f"   Saved knowledge base to: {kb_path}")
        else:
            print("   Failed to create knowledge base")
    else:
        print("   Skipping knowledge base creation (AI modules not available)")
    
    # 4. 测试调度器初始化
    print("\n4. Testing scheduler initialization...")
    
    if HAS_AI_MODULES:
        try:
            # 测试启发式调度器
            heuristic_scheduler = create_scheduler("WASS (Heuristic)")
            print(f"   ✓ {heuristic_scheduler.name} initialized successfully")
            
            # 测试DRL调度器
            drl_scheduler = create_scheduler("WASS-DRL (w/o RAG)", model_path=str(model_path))
            print(f"   ✓ {drl_scheduler.name} initialized successfully")
            
            # 测试RAG调度器
            rag_scheduler = create_scheduler(
                "WASS-RAG", 
                model_path=str(model_path),
                knowledge_base_path=str(kb_path) if 'kb_path' in locals() else None
            )
            print(f"   ✓ {rag_scheduler.name} initialized successfully")
            
        except Exception as e:
            print(f"   ✗ Error testing schedulers: {e}")
    else:
        print("   Skipping scheduler tests (AI modules not available)")
    
    print("\n=== Initialization Complete ===")
    print("Your AI models and knowledge base are ready for experiments!")
    print(f"\nGenerated files:")
    print(f"  - {model_path}: Pre-trained neural network models")
    print(f"  - {training_data_path}: Synthetic training data")
    if HAS_AI_MODULES and 'kb_path' in locals():
        print(f"  - {kb_path}: RAG knowledge base")
    
    print(f"\nYou can now run the experiment framework:")
    print(f"  python experiments/real_experiment_framework.py")

def simulate_training_process():
    """模拟AI模型的训练过程（用于演示）"""
    
    print("\n=== Simulating WASS-RAG Training Process ===")
    
    # 模拟三个训练阶段
    stages = [
        ("Stage 1: Knowledge Base Seeding", 30),
        ("Stage 2: Performance Predictor Training", 45), 
        ("Stage 3: RAG-DRL Agent Training", 60)
    ]
    
    for stage_name, duration in stages:
        print(f"\n{stage_name}...")
        
        # 模拟训练进度
        for i in range(10):
            progress = (i + 1) * 10
            time.sleep(duration / 100)  # 短暂延迟模拟训练时间
            print(f"  Progress: {progress}%", end='\r')
        
        print(f"  Progress: 100% - {stage_name} completed")
    
    print(f"\n✓ All training stages completed successfully!")
    print(f"  Total simulated training time: {sum(d for _, d in stages)} seconds")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--simulate-training":
        simulate_training_process()
    
    # 初始化模型和知识库
    initialize_ai_models_and_kb()
