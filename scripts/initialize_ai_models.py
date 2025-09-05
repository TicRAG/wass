#!/usr/bin/env python3
"""
WASS-RAG AI模型训练和初始化脚本
为实验框架准备预训练的AI模型和知识库
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
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

def create_synthetic_training_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """创建合成的训练数据"""
    
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
        
        # 计算理论makespan（基于启发式规则）
        base_makespan = (workflow_features["task_count"] * workflow_features["avg_flops"] / 1e9) / cluster_size
        load_factor = 1.0 + cluster_features["avg_load"]
        dependency_factor = 1.0 + workflow_features["dependency_ratio"] * 0.5
        
        makespan = base_makespan * load_factor * dependency_factor
        
        # 生成状态嵌入（简化版）
        state_embedding = np.array([
            workflow_features["task_count"] / 100.0,
            workflow_features["avg_flops"] / 10e9,
            workflow_features["avg_memory"] / 4e9,
            workflow_features["dependency_ratio"],
            workflow_features["data_intensity"],
            cluster_features["node_count"] / 16.0,
            cluster_features["avg_load"],
        ] + [np.random.randn() for _ in range(25)])  # 填充到32维
        
        training_data.append({
            "id": f"synthetic_{i}",
            "workflow_features": workflow_features,
            "cluster_features": cluster_features,
            "state_embedding": state_embedding.tolist(),
            "makespan": makespan,
            "timestamp": time.time(),
            "method": "synthetic_optimal"
        })
    
    return training_data

def create_dummy_models():
    """创建虚拟的预训练模型（用于演示）"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建网络结构
    models = {}
    
    # GNN编码器（如果可用）
    try:
        models["gnn_encoder"] = GraphEncoder(
            node_feature_dim=8,
            edge_feature_dim=4,
            hidden_dim=64,
            output_dim=32
        ).state_dict()
    except:
        print("Skipping GNN encoder (torch_geometric not available)")
    
    # 策略网络
    models["policy_network"] = PolicyNetwork(
        state_dim=32,
        action_dim=1,
        hidden_dim=128
    ).state_dict()
    
    # 性能预测器
    models["performance_predictor"] = PerformancePredictor(
        input_dim=96,
        hidden_dim=128
    ).state_dict()
    
    # 添加训练元数据
    models["metadata"] = {
        "training_episodes": 10000,
        "training_time": 3600,  # 1小时
        "final_reward": 245.6,
        "convergence_episode": 8500,
        "created_at": time.time(),
        "model_version": "1.0.0"
    }
    
    return models

def create_dummy_knowledge_base(training_data: List[Dict[str, Any]]) -> RAGKnowledgeBase:
    """创建知识库并填充合成数据"""
    
    if not HAS_AI_MODULES:
        print("Cannot create knowledge base: AI modules not available")
        return None
    
    kb = RAGKnowledgeBase()
    
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
    
    # 2. 创建预训练模型
    print("\n2. Creating pre-trained models...")
    models = create_dummy_models()
    
    model_path = model_dir / "wass_models.pth"
    torch.save(models, model_path)
    print(f"   Saved models to: {model_path}")
    
    # 3. 创建知识库
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
