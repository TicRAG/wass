#!/usr/bin/env python3
"""
使用真实HEFT和WASS-Heuristic案例更新RAG知识库
"""

import json
import os
import shutil
from typing import Dict, List, Any

def load_real_cases() -> List[Dict]:
    """加载真实案例"""
    real_cases_path = "data/heuristic_only_real_cases.json"
    
    if not os.path.exists(real_cases_path):
        print("错误：真实案例文件不存在")
        return []
    
    with open(real_cases_path, 'r') as f:
        cases = json.load(f)
    
    print(f"已加载 {len(cases)} 个真实案例")
    return cases

def convert_to_rag_format(cases: List[Dict]) -> List[Dict]:
    """将案例转换为RAG知识库格式"""
    rag_cases = []
    
    for case in cases:
        # 创建RAG格式的案例
        rag_case = {
            "workflow_id": case["workflow_id"],
            "task_id": case["task_id"],
            "scheduler_type": case["scheduler_type"],
            "chosen_node": case["chosen_node"],
            "task_execution_time": case["task_execution_time"],
            "makespan": case["makespan"],
            "timestamp": case["timestamp"],
            "task_features": case["task_features"],
            "platform_features": case["platform_features"],
            "workflow_embedding": [0.0] * 64,  # 占位符
            "case_quality_score": 1.0,  # 真实案例质量分数为1
            "is_real_case": True  # 标记为真实案例
        }
        rag_cases.append(rag_case)
    
    return rag_cases

def create_real_only_knowledge_base():
    """创建仅使用真实案例的知识库"""
    
    # 加载真实案例
    print("正在加载真实HEFT和WASS-Heuristic案例...")
    real_cases = load_real_cases()
    
    if not real_cases:
        print("无法加载真实案例")
        return
    
    # 转换为RAG格式
    rag_cases = convert_to_rag_format(real_cases)
    
    # 创建知识库
    knowledge_base = {
        "metadata": {
            "version": "1.0",
            "description": "仅使用HEFT和WASS-Heuristic真实案例的RAG知识库",
            "total_cases": len(rag_cases),
            "heft_cases": len([c for c in rag_cases if c["scheduler_type"] == "HEFT"]),
            "wass_cases": len([c for c in rag_cases if c["scheduler_type"] == "WASS-Heuristic"]),
            "real_cases_only": True,
            "creation_date": "2024-01-01"
        },
        "cases": rag_cases
    }
    
    # 保存知识库
    output_path = "data/real_heuristic_kb.json"
    os.makedirs("data", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(knowledge_base, f, indent=2)
    
    print(f"\n真实案例知识库创建完成!")
    print(f"总案例数: {knowledge_base['metadata']['total_cases']}")
    print(f"HEFT案例: {knowledge_base['metadata']['heft_cases']}")
    print(f"WASS-Heuristic案例: {knowledge_base['metadata']['wass_cases']}")
    print(f"数据已保存到: {output_path}")
    
    # 更新WASS-RAG配置
    update_rag_config(output_path)

def update_rag_config(kb_path: str):
    """更新RAG配置以使用新的知识库"""
    
    config_path = "configs/rag.yaml"
    if not os.path.exists(config_path):
        print(f"警告：配置文件 {config_path} 不存在")
        return
    
    # 读取现有配置
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # 更新知识库路径，保持正确的缩进
    new_lines = []
    for line in lines:
        if "knowledge_base_path:" in line:
            # 保持缩进，确保在rag块内
            new_lines.append(f"  knowledge_base_path: {kb_path}\n")
        else:
            new_lines.append(line)
    
    # 写入更新后的配置
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"已更新RAG配置，使用新的知识库: {kb_path}")

def create_training_dataset():
    """创建训练数据集"""
    
    real_cases_path = "data/heuristic_only_real_cases.json"
    
    if not os.path.exists(real_cases_path):
        print("错误：真实案例文件不存在")
        return
    
    with open(real_cases_path, 'r') as f:
        cases = json.load(f)
    
    # 创建训练数据集
    training_data = []
    
    for case in cases:
        # 构建训练样本
        features = []
        
        # 任务特征
        task_feats = case["task_features"]
        features.extend([
            task_feats.get("task_flops", 0),
            task_feats.get("task_memory", 0),
            task_feats.get("task_inputs", 0),
            task_feats.get("task_outputs", 0),
            task_feats.get("task_dependencies", 0)
        ])
        
        # 平台特征
        platform_feats = case["platform_features"]
        features.extend([
            platform_feats.get("num_nodes", 0),
            platform_feats.get("avg_flops", 0),
            platform_feats.get("avg_memory", 0)
        ])
        
        # 标准化特征
        features = [float(f) for f in features]
        
        # 标签：任务执行时间
        label = case["task_execution_time"]
        
        training_sample = {
            "features": features,
            "label": label,
            "scheduler_type": case["scheduler_type"],
            "makespan": case["makespan"]
        }
        
        training_data.append(training_sample)
    
    # 保存训练数据集
    training_path = "data/heuristic_training_dataset.json"
    with open(training_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"训练数据集创建完成: {len(training_data)} 个样本")
    print(f"数据已保存到: {training_path}")

if __name__ == "__main__":
    print("=== 使用真实HEFT和WASS-Heuristic案例更新RAG知识库 ===")
    
    # 创建仅使用真实案例的知识库
    create_real_only_knowledge_base()
    
    # 创建训练数据集
    create_training_dataset()
    
    print("\n=== 更新完成 ===")
    print("下一步可以运行：")
    print("1. python scripts/train_rag_wrench.py - 重新训练RAG模型")
    print("2. python run_complete_experiment.sh - 运行完整实验")