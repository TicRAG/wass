#!/usr/bin/env python3
"""
为WASS-RAG创建扩展的JSON知识库
从现有的样本案例生成更多的知识案例
"""

import json
import random
import numpy as np
from pathlib import Path

def create_extended_rag_knowledge(num_cases=2500):
    """创建扩展的RAG知识库"""
    # 基础案例模板
    base_cases = [
        {
            "workflow_id": "workflow_template_1",
            "task_id": "task_template_1",
            "scheduler_type": "HEFT",
            "chosen_node": "ComputeHost4",
            "task_execution_time": 1.5,
            "workflow_makespan": 1.5
        },
        {
            "workflow_id": "workflow_template_2", 
            "task_id": "task_template_2",
            "scheduler_type": "FIFO",
            "chosen_node": "ComputeHost1",
            "task_execution_time": 2.0,
            "workflow_makespan": 2.0
        },
        {
            "workflow_id": "workflow_template_3",
            "task_id": "task_template_3", 
            "scheduler_type": "Random",
            "chosen_node": "ComputeHost2",
            "task_execution_time": 1.8,
            "workflow_makespan": 1.8
        }
    ]
    
    # 节点性能配置
    node_capacities = {
        "ComputeHost1": 2.0,
        "ComputeHost2": 3.0,
        "ComputeHost3": 2.5,
        "ComputeHost4": 4.0
    }
    
    schedulers = ["HEFT", "FIFO", "Random"]
    nodes = list(node_capacities.keys())
    
    # 生成扩展案例
    extended_cases = []
    
    for i in range(num_cases):  # 生成指定数量的案例
        # 随机选择调度器和节点
        scheduler = random.choice(schedulers)
        node = random.choice(nodes)
        node_capacity = node_capacities[node]
        
        # 生成任务特征
        task_flops = random.uniform(1e9, 10e9)  # 1-10 GFlops
        exec_time = task_flops / (node_capacity * 1e9)
        
        # 添加一些噪声和调度器特性
        if scheduler == "HEFT":
            # HEFT倾向于选择高性能节点
            if node == "ComputeHost4":
                exec_time *= 0.9  # 10%性能提升
        elif scheduler == "FIFO":
            # FIFO可能有排队延迟
            exec_time *= random.uniform(1.0, 1.3)
        else:  # Random
            # 随机调度可能选择次优节点
            exec_time *= random.uniform(1.0, 1.5)
        
        case = {
            "workflow_id": f"workflow_{i}",
            "task_id": f"task_{i}_0",
            "scheduler_type": scheduler,
            "chosen_node": node,
            "task_execution_time": exec_time,
            "workflow_makespan": exec_time,
            "task_flops": task_flops,
            "node_capacity": node_capacity,
            "performance_ratio": task_flops / (exec_time * node_capacity * 1e9)
        }
        
        extended_cases.append(case)
    
    # 创建完整的知识库数据
    knowledge_base = {
        "metadata": {
            "total_cases": len(extended_cases),
            "schedulers": schedulers,
            "nodes": nodes,
            "node_capacities": node_capacities,
            "generated_at": "2025-09-11 12:00:00",
            "description": "Extended RAG knowledge base for WASS-RAG experiment"
        },
        "cases": extended_cases
    }
    
    # 保存到文件
    output_path = Path("data/extended_rag_knowledge.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 扩展RAG知识库已创建: {output_path}")
    print(f"📊 包含 {len(extended_cases)} 个案例")
    
    # 打印统计信息
    scheduler_counts = {}
    node_counts = {}
    
    for case in extended_cases:
        scheduler = case['scheduler_type']
        node = case['chosen_node']
        
        scheduler_counts[scheduler] = scheduler_counts.get(scheduler, 0) + 1
        node_counts[node] = node_counts.get(node, 0) + 1
    
    print("\n📈 案例分布:")
    print("调度器分布:")
    for scheduler, count in scheduler_counts.items():
        print(f"  {scheduler}: {count} 个案例")
    
    print("节点分布:")
    for node, count in node_counts.items():
        print(f"  {node}: {count} 个案例")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='创建扩展的RAG知识库')
    parser.add_argument('--num_cases', type=int, default=2500, help='生成的案例数量')
    
    args = parser.parse_args()
    
    create_extended_rag_knowledge(args.num_cases)
