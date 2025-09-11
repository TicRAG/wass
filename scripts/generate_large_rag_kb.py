#!/usr/bin/env python3
"""
WASS-RAG 大规模知识库生成器 (方案A)
生成2500个高质量RAG知识案例
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime

def generate_large_rag_knowledge(num_cases=2500):
    """生成大规模RAG知识库"""
    
    print(f"🚀 开始生成 {num_cases} 个RAG知识案例...")
    
    # 节点配置
    node_configs = {
        "ComputeHost1": {"capacity": 2.0, "memory": 16},
        "ComputeHost2": {"capacity": 3.0, "memory": 24}, 
        "ComputeHost3": {"capacity": 2.5, "memory": 20},
        "ComputeHost4": {"capacity": 4.0, "memory": 32}
    }
    
    # 调度器类型及其特性
    scheduler_types = {
        "FIFO": {"efficiency": 0.7, "variance": 0.3},
        "HEFT": {"efficiency": 0.85, "variance": 0.15},
        "Random": {"efficiency": 0.6, "variance": 0.4},
        "WASS-Heuristic": {"efficiency": 0.9, "variance": 0.1},
        "Optimal": {"efficiency": 1.0, "variance": 0.05}
    }
    
    # 工作流模式
    workflow_patterns = ["montage", "ligo", "cybershake", "sipht", "genome"]
    
    all_cases = []
    scheduler_counts = {}
    node_counts = {}
    
    for case_id in range(num_cases):
        # 随机选择参数
        scheduler = random.choice(list(scheduler_types.keys()))
        node = random.choice(list(node_configs.keys()))
        pattern = random.choice(workflow_patterns)
        
        # 生成任务特征
        task_size = random.choice(["small", "medium", "large"])
        if task_size == "small":
            task_flops = random.uniform(1e8, 1e9)
            workflow_size = random.randint(5, 20)
        elif task_size == "medium":
            task_flops = random.uniform(1e9, 10e9)
            workflow_size = random.randint(20, 100)
        else:  # large
            task_flops = random.uniform(10e9, 100e9)
            workflow_size = random.randint(100, 500)
        
        # 计算执行时间（考虑调度器效率）
        node_capacity = node_configs[node]["capacity"]
        scheduler_efficiency = scheduler_types[scheduler]["efficiency"]
        base_exec_time = task_flops / (node_capacity * 1e9)
        actual_exec_time = base_exec_time / scheduler_efficiency
        
        # 添加随机性
        variance = scheduler_types[scheduler]["variance"]
        noise = random.uniform(1 - variance, 1 + variance)
        actual_exec_time *= noise
        
        # 计算工作流makespan（简化估算）
        critical_path_length = workflow_size * 0.3  # 假设关键路径占30%
        workflow_makespan = critical_path_length * actual_exec_time
        
        # 计算性能比率
        optimal_time = base_exec_time / scheduler_types["Optimal"]["efficiency"]
        performance_ratio = optimal_time / actual_exec_time
        
        # 数据局部性评分
        data_locality_score = random.uniform(0.1, 1.0)
        if scheduler == "WASS-Heuristic":
            data_locality_score *= 1.2  # WASS-Heuristic考虑数据局部性
        
        # 创建案例
        case = {
            "case_id": f"case_{case_id:06d}",
            "workflow_id": f"{pattern}_workflow_{case_id // 10}",
            "task_id": f"task_{case_id % workflow_size}",
            "scheduler_type": scheduler,
            "chosen_node": node,
            "workflow_pattern": pattern,
            "workflow_size": workflow_size,
            "task_flops": task_flops,
            "task_execution_time": actual_exec_time,
            "workflow_makespan": workflow_makespan,
            "node_capacity": node_capacity,
            "node_memory": node_configs[node]["memory"],
            "performance_ratio": performance_ratio,
            "data_locality_score": data_locality_score,
            "scheduler_efficiency": scheduler_efficiency,
            "task_size_category": task_size,
            "timestamp": datetime.now().isoformat(),
            "features": {
                "cpu_intensive": task_flops > 5e9,
                "memory_intensive": node_configs[node]["memory"] > 20,
                "io_intensive": pattern in ["genome", "montage"],
                "critical_path_task": random.random() > 0.7,
                "data_dependent": data_locality_score > 0.7
            }
        }
        
        all_cases.append(case)
        
        # 统计
        scheduler_counts[scheduler] = scheduler_counts.get(scheduler, 0) + 1
        node_counts[node] = node_counts.get(node, 0) + 1
        
        # 进度显示
        if (case_id + 1) % 500 == 0:
            print(f"已生成 {case_id + 1}/{num_cases} 个案例...")
    
    return all_cases, scheduler_counts, node_counts

def save_knowledge_base(cases, output_path="data/extended_rag_knowledge_v2.json"):
    """保存知识库到文件"""
    
    knowledge_base = {
        "metadata": {
            "version": "2.0",
            "description": "扩展的WASS-RAG知识库 (方案A)",
            "total_cases": len(cases),
            "generated_at": datetime.now().isoformat(),
            "features": [
                "多种工作流模式",
                "5种调度器类型", 
                "详细性能指标",
                "数据局部性建模",
                "任务特征分类"
            ]
        },
        "cases": cases
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 扩展RAG知识库已保存: {output_path}")
    return output_path

def analyze_distribution(scheduler_counts, node_counts):
    """分析案例分布"""
    print("\n📊 案例分布分析:")
    
    print("\n调度器分布:")
    for scheduler, count in sorted(scheduler_counts.items()):
        percentage = count / sum(scheduler_counts.values()) * 100
        print(f"  {scheduler}: {count} 个案例 ({percentage:.1f}%)")
    
    print("\n节点分布:")
    for node, count in sorted(node_counts.items()):
        percentage = count / sum(node_counts.values()) * 100
        print(f"  {node}: {count} 个案例 ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='生成大规模RAG知识库')
    parser.add_argument('--num_cases', type=int, default=2500, 
                       help='生成的案例数量')
    parser.add_argument('--output', default='data/extended_rag_knowledge_v2.json',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    # 生成知识库
    cases, scheduler_counts, node_counts = generate_large_rag_knowledge(args.num_cases)
    
    # 保存文件
    output_path = save_knowledge_base(cases, args.output)
    
    # 分析分布
    analyze_distribution(scheduler_counts, node_counts)
    
    print(f"\n🎉 大规模RAG知识库生成完成!")
    print(f"📁 文件位置: {output_path}")
    print(f"📊 案例总数: {len(cases)}")
    
    # 生成质量报告
    file_size = Path(output_path).stat().st_size / 1024 / 1024  # MB
    print(f"💾 文件大小: {file_size:.2f} MB")

if __name__ == "__main__":
    main()
