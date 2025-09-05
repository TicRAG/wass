#!/usr/bin/env python3
"""
WASS-RAG 快速演示脚本
在没有完整AI依赖的情况下演示实验框架
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def create_demo_experiment():
    """创建演示实验配置"""
    
    print("=== WASS-RAG Demo Experiment ===")
    print("This demo shows how the experiment framework works")
    print("(Running in simulation mode without full AI dependencies)\n")
    
    # 创建输出目录
    output_dir = Path("results/demo_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模拟实验配置
    config = {
        "name": "WASS-RAG Demo Evaluation",
        "description": "Demonstration of WASS-RAG scheduling framework",
        "workflow_sizes": [10, 20, 49],  # 包含论文中的49任务工作流
        "scheduling_methods": [
            "FIFO",                    # 传统Slurm基准
            "HEFT",                    # 学术界经典算法
            "WASS (Heuristic)",        # 我们的启发式基线
            "WASS-DRL (w/o RAG)",      # 标准DRL基线  
            "WASS-RAG"                 # 我们的完整方法
        ],
        "cluster_sizes": [4, 8, 16],
        "repetitions": 2,  # 减少重复次数以加快演示
        "ai_model_path": "models/wass_models.pth",
        "knowledge_base_path": "data/knowledge_base.pkl"
    }
    
    # 计算总实验数
    total_experiments = (len(config["workflow_sizes"]) * 
                        len(config["scheduling_methods"]) * 
                        len(config["cluster_sizes"]) * 
                        config["repetitions"])
    
    print(f"Configuration:")
    print(f"  Workflow sizes: {config['workflow_sizes']}")
    print(f"  Scheduling methods: {len(config['scheduling_methods'])}")
    for i, method in enumerate(config["scheduling_methods"], 1):
        print(f"    {i}. {method}")
    print(f"  Cluster sizes: {config['cluster_sizes']}")
    print(f"  Total experiments: {total_experiments}")
    
    return config, output_dir

def simulate_single_experiment(workflow_size, method, cluster_size, rep):
    """模拟单个实验的执行"""
    
    # 基础性能（模拟WRENCH仿真结果）
    base_makespan = (workflow_size * 2.0) / cluster_size  # 简化的makespan计算
    
    # 不同方法的性能因子（基于论文数据）
    method_factors = {
        "FIFO": 1.0,                    # 基准
        "HEFT": 0.79,                   # 21% improvement over FIFO
        "WASS (Heuristic)": 0.75,       # 25% improvement 
        "WASS-DRL (w/o RAG)": 0.67,     # 33% improvement
        "WASS-RAG": 0.62               # 38% improvement (论文结果)
    }
    
    improvement_factor = method_factors.get(method, 1.0)
    
    # 添加一些随机变化来模拟真实实验
    import random
    random.seed(hash(f"{workflow_size}_{method}_{cluster_size}_{rep}"))
    noise = random.uniform(0.95, 1.05)
    
    makespan = base_makespan * improvement_factor * noise
    
    # 计算其他指标
    cpu_utilization = min(0.9, 0.5 + (1 - improvement_factor) * 0.8)
    data_locality = min(1.0, 0.4 + (1 - improvement_factor) * 0.9)
    throughput = cluster_size / makespan
    energy = makespan * cluster_size * 100
    
    return {
        "workflow_size": workflow_size,
        "method": method,
        "cluster_size": cluster_size,
        "repetition": rep,
        "makespan": round(makespan, 2),
        "cpu_utilization": round(cpu_utilization, 3),
        "data_locality_score": round(data_locality, 3),
        "throughput": round(throughput, 2),
        "energy_consumption": round(energy, 1),
        "improvement_over_fifo": round((1 - improvement_factor) * 100, 1),
        "timestamp": datetime.now().isoformat()
    }

def run_demo_experiments(config, output_dir):
    """运行演示实验"""
    
    print(f"\nRunning experiments...")
    
    results = []
    total_experiments = (len(config["workflow_sizes"]) * 
                        len(config["scheduling_methods"]) * 
                        len(config["cluster_sizes"]) * 
                        config["repetitions"])
    
    experiment_count = 0
    start_time = time.time()
    
    for workflow_size in config["workflow_sizes"]:
        for method in config["scheduling_methods"]:
            for cluster_size in config["cluster_sizes"]:
                for rep in range(config["repetitions"]):
                    experiment_count += 1
                    
                    # 运行单个实验
                    result = simulate_single_experiment(workflow_size, method, cluster_size, rep)
                    results.append(result)
                    
                    # 显示进度
                    progress = (experiment_count / total_experiments) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / experiment_count) * (total_experiments - experiment_count)
                    
                    print(f"  Progress: {progress:.1f}% ({experiment_count}/{total_experiments}), "
                          f"ETA: {eta:.1f}s", end='\r')
                    
                    # 模拟一些处理时间
                    time.sleep(0.1)
    
    print(f"\nCompleted all experiments in {time.time() - start_time:.2f}s")
    
    return results

def generate_analysis(results, config, output_dir):
    """生成分析报告"""
    
    print(f"\nGenerating analysis...")
    
    # 基础统计
    analysis = {
        "experiment_summary": {
            "total_experiments": len(results),
            "config": config,
            "generated_at": datetime.now().isoformat()
        },
        "performance_by_method": {},
        "case_study_49_tasks": {},
        "scalability_analysis": {}
    }
    
    # 按方法分组分析
    for method in config["scheduling_methods"]:
        method_results = [r for r in results if r["method"] == method]
        
        if method_results:
            avg_makespan = sum(r["makespan"] for r in method_results) / len(method_results)
            avg_improvement = sum(r["improvement_over_fifo"] for r in method_results) / len(method_results)
            avg_cpu_util = sum(r["cpu_utilization"] for r in method_results) / len(method_results)
            avg_data_locality = sum(r["data_locality_score"] for r in method_results) / len(method_results)
            
            analysis["performance_by_method"][method] = {
                "avg_makespan": round(avg_makespan, 2),
                "improvement_over_fifo": round(avg_improvement, 1),
                "cpu_utilization": round(avg_cpu_util * 100, 1),
                "data_locality": round(avg_data_locality * 100, 1),
                "sample_count": len(method_results)
            }
    
    # 49任务案例研究（对应论文Table 3）
    task_49_results = [r for r in results if r["workflow_size"] == 49]
    if task_49_results:
        case_study = {}
        
        for method in config["scheduling_methods"]:
            method_49_results = [r for r in task_49_results if r["method"] == method]
            if method_49_results:
                avg_makespan = sum(r["makespan"] for r in method_49_results) / len(method_49_results)
                avg_improvement = sum(r["improvement_over_fifo"] for r in method_49_results) / len(method_49_results)
                
                case_study[method] = {
                    "makespan": round(avg_makespan, 2),
                    "improvement_over_fifo": round(avg_improvement, 1)
                }
        
        analysis["case_study_49_tasks"] = case_study
    
    # 保存结果
    results_file = output_dir / "demo_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    analysis_file = output_dir / "demo_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    return analysis

def print_paper_tables(analysis):
    """打印论文用的表格"""
    
    print(f"\n=== Paper Tables (Demo Results) ===")
    
    # Table 2: 调度方法比较
    print(f"\nTable 2: Scheduling Method Comparison")
    print(f"{'Method':<20} {'Makespan (s)':<12} {'Improvement':<12} {'CPU Util':<10} {'Data Locality':<12}")
    print("-" * 70)
    
    performance = analysis.get("performance_by_method", {})
    for method, data in performance.items():
        print(f"{method:<20} {data['avg_makespan']:<12} {data['improvement_over_fifo']:<12}% "
              f"{data['cpu_utilization']:<10}% {data['data_locality']:<12}%")
    
    # Table 3: 49任务案例研究
    case_study = analysis.get("case_study_49_tasks", {})
    if case_study:
        print(f"\nTable 3: 49-task Genomics Workflow Case Study")
        print(f"{'Method':<20} {'Makespan (s)':<12} {'Improvement over Slurm':<20}")
        print("-" * 55)
        
        for method, data in case_study.items():
            improvement_label = f"{data['improvement_over_fifo']}%" if method != "FIFO" else "-"
            print(f"{method:<20} {data['makespan']:<12} {improvement_label:<20}")
    
    print(f"\n=== Key Findings (Demo) ===")
    
    if "WASS-RAG" in performance:
        wass_rag_improvement = performance["WASS-RAG"]["improvement_over_fifo"]
        print(f"• WASS-RAG achieves {wass_rag_improvement}% makespan reduction over traditional scheduling")
    
    if "WASS (Heuristic)" in performance and "WASS-RAG" in performance:
        heuristic_improvement = performance["WASS (Heuristic)"]["improvement_over_fifo"]
        rag_improvement = performance["WASS-RAG"]["improvement_over_fifo"]
        additional_improvement = rag_improvement - heuristic_improvement
        print(f"• RAG enhancement provides additional {additional_improvement:.1f}% improvement over heuristic baseline")
    
    if "WASS-DRL (w/o RAG)" in performance and "WASS-RAG" in performance:
        drl_improvement = performance["WASS-DRL (w/o RAG)"]["improvement_over_fifo"]
        rag_improvement = performance["WASS-RAG"]["improvement_over_fifo"]
        rag_benefit = rag_improvement - drl_improvement
        print(f"• Knowledge-guided RAG provides {rag_benefit:.1f}% improvement over standard DRL")

def main():
    """主演示函数"""
    
    # 1. 创建演示实验
    config, output_dir = create_demo_experiment()
    
    # 2. 运行实验
    results = run_demo_experiments(config, output_dir)
    
    # 3. 生成分析
    analysis = generate_analysis(results, config, output_dir)
    
    # 4. 显示结果
    print_paper_tables(analysis)
    
    print(f"\n=== Demo Complete ===")
    print(f"Results saved to: {output_dir}")
    print(f"  - demo_results.json: Raw experimental data")
    print(f"  - demo_analysis.json: Analysis and statistics")
    
    print(f"\nThis demo shows the experimental framework structure.")
    print(f"With full AI dependencies installed, you would get:")
    print(f"  - Real GNN-based state encoding")
    print(f"  - Actual DRL policy decisions") 
    print(f"  - RAG knowledge base retrieval")
    print(f"  - Explainable AI decision reasoning")
    
    print(f"\nTo install full dependencies:")
    print(f"  pip install -r requirements.txt")
    print(f"  python scripts/initialize_ai_models.py")
    print(f"  python experiments/real_experiment_framework.py")

if __name__ == "__main__":
    main()
