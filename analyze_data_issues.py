#!/usr/bin/env python3
"""
数据分析脚本：检查实验数据和图表数据的一致性
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

def load_experiment_data():
    """加载原始实验数据"""
    with open('experiments/results/real_experiments/experiment_results.json', 'r') as f:
        return json.load(f)

def load_chart_data():
    """加载图表数据"""
    chart_data = {}
    
    # 加载所有图表数据文件
    chart_files = [
        'performance_improvement_data.json',
        'scheduler_radar_data.json', 
        'scheduling_comparison_data.json',
        'stability_analysis_data.json',
        'performance_summary_data.json'
    ]
    
    for file in chart_files:
        try:
            with open(f'charts/charts/output/data/{file}', 'r') as f:
                chart_data[file.replace('_data.json', '')] = json.load(f)
        except FileNotFoundError:
            print(f"警告：找不到文件 {file}")
    
    return chart_data

def analyze_experiment_statistics(experiments):
    """分析原始实验数据的统计信息"""
    print("=== 原始实验数据分析 ===")
    
    # 按调度算法分组
    by_scheduler = defaultdict(list)
    for exp in experiments:
        scheduler = exp['scheduling_method']
        by_scheduler[scheduler].append(exp)
    
    print(f"总实验数量: {len(experiments)}")
    print(f"调度算法数量: {len(by_scheduler)}")
    
    for scheduler, exps in by_scheduler.items():
        print(f"\n{scheduler}: {len(exps)} 个实验")
        
        # 分析执行时间
        exec_times = [exp['execution_time'] for exp in exps]
        print(f"  执行时间: 平均={np.mean(exec_times):.2f}, 标准差={np.std(exec_times):.2f}")
        print(f"  执行时间范围: [{np.min(exec_times):.2f}, {np.max(exec_times):.2f}]")
        
        # 分析CPU利用率
        cpu_utils = [exp['cpu_utilization'] for exp in exps]
        print(f"  CPU利用率: 平均={np.mean(cpu_utils):.3f}, 标准差={np.std(cpu_utils):.3f}")
        
        # 分析数据局部性
        data_locality = [exp['data_locality_score'] for exp in exps]
        print(f"  数据局部性: 平均={np.mean(data_locality):.3f}, 标准差={np.std(data_locality):.3f}")

def analyze_performance_improvement_data(experiments, chart_data):
    """分析性能改善数据的问题"""
    print("\n=== 性能改善热力图数据分析 ===")
    
    if 'performance_improvement' not in chart_data:
        print("错误：找不到性能改善数据")
        return
    
    heatmap_data = chart_data['performance_improvement']['data']
    improvement_matrix = heatmap_data['improvement_matrix']
    
    print(f"热力图矩阵形状: {len(improvement_matrix)} x {len(improvement_matrix[0])}")
    print(f"集群大小: {heatmap_data['cluster_sizes']}")
    print(f"工作流大小: {heatmap_data['workflow_sizes']}")
    
    # 检查所有值是否相同
    flat_values = [val for row in improvement_matrix for val in row]
    unique_values = set(flat_values)
    print(f"唯一改善值数量: {len(unique_values)}")
    print(f"改善值: {unique_values}")
    
    if len(unique_values) <= 2:
        print("⚠️  警告：性能改善数据几乎没有变化，可能是计算错误")
        
        # 手动计算实际的性能改善
        print("\n手动计算实际性能改善:")
        
        # 按集群大小和工作流大小分组
        by_config = defaultdict(lambda: defaultdict(list))
        for exp in experiments:
            cluster_size = exp['cluster_size']
            workflow_size = exp['workflow_spec']['task_count']
            scheduler = exp['scheduling_method']
            
            by_config[(cluster_size, workflow_size)][scheduler].append(exp['execution_time'])
        
        # 计算WASS-RAG相对于HEFT的改善
        for (cluster_size, workflow_size), schedulers in by_config.items():
            if 'WASS-RAG' in schedulers and 'HEFT' in schedulers:
                wass_avg = np.mean(schedulers['WASS-RAG'])
                heft_avg = np.mean(schedulers['HEFT'])
                improvement = ((heft_avg - wass_avg) / heft_avg) * 100
                print(f"  集群{cluster_size}, 工作流{workflow_size}: {improvement:.2f}% 改善")

def analyze_radar_data(experiments, chart_data):
    """分析雷达图数据的问题"""
    print("\n=== 雷达图数据分析 ===")
    
    if 'scheduler_radar' not in chart_data:
        print("错误：找不到雷达图数据")
        return
    
    radar_data = chart_data['scheduler_radar']['data']['metrics']
    
    for scheduler, metrics in radar_data.items():
        print(f"\n{scheduler}:")
        for metric, value in metrics.items():
            if isinstance(value, float) and np.isnan(value):
                print(f"  {metric}: NaN ⚠️")
            else:
                print(f"  {metric}: {value}")
        
        # 检查是否有NaN值
        nan_count = sum(1 for v in metrics.values() if isinstance(v, float) and np.isnan(v))
        if nan_count > 0:
            print(f"  ⚠️  警告：{scheduler}有{nan_count}个NaN值")

def analyze_boxplot_data(experiments, chart_data):
    """分析箱型图数据的问题"""
    print("\n=== 箱型图数据分析 ===")
    
    if 'stability_analysis' not in chart_data:
        print("错误：找不到箱型图数据")
        return
    
    boxplot_data = chart_data['stability_analysis']['data']['complex_scenario_data']
    
    # 按调度器分组
    by_scheduler = defaultdict(list)
    for item in boxplot_data:
        by_scheduler[item['scheduler']].append(item)
    
    print(f"箱型图数据点数量: {len(boxplot_data)}")
    print(f"调度器数量: {len(by_scheduler)}")
    
    for scheduler, items in by_scheduler.items():
        makespans = [item['makespan'] for item in items]
        print(f"\n{scheduler}: {len(items)} 个数据点")
        print(f"  Makespan范围: [{np.min(makespans):.2f}, {np.max(makespans):.2f}]")
        print(f"  Makespan标准差: {np.std(makespans):.2f}")
        
        # 检查是否有重复数据
        unique_makespans = len(set(makespans))
        if unique_makespans < len(makespans):
            print(f"  ⚠️  警告：{len(makespans) - unique_makespans}个重复的makespan值")

def main():
    """主函数"""
    print("开始分析实验数据和图表数据...")
    
    # 加载数据
    experiments = load_experiment_data()
    chart_data = load_chart_data()
    
    # 分析原始实验数据
    analyze_experiment_statistics(experiments)
    
    # 分析各个图表的数据问题
    analyze_performance_improvement_data(experiments, chart_data)
    analyze_radar_data(experiments, chart_data)
    analyze_boxplot_data(experiments, chart_data)
    
    print("\n=== 分析完成 ===")
    print("请检查上述警告信息，这些可能是数据处理中的问题。")

if __name__ == "__main__":
    main()
