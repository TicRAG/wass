#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文图表生成脚本
根据实验结果生成ACM标准的图表
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_results():
    """加载实验结果数据"""
    results_file = "results/wrench_experiments/detailed_results.json"
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"实验结果文件不存在: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['results']

def create_scheduler_performance_comparison(results):
    """创建调度器性能比较图表"""
    # 按调度器分组统计makespan
    scheduler_stats = defaultdict(list)
    
    for result in results:
        scheduler = result['scheduler_name']
        makespan = result['makespan']
        scheduler_stats[scheduler].append(makespan)
    
    # 计算平均值和标准差
    schedulers = []
    avg_makespans = []
    std_makespans = []
    
    # 按照预期性能排序
    scheduler_order = ['FIFO', 'HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG']
    
    for scheduler in scheduler_order:
        if scheduler in scheduler_stats:
            schedulers.append(scheduler)
            avg_makespans.append(np.mean(scheduler_stats[scheduler]))
            std_makespans.append(np.std(scheduler_stats[scheduler]))
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(schedulers, avg_makespans, yerr=std_makespans, capsize=5, 
                  color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
    
    ax.set_xlabel('调度器')
    ax.set_ylabel('平均完成时间 (秒)')
    ax.set_title('不同调度器性能比较')
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, (avg, std) in enumerate(zip(avg_makespans, std_makespans)):
        ax.text(i, avg + std + 0.1, f'{avg:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/scheduler_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_makespan_distribution(results):
    """创建完成时间分布图表"""
    # 按调度器分组
    scheduler_data = defaultdict(list)
    
    for result in results:
        scheduler = result['scheduler_name']
        makespan = result['makespan']
        scheduler_data[scheduler].append(makespan)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 按照预期性能排序
    scheduler_order = ['FIFO', 'HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG']
    data_to_plot = [scheduler_data[scheduler] for scheduler in scheduler_order if scheduler in scheduler_data]
    labels = [scheduler for scheduler in scheduler_order if scheduler in scheduler_data]
    
    box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # 设置颜色
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('调度器')
    ax.set_ylabel('完成时间 (秒)')
    ax.set_title('不同调度器完成时间分布')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/makespan_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cpu_utilization_chart(results):
    """创建CPU利用率图表"""
    # 计算每个调度器的平均CPU利用率
    scheduler_utilization = defaultdict(list)
    
    for result in results:
        scheduler = result['scheduler_name']
        # 计算所有节点的平均CPU利用率
        avg_util = np.mean(list(result['cpu_utilization'].values()))
        scheduler_utilization[scheduler].append(avg_util)
    
    # 计算平均值
    schedulers = []
    avg_utils = []
    
    scheduler_order = ['FIFO', 'HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG']
    
    for scheduler in scheduler_order:
        if scheduler in scheduler_utilization:
            schedulers.append(scheduler)
            avg_utils.append(np.mean(scheduler_utilization[scheduler]) * 100)  # 转换为百分比
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(schedulers, avg_utils, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
    
    ax.set_xlabel('调度器')
    ax.set_ylabel('平均CPU利用率 (%)')
    ax.set_title('不同调度器CPU利用率比较')
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, avg in enumerate(avg_utils):
        ax.text(i, avg + 1, f'{avg:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/cpu_utilization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_task_scaling_analysis(results):
    """创建任务规模分析图表"""
    # 按任务数量和调度器分组
    task_scaling = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        scheduler = result['scheduler_name']
        task_count = result['task_count']
        makespan = result['makespan']
        task_scaling[task_count][scheduler].append(makespan)
    
    # 计算每个任务规模下各调度器的平均完成时间
    task_counts = sorted(task_scaling.keys())
    scheduler_order = ['FIFO', 'HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG']
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scheduler in scheduler_order:
        avg_makespans = []
        for task_count in task_counts:
            if scheduler in task_scaling[task_count]:
                avg_makespans.append(np.mean(task_scaling[task_count][scheduler]))
            else:
                avg_makespans.append(0)
        
        # 只绘制有数据的调度器
        if any(avg_makespans):
            ax.plot(task_counts, avg_makespans, marker='o', label=scheduler, linewidth=2, markersize=8)
    
    ax.set_xlabel('任务数量')
    ax.set_ylabel('平均完成时间 (秒)')
    ax.set_title('不同任务规模下的调度器性能')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/task_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_improvement_chart(results):
    """创建性能提升图表"""
    # 计算相对于FIFO的性能提升
    scheduler_makespans = defaultdict(list)
    
    for result in results:
        scheduler = result['scheduler_name']
        makespan = result['makespan']
        scheduler_makespans[scheduler].append(makespan)
    
    # 计算平均完成时间
    avg_makespans = {scheduler: np.mean(makespans) 
                     for scheduler, makespans in scheduler_makespans.items()}
    
    if 'FIFO' not in avg_makespans:
        print("警告: 缺少FIFO基准数据")
        return
    
    fifo_avg = avg_makespans['FIFO']
    scheduler_order = ['HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG']
    
    improvements = []
    labels = []
    
    for scheduler in scheduler_order:
        if scheduler in avg_makespans:
            improvement = (fifo_avg - avg_makespans[scheduler]) / fifo_avg * 100
            improvements.append(improvement)
            labels.append(scheduler)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, improvements, color=['#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
    
    ax.set_xlabel('调度器')
    ax.set_ylabel('性能提升 (%)')
    ax.set_title('相对于FIFO调度器的性能提升')
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, improvement in enumerate(improvements):
        ax.text(i, improvement + 1, f'{improvement:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/performance_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("开始生成论文图表...")
    
    # 确保charts目录存在
    os.makedirs('charts', exist_ok=True)
    
    try:
        # 加载实验结果
        results = load_experiment_results()
        print(f"加载了 {len(results)} 个实验结果")
        
        # 生成各类图表
        print("生成调度器性能比较图表...")
        create_scheduler_performance_comparison(results)
        
        print("生成完成时间分布图表...")
        create_makespan_distribution(results)
        
        print("生成CPU利用率图表...")
        create_cpu_utilization_chart(results)
        
        print("生成任务规模分析图表...")
        create_task_scaling_analysis(results)
        
        print("生成性能提升图表...")
        create_performance_improvement_chart(results)
        
        print("所有图表生成完成!")
        print("\n生成的图表:")
        for chart_file in os.listdir('charts'):
            if chart_file.endswith('.png'):
                print(f"  - {chart_file}")
                
    except Exception as e:
        print(f"图表生成过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
