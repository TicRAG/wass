#!/usr/bin/env python3
"""
修复图表生成中的数据问题
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_actual_schedulers():
    """分析实际存在的调度器"""
    with open('experiments/results/real_experiments/experiment_results.json', 'r') as f:
        experiments = json.load(f)  # 直接是列表格式
    
    # 获取所有调度器
    schedulers = set()
    for exp in experiments:
        schedulers.add(exp['scheduling_method'])
    
    print("实际调度器列表:")
    for scheduler in sorted(schedulers):
        print(f"  - {scheduler}")
    
    return sorted(schedulers)

def analyze_performance_variations():
    """分析性能变化"""
    with open('experiments/results/real_experiments/experiment_results.json', 'r') as f:
        experiments = json.load(f)  # 直接是列表格式
    
    df = pd.DataFrame(experiments)
    
    # 展开workflow_spec字段
    df['workflow_size'] = df['workflow_spec'].apply(lambda x: x['task_count'])
    
    print("\n=== 性能变化分析 ===")
    
    # 按配置分组检查
    configs = df.groupby(['scheduling_method', 'cluster_size', 'workflow_size'])
    
    print(f"配置组合数量: {len(configs)}")
    
    # 检查每个配置的性能范围
    for name, group in configs:
        scheduler, cluster_size, workflow_size = name
        makespans = group['makespan'].values
        
        print(f"{scheduler} (cluster:{cluster_size}, workflow:{workflow_size}):")
        print(f"  样本数: {len(makespans)}")
        print(f"  Makespan范围: [{min(makespans):.2f}, {max(makespans):.2f}]")
        print(f"  标准差: {np.std(makespans):.2f}")
        
        if len(set(makespans)) == 1:
            print(f"  ⚠️  所有值相同: {makespans[0]:.2f}")
        print()

def check_data_locality_calculation():
    """检查数据局部性计算"""
    with open('experiments/results/real_experiments/experiment_results.json', 'r') as f:
        experiments = json.load(f)  # 直接是列表格式
    
    print("=== 数据局部性分析 ===")
    
    # 按调度器分组
    by_scheduler = defaultdict(list)
    for exp in experiments:
        scheduler = exp['scheduling_method']
        locality = exp.get('data_locality_score', None)
        by_scheduler[scheduler].append(locality)
    
    for scheduler, localities in by_scheduler.items():
        unique_values = set(localities)
        print(f"{scheduler}:")
        print(f"  数据局部性值: {unique_values}")
        print(f"  是否所有值相同: {len(unique_values) == 1}")
        if len(unique_values) == 1:
            print(f"  ⚠️  固定值: {list(unique_values)[0]}")
        print()

def generate_suggested_fixes():
    """生成建议的修复方案"""
    print("=== 建议的修复方案 ===")
    
    schedulers = analyze_actual_schedulers()
    
    print("\n1. 修复调度器列表:")
    print("   将 paper_charts.py 中的:")
    print("   schedulers = ['HEFT', 'WASS-DRL', 'WASS-RAG']")
    print("   替换为:")
    print(f"   schedulers = {schedulers[:3]}  # 选择前3个作为主要对比")
    
    print("\n2. 修复数据聚合问题:")
    print("   - 确保每个配置组合的数据被正确分组")
    print("   - 使用多个实验运行的平均值而非单个值")
    print("   - 检查makespan计算是否考虑了所有因素")
    
    print("\n3. 修复数据局部性计算:")
    print("   - 检查实验框架中的data_locality_score计算")
    print("   - 确保不是硬编码的固定值")
    print("   - 可能需要基于实际的数据传输模式重新计算")
    
    print("\n4. 修复热力图数据:")
    print("   - 检查性能改善的计算公式")
    print("   - 确保不同配置产生不同的改善值")
    print("   - 可能需要重新计算基线性能")

def main():
    """主函数"""
    print("开始分析图表数据问题...")
    
    try:
        analyze_actual_schedulers()
        analyze_performance_variations()
        check_data_locality_calculation()
        generate_suggested_fixes()
        
        print("\n✅ 分析完成！请根据建议修复相关问题。")
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
