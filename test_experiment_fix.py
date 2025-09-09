#!/usr/bin/env python3
"""
测试修复后的实验框架
"""

import sys
import os
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'experiments'))

from experiments.real_experiment_framework import WassExperimentRunner, ExperimentConfig

def test_experiment_variability():
    """测试实验的变异性"""
    
    print("🧪 测试实验数据变异性...")
    
    # 创建小规模测试配置
    config = ExperimentConfig(
        name="Variability Test",
        description="Test data variability",
        workflow_sizes=[10],
        scheduling_methods=["FIFO", "HEFT", "WASS-RAG"],
        cluster_sizes=[4],
        repetitions=5,  # 5次重复以检查变异性
        output_dir="test_results"
    )
    
    runner = WassExperimentRunner(config)
    
    # 运行测试
    runner.run_all_experiments()
    
    # 分析结果
    results = runner.results
    
    print(f"\n📊 结果分析:")
    print(f"总实验数: {len(results)}")
    
    # 按调度方法分组检查变异性
    by_method = {}
    for result in results:
        method = result.scheduling_method
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(result.makespan)
    
    print("\n📈 Makespan变异性分析:")
    for method, makespans in by_method.items():
        std = np.std(makespans)
        mean = np.mean(makespans)
        cv = std / mean if mean > 0 else 0  # 变异系数
        
        print(f"{method}:")
        print(f"  平均makespan: {mean:.2f}")
        print(f"  标准差: {std:.2f}")
        print(f"  变异系数: {cv:.3f}")
        print(f"  范围: [{min(makespans):.2f}, {max(makespans):.2f}]")
        
        if cv < 0.01:
            print(f"  ⚠️  变异性太低，可能仍有问题")
        else:
            print(f"  ✅ 变异性正常")
        print()
    
    # 检查数据局部性变异性
    print("📍 数据局部性变异性分析:")
    by_method_locality = {}
    for result in results:
        method = result.scheduling_method
        if method not in by_method_locality:
            by_method_locality[method] = []
        by_method_locality[method].append(result.data_locality_score)
    
    for method, localities in by_method_locality.items():
        std = np.std(localities)
        mean = np.mean(localities)
        unique_values = len(set(localities))
        
        print(f"{method}:")
        print(f"  平均数据局部性: {mean:.3f}")
        print(f"  标准差: {std:.4f}")
        print(f"  唯一值数量: {unique_values}")
        
        if std < 0.001:
            print(f"  ⚠️  数据局部性仍为固定值")
        else:
            print(f"  ✅ 数据局部性有变异")
        print()

if __name__ == "__main__":
    test_experiment_variability()
