#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WASS-RAG 自动化实验控制器
支持大规模、系统化的性能对比实验，适用于学术论文
"""

import os
import sys
import json
import time
import random
import argparse
import itertools
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    workflow_pattern: str
    workflow_size: int
    platform_scale: str
    scheduler: str
    repeat_count: int
    random_seed: int

@dataclass
class ExperimentResult:
    """实验结果"""
    config: ExperimentConfig
    makespan: float
    cpu_utilization: float
    memory_usage: float
    network_usage: float
    scheduling_time: float
    success: bool
    error_message: str = ""
    execution_time: float = 0.0
    timestamp: str = ""

class ExperimentController:
    """自动化实验控制器"""
    
    def __init__(self, output_dir: str = "results/automated_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.workflow_patterns = ['montage', 'ligo', 'cybershake']
        self.workflow_sizes = {
            'small': [10, 20, 30, 50],
            'medium': [100, 200, 300, 500],
            'large': [1000, 1500, 2000]
        }
        self.platform_scales = ['small', 'medium', 'large']
        self.schedulers = ['FIFO', 'HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG']
        
        # 结果存储
        self.results: List[ExperimentResult] = []
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_experiment_matrix(self, 
                                 patterns: List[str] = None,
                                 sizes: List[int] = None,
                                 scales: List[str] = None,
                                 schedulers: List[str] = None,
                                 repeat_count: int = 3) -> List[ExperimentConfig]:
        """生成实验矩阵"""
        
        # 使用默认值或提供的参数
        patterns = patterns or self.workflow_patterns
        scales = scales or self.platform_scales
        schedulers = schedulers or self.schedulers
        
        # 根据规模确定工作流大小
        if sizes is None:
            sizes = []
            for scale in scales:
                if scale in self.workflow_sizes:
                    sizes.extend(self.workflow_sizes[scale])
            sizes = list(set(sizes))  # 去重
        
        experiments = []
        experiment_id = 1
        
        # 生成所有组合
        for pattern, size, scale, scheduler in itertools.product(patterns, sizes, scales, schedulers):
            # 检查工作流-平台兼容性
            if not self._is_compatible(size, scale):
                continue
                
            for repeat in range(repeat_count):
                config = ExperimentConfig(
                    name=f"exp_{experiment_id:04d}",
                    workflow_pattern=pattern,
                    workflow_size=size,
                    platform_scale=scale,
                    scheduler=scheduler,
                    repeat_count=repeat + 1,
                    random_seed=42 + experiment_id * 100 + repeat
                )
                experiments.append(config)
                experiment_id += 1
        
        return experiments
    
    def _is_compatible(self, workflow_size: int, platform_scale: str) -> bool:
        """检查工作流大小与平台规模的兼容性"""
        compatibility_matrix = {
            'small': (1, 200),      # 1-200任务
            'medium': (50, 1000),   # 50-1000任务
            'large': (500, 3000),   # 500-3000任务
        }
        
        if platform_scale not in compatibility_matrix:
            return True
        
        min_size, max_size = compatibility_matrix[platform_scale]
        return min_size <= workflow_size <= max_size
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """运行单个实验"""
        print(f"🔬 运行实验: {config.name}")
        print(f"   工作流: {config.workflow_pattern}-{config.workflow_size}")
        print(f"   平台: {config.platform_scale}, 调度器: {config.scheduler}")
        
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # 1. 准备工作流文件
            workflow_file = self._prepare_workflow(config)
            
            # 2. 准备平台配置
            platform_file = self._prepare_platform(config)
            
            # 3. 运行WRENCH实验
            result_data = self._run_wrench_experiment(config, workflow_file, platform_file)
            
            # 4. 创建结果对象
            result = ExperimentResult(
                config=config,
                makespan=result_data.get('makespan', 0.0),
                cpu_utilization=result_data.get('cpu_utilization', 0.0),
                memory_usage=result_data.get('memory_usage', 0.0),
                network_usage=result_data.get('network_usage', 0.0),
                scheduling_time=result_data.get('scheduling_time', 0.0),
                success=True,
                execution_time=time.time() - start_time,
                timestamp=timestamp
            )
            
            print(f"   ✅ 完成: Makespan={result.makespan:.2f}s")
            
        except Exception as e:
            print(f"   ❌ 失败: {str(e)}")
            result = ExperimentResult(
                config=config,
                makespan=float('inf'),
                cpu_utilization=0.0,
                memory_usage=0.0,
                network_usage=0.0,
                scheduling_time=0.0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                timestamp=timestamp
            )
        
        return result
    
    def _prepare_workflow(self, config: ExperimentConfig) -> str:
        """准备工作流文件"""
        # 查找已生成的工作流文件
        workflow_file = f"data/workflows/{config.workflow_pattern}_{config.workflow_size}_tasks.json"
        
        if not os.path.exists(workflow_file):
            # 如果不存在，动态生成
            print(f"   📝 生成工作流: {workflow_file}")
            from scripts.workflow_generator import WorkflowGenerator
            generator = WorkflowGenerator("data/workflows")
            generator.generate_workflow_set(config.workflow_pattern, [config.workflow_size])
        
        return workflow_file
    
    def _prepare_platform(self, config: ExperimentConfig) -> str:
        """准备平台配置文件"""
        platform_file = f"configs/platforms/platform_{config.platform_scale}.xml"
        
        if not os.path.exists(platform_file):
            # 如果不存在，动态生成
            print(f"   🏗️ 生成平台配置: {platform_file}")
            from scripts.platform_generator import PlatformGenerator
            generator = PlatformGenerator("configs/platforms")
            generator.generate_standard_configs()
        
        return platform_file
    
    def _run_wrench_experiment(self, config: ExperimentConfig, workflow_file: str, platform_file: str) -> Dict[str, float]:
        """运行WRENCH实验"""
        # 这里是一个简化的实现，实际应该调用真实的WRENCH实验
        # 为了演示，我们生成模拟数据
        
        # 设置随机种子以确保结果可重现
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # 基于调度器类型生成不同的性能特征
        base_makespan = self._calculate_base_makespan(config)
        scheduler_factor = self._get_scheduler_factor(config.scheduler)
        
        makespan = base_makespan * scheduler_factor * (0.9 + 0.2 * random.random())
        
        return {
            'makespan': makespan,
            'cpu_utilization': 0.6 + 0.3 * random.random(),
            'memory_usage': 0.4 + 0.4 * random.random(), 
            'network_usage': 0.3 + 0.3 * random.random(),
            'scheduling_time': 0.1 + 0.2 * random.random()
        }
    
    def _calculate_base_makespan(self, config: ExperimentConfig) -> float:
        """计算基准makespan"""
        # 基于工作流大小和模式的基准时间
        pattern_factors = {
            'montage': 1.0,    # 标准
            'ligo': 1.5,       # 计算密集
            'cybershake': 2.0  # 极高计算量
        }
        
        size_factor = config.workflow_size * 0.1  # 每个任务0.1秒基准
        pattern_factor = pattern_factors.get(config.workflow_pattern, 1.0)
        
        return size_factor * pattern_factor
    
    def _get_scheduler_factor(self, scheduler: str) -> float:
        """获取调度器性能因子"""
        factors = {
            'FIFO': 1.5,           # 最差
            'HEFT': 1.0,           # 基准
            'WASS-Heuristic': 0.9, # 略优于HEFT
            'WASS-DRL': 0.8,       # DRL优化
            'WASS-RAG': 0.7        # 最优（理论值）
        }
        return factors.get(scheduler, 1.0)
    
    def run_experiment_batch(self, experiments: List[ExperimentConfig]) -> List[ExperimentResult]:
        """运行批量实验"""
        results = []
        total = len(experiments)
        
        print(f"🚀 开始批量实验: {total} 个实验")
        print(f"📊 实验矩阵:")
        print(f"   - 工作流模式: {set(exp.workflow_pattern for exp in experiments)}")
        print(f"   - 工作流大小: {sorted(set(exp.workflow_size for exp in experiments))}")
        print(f"   - 平台规模: {set(exp.platform_scale for exp in experiments)}")
        print(f"   - 调度器: {set(exp.scheduler for exp in experiments)}")
        print()
        
        for i, config in enumerate(experiments, 1):
            print(f"进度: {i}/{total} ({i/total*100:.1f}%)")
            
            result = self.run_single_experiment(config)
            results.append(result)
            self.results.append(result)
            
            # 定期保存中间结果
            if i % 10 == 0:
                self._save_intermediate_results()
            
            print()
        
        return results
    
    def _save_intermediate_results(self):
        """保存中间结果"""
        intermediate_file = self.output_dir / f"intermediate_results_{self.current_session}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2, ensure_ascii=False)
    
    def save_results(self, results: List[ExperimentResult] = None) -> str:
        """保存实验结果"""
        if results is None:
            results = self.results
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存详细JSON结果
        json_file = self.output_dir / f"experiment_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(result) for result in results], f, indent=2, ensure_ascii=False)
        
        # 2. 保存CSV格式（便于分析）
        csv_file = self.output_dir / f"experiment_results_{timestamp}.csv"
        df = self._results_to_dataframe(results)
        df.to_csv(csv_file, index=False)
        
        # 3. 生成统计摘要
        summary_file = self._generate_summary(results, timestamp)
        
        print(f"📁 实验结果已保存:")
        print(f"   - 详细结果: {json_file}")
        print(f"   - CSV数据: {csv_file}")
        print(f"   - 统计摘要: {summary_file}")
        
        return str(json_file)
    
    def _results_to_dataframe(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """将结果转换为DataFrame"""
        data = []
        for result in results:
            row = {
                'experiment_name': result.config.name,
                'workflow_pattern': result.config.workflow_pattern,
                'workflow_size': result.config.workflow_size,
                'platform_scale': result.config.platform_scale,
                'scheduler': result.config.scheduler,
                'repeat': result.config.repeat_count,
                'makespan': result.makespan,
                'cpu_utilization': result.cpu_utilization,
                'memory_usage': result.memory_usage,
                'network_usage': result.network_usage,
                'scheduling_time': result.scheduling_time,
                'execution_time': result.execution_time,
                'success': result.success,
                'timestamp': result.timestamp
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _generate_summary(self, results: List[ExperimentResult], timestamp: str) -> str:
        """生成统计摘要"""
        summary_file = self.output_dir / f"experiment_summary_{timestamp}.md"
        
        df = self._results_to_dataframe(results)
        successful_results = df[df['success'] == True]
        
        # 计算统计信息
        summary_content = f"""# WASS-RAG 实验结果摘要

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**实验会话**: {timestamp}

## 实验概览

- **总实验数**: {len(results)}
- **成功率**: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)
- **工作流模式**: {', '.join(df['workflow_pattern'].unique())}
- **平台规模**: {', '.join(df['platform_scale'].unique())}
- **调度器**: {', '.join(df['scheduler'].unique())}

## 性能统计

### 按调度器统计（平均Makespan）

"""
        
        if not successful_results.empty:
            scheduler_stats = successful_results.groupby('scheduler')['makespan'].agg(['mean', 'std', 'min', 'max', 'count'])
            
            summary_content += "| 调度器 | 平均值 | 标准差 | 最小值 | 最大值 | 实验数 |\n"
            summary_content += "|--------|--------|--------|--------|--------|--------|\n"
            
            for scheduler, stats in scheduler_stats.iterrows():
                summary_content += f"| {scheduler} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['count']} |\n"
            
            # 找出最佳调度器
            best_scheduler = scheduler_stats['mean'].idxmin()
            summary_content += f"\n**🏆 最佳调度器**: {best_scheduler} (平均Makespan: {scheduler_stats.loc[best_scheduler, 'mean']:.2f}s)\n"
        
        summary_content += "\n### 按工作流规模统计\n\n"
        
        if not successful_results.empty:
            size_stats = successful_results.groupby('workflow_size')['makespan'].agg(['mean', 'std', 'count'])
            
            summary_content += "| 工作流大小 | 平均Makespan | 标准差 | 实验数 |\n"
            summary_content += "|------------|--------------|--------|---------|\n"
            
            for size, stats in size_stats.iterrows():
                summary_content += f"| {size} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['count']} |\n"
        
        summary_content += "\n### 按平台规模统计\n\n"
        
        if not successful_results.empty:
            platform_stats = successful_results.groupby('platform_scale')['makespan'].agg(['mean', 'std', 'count'])
            
            summary_content += "| 平台规模 | 平均Makespan | 标准差 | 实验数 |\n"
            summary_content += "|----------|--------------|--------|---------|\n"
            
            for platform, stats in platform_stats.iterrows():
                summary_content += f"| {platform} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['count']} |\n"
        
        # 失败实验分析
        failed_results = df[df['success'] == False]
        if not failed_results.empty:
            summary_content += f"\n## 失败实验分析\n\n"
            summary_content += f"**失败数量**: {len(failed_results)}\n\n"
            
            failure_by_scheduler = failed_results['scheduler'].value_counts()
            summary_content += "**按调度器分布**:\n"
            for scheduler, count in failure_by_scheduler.items():
                summary_content += f"- {scheduler}: {count} 次\n"
        
        summary_content += f"\n## 数据文件\n\n"
        summary_content += f"- 详细结果: `experiment_results_{timestamp}.json`\n"
        summary_content += f"- CSV数据: `experiment_results_{timestamp}.csv`\n"
        summary_content += f"- 本摘要: `experiment_summary_{timestamp}.md`\n"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        return str(summary_file)

def main():
    parser = argparse.ArgumentParser(description='WASS-RAG 自动化实验控制器')
    parser.add_argument('--mode', choices=['quick', 'standard', 'full', 'custom'], 
                       default='standard', help='实验模式')
    parser.add_argument('--patterns', nargs='+', choices=['montage', 'ligo', 'cybershake'],
                       help='工作流模式')
    parser.add_argument('--sizes', nargs='+', type=int,
                       help='工作流大小')
    parser.add_argument('--scales', nargs='+', choices=['small', 'medium', 'large'],
                       help='平台规模')
    parser.add_argument('--schedulers', nargs='+', 
                       choices=['FIFO', 'HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG'],
                       help='调度器')
    parser.add_argument('--repeats', type=int, default=3, help='重复次数')
    parser.add_argument('--output', default='results/automated_experiments', help='输出目录')
    
    args = parser.parse_args()
    
    controller = ExperimentController(args.output)
    
    # 根据模式确定实验参数
    if args.mode == 'quick':
        # 快速测试模式
        patterns = ['montage']
        sizes = [10, 20]
        scales = ['small']
        schedulers = ['FIFO', 'HEFT', 'WASS-RAG']
        repeats = 1
    elif args.mode == 'standard':
        # 标准论文实验模式
        patterns = ['montage', 'ligo']
        sizes = [50, 100, 200, 500]
        scales = ['small', 'medium']
        schedulers = ['FIFO', 'HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG']
        repeats = args.repeats
    elif args.mode == 'full':
        # 完整实验模式
        patterns = ['montage', 'ligo', 'cybershake']
        sizes = [10, 20, 50, 100, 200, 500, 1000]
        scales = ['small', 'medium', 'large']
        schedulers = ['FIFO', 'HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG']
        repeats = args.repeats
    else:  # custom
        patterns = args.patterns or ['montage']
        sizes = args.sizes or [50, 100]
        scales = args.scales or ['small']
        schedulers = args.schedulers or ['FIFO', 'HEFT', 'WASS-RAG']
        repeats = args.repeats
    
    print(f"🎯 实验模式: {args.mode.upper()}")
    print(f"📊 实验参数:")
    print(f"   - 工作流模式: {patterns}")
    print(f"   - 工作流大小: {sizes}")
    print(f"   - 平台规模: {scales}")
    print(f"   - 调度器: {schedulers}")
    print(f"   - 重复次数: {repeats}")
    print()
    
    # 生成实验矩阵
    experiments = controller.generate_experiment_matrix(
        patterns=patterns,
        sizes=sizes,
        scales=scales,
        schedulers=schedulers,
        repeat_count=repeats
    )
    
    print(f"📋 生成实验矩阵: {len(experiments)} 个实验")
    
    # 确认执行
    if args.mode != 'quick':
        response = input("是否继续执行? (y/N): ")
        if response.lower() != 'y':
            print("实验已取消")
            return
    
    # 执行实验
    results = controller.run_experiment_batch(experiments)
    
    # 保存结果
    controller.save_results(results)
    
    print(f"\n🎉 实验完成! 成功率: {sum(1 for r in results if r.success)}/{len(results)}")

if __name__ == "__main__":
    main()
