#!/usr/bin/env python3
"""
从实验结果中提取学习数据的脚本
从HEFT和WASS-Heuristic的调度结果中提取高质量的学习案例
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_base.wrench_cases import WrenchKnowledgeCaseMinimal

class LearningDataExtractor:
    """从实验结果中提取学习数据"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        
    def load_experiment_results(self, filename: str) -> Dict[str, Any]:
        """加载实验结果文件"""
        file_path = self.results_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"结果文件不存在: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_scheduler_data(self, results: Dict[str, Any], 
                             scheduler_names: List[str]) -> List[Dict[str, Any]]:
        """从结果中提取指定调度器的数据"""
        learning_cases = []
        
        for result in results.get('results', []):
            scheduler_name = result.get('scheduler_name')
            
            if scheduler_name not in scheduler_names:
                continue
                
            workflow_id = result.get('workflow_id')
            makespan = result.get('makespan', 0)
            task_count = result.get('task_count', 0)
            
            # 跳过无效结果
            if makespan <= 0 or makespan == float('inf'):
                continue
                
            # 提取每个任务的调度决策
            scheduling_decisions = result.get('scheduling_decisions', [])
            task_execution_times = result.get('task_execution_times', {})
            
            for decision in scheduling_decisions:
                task_id = decision.get('task_id')
                chosen_node = decision.get('chosen_node')
                execution_time = decision.get('execution_time', 0)
                
                # 获取任务执行时间
                task_time = task_execution_times.get(task_id, execution_time)
                
                # 计算任务计算量（简化：假设计算量与执行时间成正比）
                # 这里可以根据实际节点速度进行调整
                task_flops = task_time * 1e9  # 假设1秒 = 1GFLOP
                
                # 创建学习案例
                case = {
                    'workflow_id': workflow_id,
                    'task_id': task_id,
                    'scheduler_type': scheduler_name,
                    'chosen_node': chosen_node,
                    'task_flops': task_flops,
                    'task_execution_time': task_time,
                    'workflow_makespan': makespan,
                    'task_count': task_count,
                    'quality_score': 1.0 / makespan if makespan > 0 else 0.0
                }
                
                learning_cases.append(case)
        
        return learning_cases
    
    def filter_high_quality_cases(self, cases: List[Dict[str, Any]], 
                                top_percentile: float = 0.8) -> List[Dict[str, Any]]:
        """过滤高质量的学习案例"""
        if not cases:
            return cases
            
        # 按质量分数排序
        cases_sorted = sorted(cases, key=lambda x: x['quality_score'], reverse=True)
        
        # 选择前top_percentile的案例
        cutoff_idx = max(1, int(len(cases_sorted) * top_percentile))
        high_quality_cases = cases_sorted[:cutoff_idx]
        
        return high_quality_cases
    
    def balance_dataset(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """平衡不同调度器的数据"""
        from collections import defaultdict
        
        # 按调度器分组
        scheduler_groups = defaultdict(list)
        for case in cases:
            scheduler_groups[case['scheduler_type']].append(case)
        
        # 找到最小的组大小
        min_size = min(len(cases) for cases in scheduler_groups.values())
        
        # 从每个组中随机采样相同数量的案例
        balanced_cases = []
        for scheduler_type, scheduler_cases in scheduler_groups.items():
            import random
            sampled = random.sample(scheduler_cases, min(min_size, len(scheduler_cases)))
            balanced_cases.extend(sampled)
        
        return balanced_cases
    
    def save_learning_data(self, cases: List[Dict[str, Any]], output_path: str):
        """保存学习数据"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(cases)} 个学习案例到 {output_file}")
    
    def generate_summary(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成数据摘要"""
        from collections import Counter
        
        if not cases:
            return {}
        
        scheduler_counts = Counter(case['scheduler_type'] for case in cases)
        node_counts = Counter(case['chosen_node'] for case in cases)
        
        avg_makespan = np.mean([case['workflow_makespan'] for case in cases])
        avg_task_time = np.mean([case['task_execution_time'] for case in cases])
        
        return {
            'total_cases': len(cases),
            'scheduler_distribution': dict(scheduler_counts),
            'node_distribution': dict(node_counts),
            'avg_workflow_makespan': float(avg_makespan),
            'avg_task_execution_time': float(avg_task_time)
        }

def main():
    """主函数"""
    extractor = LearningDataExtractor()
    
    try:
        # 加载实验结果
        print("正在加载实验结果...")
        results = extractor.load_experiment_results('wrench_experiments/detailed_results.json')
        
        # 提取HEFT和WASS-Heuristic的数据
        print("正在提取学习数据...")
        scheduler_names = ['HEFT', 'WASS-Heuristic']
        cases = extractor.extract_scheduler_data(results, scheduler_names)
        
        print(f"提取了 {len(cases)} 个原始案例")
        
        # 过滤高质量案例
        high_quality_cases = extractor.filter_high_quality_cases(cases, top_percentile=0.8)
        print(f"过滤后得到 {len(high_quality_cases)} 个高质量案例")
        
        # 平衡数据集
        balanced_cases = extractor.balance_dataset(high_quality_cases)
        print(f"平衡后得到 {len(balanced_cases)} 个案例")
        
        # 保存学习数据
        output_path = "data/heuristic_learning_data.json"
        extractor.save_learning_data(balanced_cases, output_path)
        
        # 生成摘要
        summary = extractor.generate_summary(balanced_cases)
        print("\n数据摘要:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())