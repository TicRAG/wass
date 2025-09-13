#!/usr/bin/env python3
"""
WASS-RAG系统基准测试脚本
用于验证重构后的系统在不同CCR值下的性能
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def run_ccr_benchmark():
    """运行CCR基准测试"""
    print("🚀 WASS-RAG系统CCR基准测试")
    print("=" * 60)
    
    try:
        from scripts.workflow_generator import WorkflowGenerator
        from src.performance_predictor import PerformancePredictor
        from src.drl_agent import DQNAgent
        from src.ai_schedulers import WASSRAGScheduler
        
        # 测试配置
        ccr_values = [0.1, 1.0, 5.0, 10.0]
        task_counts = [20, 50, 100]
        patterns = ['montage', 'ligo', 'cybershake']
        
        results = []
        
        # 初始化组件
        predictor = PerformancePredictor()
        drl_agent = DQNAgent(state_dim=50, action_dim=4)
        node_names = ["node1", "node2", "node3", "node4"]
        
        total_start_time = time.time()
        
        for pattern in patterns:
            print(f"\n📊 测试模式: {pattern.upper()}")
            print("-" * 40)
            
            for task_count in task_counts:
                for ccr in ccr_values:
                    try:
                        # 生成工作流
                        generator = WorkflowGenerator(
                            output_dir="data/benchmark_workflows", 
                            ccr=ccr
                        )
                        workflow_files = generator.generate_workflow_set(pattern, [task_count])
                        
                        if workflow_files:
                            # 从JSON文件加载工作流信息
                            with open(workflow_files[0], 'r') as f:
                                data = json.load(f)
                                tasks = data['workflow']['tasks']
                                files = data['workflow']['files']
                                
                                # 计算实际CCR
                                total_compute = sum(task['flops'] for task in tasks)
                                total_comm = sum(
                                    edge.get('data_size', 0) 
                                    for task in tasks 
                                    for dep in task.get('dependencies', [])
                                )
                                actual_ccr = total_comm / total_compute if total_compute > 0 else 0
                                
                                # 初始化RAG调度器
                                scheduler = WASSRAGScheduler(
                                    drl_agent=drl_agent,
                                    node_names=node_names,
                                    predictor=predictor
                                )
                                
                                result = {
                                    'pattern': pattern,
                                    'task_count': task_count,
                                    'target_ccr': ccr,
                                    'actual_ccr': actual_ccr,
                                    'total_compute': total_compute,
                                    'total_comm': total_comm,
                                    'workflow_file': workflow_files[0],
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                                results.append(result)
                                
                                print(f"  ✅ {pattern} | {task_count}任务 | CCR={ccr} -> 实际CCR={actual_ccr:.2f}")
                        
                    except Exception as e:
                        print(f"  ❌ {pattern} | {task_count}任务 | CCR={ccr} | 错误: {e}")
        
        total_time = time.time() - total_start_time
        
        # 保存结果
        results_file = "data/benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': len(results),
                    'total_time': total_time,
                    'patterns_tested': patterns,
                    'ccr_values': ccr_values,
                    'task_counts': task_counts
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 基准测试完成")
        print(f"   总测试数: {len(results)}")
        print(f"   总耗时: {total_time:.2f}秒")
        print(f"   结果保存: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        return False

def analyze_results():
    """分析基准测试结果"""
    print("\n📈 结果分析")
    print("=" * 60)
    
    try:
        with open('data/benchmark_results.json', 'r') as f:
            data = json.load(f)
        
        results = data['results']
        
        # 按模式分组分析
        patterns = {}
        for result in results:
            pattern = result['pattern']
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(result)
        
        for pattern, pattern_results in patterns.items():
            print(f"\n📊 {pattern.upper()} 模式分析:")
            
            # 计算CCR偏差
            for task_count in [20, 50, 100]:
                task_results = [r for r in pattern_results if r['task_count'] == task_count]
                if task_results:
                    avg_deviation = sum(abs(r['actual_ccr'] - r['target_ccr']) for r in task_results) / len(task_results)
                    print(f"  {task_count}任务: 平均CCR偏差={avg_deviation:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 结果分析失败: {e}")
        return False

def main():
    """主函数"""
    if run_ccr_benchmark():
        analyze_results()
    else:
        print("❌ 基准测试未成功完成")

if __name__ == "__main__":
    main()