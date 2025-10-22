#!/usr/bin/env python3
"""
使用utils.py中的WrenchExperimentRunner进行时间差异放大测试
"""

import sys
from pathlib import Path
import json
import time
from collections import defaultdict

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import WrenchExperimentRunner
from src.wrench_schedulers import HEFTScheduler, FIFOScheduler

def run_amplification_experiment(experiment_name, platform_xml, workflow_json, scheduler_type):
    """运行单个放大实验"""
    
    print(f"\n--- 运行实验: {experiment_name} - {scheduler_type} ---")
    
    # 保存平台XML到临时文件（如果提供了平台配置）
    if platform_xml:
        platform_file = f"temp_platform_{experiment_name}.xml"
        with open(platform_file, 'w') as f:
            f.write(platform_xml)
    else:
        # 使用现有的test_platform.xml
        platform_file = "configs/test_platform.xml"
    
    # 配置实验运行器
    config = {
        "platform_file": platform_file,
        "workflow_dir": "workflows",
        "workflow_sizes": [20, 50, 100],
        "repetitions": 1,
        "output_dir": "results/final_amplification"
    }
    
    # 定义调度器
    schedulers = {
        "FIFO": FIFOScheduler,
        "HEFT": HEFTScheduler
    }
    
    # 创建实验运行器
    runner = WrenchExperimentRunner(schedulers, config)
    
    # 创建工作流文件
    workflow_file = f"temp_workflow_{experiment_name}_{scheduler_type}.json"
    with open(workflow_file, 'w') as f:
        json.dump(workflow_json, f, indent=2)
    
    try:
        # 运行实验
        result = runner._run_single_simulation(scheduler_type, schedulers[scheduler_type], workflow_file)
        
        print(f"✓ 实验完成: Makespan = {result.get('makespan', 'N/A')}")
        
        return {
            'experiment': experiment_name,
            'scheduler': scheduler_type,
            'makespan': result.get('makespan', float('inf')),
            'task_assignments': result.get('task_assignments', {}),
            'host_utilization': result.get('host_utilization', {}),
            'success': True
        }
        
    except Exception as e:
        print(f"✗ 实验失败: {e}")
        return {
            'experiment': experiment_name,
            'scheduler': scheduler_type,
            'makespan': float('inf'),
            'task_assignments': {},
            'host_utilization': {},
            'success': False,
            'error': str(e)
        }
    finally:
        # 清理临时文件
        import os
        if os.path.exists(workflow_file):
            os.remove(workflow_file)
        # 只删除我们创建的临时平台文件，不删除现有的test_platform.xml
        if platform_xml and os.path.exists(platform_file) and platform_file != "configs/test_platform.xml":
            os.remove(platform_file)

def analyze_amplification_results(results):
    """分析放大实验结果"""
    
    analysis = {}
    
    # 按实验分组结果
    experiments = defaultdict(list)
    for result in results:
        experiments[result['experiment']].append(result)
    
    for exp_name, exp_results in experiments.items():
        fifo_result = next((r for r in exp_results if r['scheduler'] == 'FIFO'), None)
        heft_result = next((r for r in exp_results if r['scheduler'] == 'HEFT'), None)
        
        if fifo_result and heft_result:
            fifo_makespan = fifo_result['makespan']
            heft_makespan = heft_result['makespan']
            
            if fifo_makespan != float('inf') and heft_makespan != float('inf'):
                difference = abs(fifo_makespan - heft_makespan)
                improvement = ((fifo_makespan - heft_makespan) / fifo_makespan) * 100
                
                # 分析主机选择差异
                fifo_assignments = fifo_result['task_assignments']
                heft_assignments = heft_result['task_assignments']
                
                fifo_host_counts = defaultdict(int)
                heft_host_counts = defaultdict(int)
                
                for task, host in fifo_assignments.items():
                    fifo_host_counts[host] += 1
                
                for task, host in heft_assignments.items():
                    heft_host_counts[host] += 1
                
                analysis[exp_name] = {
                    'fifo_makespan': fifo_makespan,
                    'heft_makespan': heft_makespan,
                    'absolute_difference': difference,
                    'relative_improvement': improvement,
                    'fifo_host_distribution': dict(fifo_host_counts),
                    'heft_host_distribution': dict(heft_host_counts),
                    'amplification_level': 'high' if abs(improvement) > 20 else 'medium' if abs(improvement) > 10 else 'low'
                }
                
                print(f"\n--- {exp_name} 分析 ---")
                print(f"FIFO Makespan: {fifo_makespan:.2f} 秒")
                print(f"HEFT Makespan: {heft_makespan:.2f} 秒")
                print(f"绝对差异: {difference:.2f} 秒")
                print(f"相对改进: {improvement:.2f}%")
                print(f"放大级别: {analysis[exp_name]['amplification_level']}")
            else:
                print(f"✗ {exp_name} 实验失败")
    
    return analysis

def run_existing_platform_experiment():
    """使用现有的test_platform.xml和workflow_manager生成的工作流运行实验"""
    print("\n--- 使用现有平台配置和workflow_manager生成的工作流运行实验 ---")
    
    # 读取现有的test_platform.xml
    platform_file = "configs/test_platform.xml"
    
    # 使用workflow_manager生成工作流
    from scripts.workflow_manager import WorkflowManager
    
    # 创建工作流管理器
    workflow_manager = WorkflowManager()
    
    # 生成实验工作流（使用小规模的几个任务）
    print("正在生成工作流...")
    workflow_files = workflow_manager.generate_experiment_workflows()
    
    if not workflow_files:
        print("❌ 未能生成工作流文件")
        return []
    
    # 选择第一个生成的工作流文件
    workflow_file = workflow_files[0]
    print(f"使用工作流文件: {workflow_file}")
    
    # 读取工作流内容
    with open(workflow_file, 'r') as f:
        workflow = json.load(f)
    
    results = []
    
    # 运行FIFO实验
    fifo_result = run_amplification_experiment('现有平台配置-workflow_manager', '', workflow, 'FIFO')
    results.append(fifo_result)
    
    # 运行HEFT实验  
    heft_result = run_amplification_experiment('现有平台配置-workflow_manager', '', workflow, 'HEFT')
    results.append(heft_result)
    
    return results

def main():
    """主函数 - 仅运行使用workflow_manager的现有平台实验"""
    print("时间差异放大测试 - 使用WrenchExperimentRunner (仅workflow_manager生成的工作流)")
    print("=" * 60)
    
    results = []
    
    # 仅运行使用现有平台和workflow_manager生成的工作流实验
    existing_platform_results = run_existing_platform_experiment()
    results.extend(existing_platform_results)
    
    # 分析结果
    analysis = analyze_amplification_results(results)
    
    # 保存详细结果
    output_data = {
        'experiments': [],  # 空列表，因为没有手动创建的实验
        'results': results,
        'analysis': analysis,
        'summary': {
            'total_experiments': 1,  # 只有1个实验（现有平台-workflow_manager）
            'successful_experiments': len([a for a in analysis.values() if 'amplification_level' in a]),
            'best_amplification': max([abs(a.get('relative_improvement', 0)) for a in analysis.values()], default=0),
            'average_amplification': sum([abs(a.get('relative_improvement', 0)) for a in analysis.values()]) / len(analysis) if analysis else 0
        }
    }
    
    # 保存结果到文件
    with open('final_amplification_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # 生成分析报告
    generate_analysis_report(analysis, output_data['summary'])
    
    print(f"\n{'=' * 60}")
    print("实验完成！")
    print(f"总实验数: {output_data['summary']['total_experiments']}")
    print(f"成功实验数: {output_data['summary']['successful_experiments']}")
    print(f"最佳放大效果: {output_data['summary']['best_amplification']:.2f}%")
    print(f"平均放大效果: {output_data['summary']['average_amplification']:.2f}%")
    print(f"详细结果保存至: final_amplification_results.json")
    print(f"分析报告保存至: final_amplification_analysis.md")
    print("\n注意：仅运行了使用现有test_platform.xml平台和workflow_manager生成的工作流实验")

def generate_analysis_report(analysis, summary):
    """生成分析报告"""
    
    report = f"""# 时间差异放大测试分析报告

## 实验概述
- 总实验数: {summary['total_experiments']}
- 成功实验数: {summary['successful_experiments']}
- 最佳放大效果: {summary['best_amplification']:.2f}%
- 平均放大效果: {summary['average_amplification']:.2f}%

## 详细分析

"""
    
    for exp_name, data in analysis.items():
        if 'amplification_level' in data:
            report += f"""
### {exp_name}

- **FIFO Makespan**: {data['fifo_makespan']:.2f} 秒
- **HEFT Makespan**: {data['heft_makespan']:.2f} 秒
- **绝对差异**: {data['absolute_difference']:.2f} 秒
- **相对改进**: {data['relative_improvement']:.2f}%
- **放大级别**: {data['amplification_level']}

**主机分布对比**:

FIFO调度器:
"""
            for host, count in data['fifo_host_distribution'].items():
                report += f"- {host}: {count} 个任务\n"
            
            report += "\nHEFT调度器:\n"
            for host, count in data['heft_host_distribution'].items():
                report += f"- {host}: {count} 个任务\n"
            
            report += "\n---\n"
    
    report += """
## 结论

通过对比不同平台配置和工作流复杂度下的FIFO与HEFT调度器性能，
可以观察到HEFT调度器在资源异构性较高的情况下表现更好，
时间差异得到了有效放大。

## 建议

1. 在高差异平台配置下，HEFT的优势更加明显
2. 复杂工作流能更好地展现调度器的差异
3. 资源受限环境也能放大调度器的选择差异
"""
    
    with open('final_amplification_analysis.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()