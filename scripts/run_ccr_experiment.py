#!/usr/bin/env python3
"""
CCR对比实验脚本
测试HEFT vs FIFO在不同CCR值下的表现
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import subprocess

class CCRExperimentRunner:
    """CCR实验运行器"""
    
    def __init__(self, output_dir: str = "experiments/ccr_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CCR测试值
        self.ccr_values = [0.1, 1.0, 10.0]
        self.task_counts = [20, 50, 100]  # 小规模测试
        
    def generate_workflows(self):
        """生成不同CCR的工作流"""
        print("🔄 生成不同CCR的工作流...")
        
        for ccr in self.ccr_values:
            ccr_dir = self.output_dir / f"ccr_{ccr}"
            ccr_dir.mkdir(exist_ok=True)
            
            for task_count in self.task_counts:
                # 生成通信密集型工作流
                cmd = [
                    "python", "scripts/workflow_generator.py",
                    "--pattern", "comm_intensive",
                    "--tasks", str(task_count),
                    "--output", str(ccr_dir),
                    "--ccr", str(ccr)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ 生成工作流失败: {result.stderr}")
                    continue
                    
    def run_scheduler_comparison(self):
        """运行调度器对比实验"""
        print("🚀 开始调度器对比实验...")
        
        results = []
        
        for ccr in self.ccr_values:
            ccr_dir = self.output_dir / f"ccr_{ccr}"
            
            for task_count in self.task_counts:
                workflow_file = ccr_dir / f"comm_intensive_{task_count}_tasks.json"
                
                if not workflow_file.exists():
                    continue
                
                print(f"\n📊 测试CCR={ccr}, 任务数={task_count}")
                
                # 运行FIFO调度器
                fifo_result = self.run_scheduler("FIFO", workflow_file)
                
                # 运行HEFT调度器
                heft_result = self.run_scheduler("HEFT", workflow_file)
                
                if fifo_result and heft_result:
                    result = {
                        "ccr": ccr,
                        "task_count": task_count,
                        "fifo_makespan": fifo_result["makespan"],
                        "heft_makespan": heft_result["makespan"],
                        "improvement": (fifo_result["makespan"] - heft_result["makespan"]) / fifo_result["makespan"] * 100,
                        "timestamp": time.time()
                    }
                    results.append(result)
                    
                    print(f"   FIFO: {fifo_result['makespan']:.2f}s")
                    print(f"   HEFT: {heft_result['makespan']:.2f}s")
                    print(f"   改进: {result['improvement']:.1f}%")
        
        # 保存结果
        results_file = self.output_dir / "ccr_experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n✅ 实验完成！结果保存到: {results_file}")
        return results
    
    def run_scheduler(self, scheduler_name: str, workflow_file: Path) -> Dict:
        """运行单个调度器"""
        try:
            cmd = [
                "python", "-c",
                f"""
import sys
sys.path.insert(0, 'src')
from src.wrench_schedulers import WRENCHScheduler
import json

with open('{workflow_file}') as f:
    data = json.load(f)

scheduler = WRENCHScheduler()
result = scheduler.schedule_workflow(
    data['workflow']['tasks'],
    data['workflow']['files'],
    scheduler_type='{scheduler_name.lower()}'
)

print(json.dumps(result))
"""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                print(f"❌ {scheduler_name}运行失败: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ {scheduler_name}异常: {e}")
            return None
    
    def generate_summary_report(self, results: List[Dict]):
        """生成实验总结报告"""
        summary = {
            "experiment_info": {
                "name": "CCR对比实验",
                "description": "测试HEFT vs FIFO在不同CCR值下的性能表现",
                "ccr_values": self.ccr_values,
                "task_counts": self.task_counts,
                "total_tests": len(results)
            },
            "results": results,
            "analysis": {}
        }
        
        # 按CCR分组分析
        for ccr in self.ccr_values:
            ccr_results = [r for r in results if r["ccr"] == ccr]
            if ccr_results:
                avg_improvement = sum(r["improvement"] for r in ccr_results) / len(ccr_results)
                summary["analysis"][f"ccr_{ccr}"] = {
                    "avg_improvement": avg_improvement,
                    "min_improvement": min(r["improvement"] for r in ccr_results),
                    "max_improvement": max(r["improvement"] for r in ccr_results),
                    "test_count": len(ccr_results)
                }
        
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📊 生成总结报告: {summary_file}")
        return summary

def main():
    parser = argparse.ArgumentParser(description='CCR对比实验')
    parser.add_argument('--output', default='experiments/ccr_results', 
                       help='输出目录')
    parser.add_argument('--skip-generation', action='store_true',
                       help='跳过工作流生成')
    
    args = parser.parse_args()
    
    runner = CCRExperimentRunner(args.output)
    
    if not args.skip_generation:
        runner.generate_workflows()
    
    results = runner.run_scheduler_comparison()
    runner.generate_summary_report(results)
    
    print("\n" + "="*50)
    print("📋 CCR实验完成！")
    print("="*50)
    
    # 打印关键结果
    for ccr in runner.ccr_values:
        ccr_results = [r for r in results if r["ccr"] == ccr]
        if ccr_results:
            avg_improvement = sum(r["improvement"] for r in ccr_results) / len(ccr_results)
            print(f"CCR={ccr}: HEFT平均改进 {avg_improvement:.1f}%")

if __name__ == "__main__":
    main()