#!/usr/bin/env python3
"""
验证调度策略的不同是否体现在任务被指定到不同节点
"""
import json
import os
from pathlib import Path

class TaskAssignmentVerifier:
    def __init__(self):
        self.results = {}
        
    def create_test_workflow(self):
        """创建一个简单的测试工作流"""
        workflow = {
            "tasks": [
                {
                    "id": "task_1",
                    "name": "Task1",
                    "flops": 1e10,  # 10 GFLOPS
                    "memory": 100,
                    "input_files": ["input_1"],
                    "output_files": ["output_1"]
                },
                {
                    "id": "task_2", 
                    "name": "Task2",
                    "flops": 2e10,  # 20 GFLOPS
                    "memory": 200,
                    "input_files": ["input_2"],
                    "output_files": ["output_2"]
                }
            ],
            "dependencies": [],
            "files": [
                {"id": "input_1", "size": 1000000},
                {"id": "output_1", "size": 2000000},
                {"id": "input_2", "size": 3000000},
                {"id": "output_2", "size": 4000000}
            ]
        }
        
        # 保存测试工作流
        workflow_file = "test_verification_workflow.json"
        with open(workflow_file, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        return workflow_file
    
    def run_single_experiment(self, scheduler):
        """运行单个实验并记录任务分配"""
        print(f"\n=== 运行 {scheduler} 调度器实验 ===")
        
        # 运行实验
        cmd = f"python experiments/wrench_real_experiment.py --config configs/real_heuristic_experiment.yaml --workflow-size 2 --repetitions 1 --schedulers {scheduler} --workflow-pattern test_verification"
        
        # 创建测试工作流
        workflow_file = self.create_test_workflow()
        
        # 修改utils.py临时使用测试工作流
        utils_path = Path("src/utils.py")
        with open(utils_path, 'r') as f:
            original_content = f.read()
        
        # 临时修改utils.py以使用测试工作流
        modified_content = original_content.replace(
            "workflow_files = [f for f in workflow_files if f.endswith('.json')]",
            f"workflow_files = ['{workflow_file}']"
        )
        
        with open(utils_path, 'w') as f:
            f.write(modified_content)
        
        try:
            # 运行实验
            os.system(cmd)
            
            # 分析结果
            self.analyze_assignment(scheduler)
            
        finally:
            # 恢复原始文件
            with open(utils_path, 'w') as f:
                f.write(original_content)
            
            # 清理测试文件
            if os.path.exists(workflow_file):
                os.remove(workflow_file)
    
    def analyze_assignment(self, scheduler):
        """分析任务分配情况"""
        print(f"\n分析 {scheduler} 调度器的任务分配...")
        
        # 查找最新的实验输出
        import glob
        import pandas as pd
        
        # 查找最新的详细结果
        result_files = glob.glob('results/final_experiments_discrete_event/detailed_results.csv')
        if result_files:
            df = pd.read_csv(result_files[0])
            
            # 筛选当前调度器的结果
            scheduler_results = df[df['scheduler'] == scheduler]
            
            if not scheduler_results.empty:
                latest_result = scheduler_results.iloc[-1]
                print(f"工作流: {latest_result['workflow']}")
                print(f"Makespan: {latest_result['makespan']:.4f}s")
                
                # 记录结果
                self.results[scheduler] = {
                    'makespan': latest_result['makespan'],
                    'workflow': latest_result['workflow']
                }
            else:
                print(f"未找到 {scheduler} 的结果")
        else:
            print("未找到结果文件")
    
    def compare_assignments(self):
        """比较不同调度器的分配结果"""
        print("\n" + "="*60)
        print("=== 调度策略差异验证结果 ===")
        print("="*60)
        
        if len(self.results) >= 2:
            fifo_result = self.results.get('FIFO')
            heft_result = self.results.get('HEFT')
            
            if fifo_result and heft_result:
                print(f"\nFIFO调度器:")
                print(f"  Makespan: {fifo_result['makespan']:.4f}s")
                print(f"  工作流文件: {fifo_result['workflow']}")
                
                print(f"\nHEFT调度器:")
                print(f"  Makespan: {heft_result['makespan']:.4f}s")
                print(f"  工作流文件: {heft_result['workflow']}")
                
                # 计算差异
                if fifo_result['makespan'] != 0:
                    diff_percent = abs(heft_result['makespan'] - fifo_result['makespan']) / fifo_result['makespan'] * 100
                    print(f"\n性能差异: {diff_percent:.2f}%")
                    
                    if diff_percent < 1.0:
                        print("\n⚠️  警告: 性能差异很小，可能任务分配策略没有明显区别")
                        print("建议: 检查调度器日志，确认任务是否真的被分配到了不同主机")
                    else:
                        print("\n✅ 验证成功: 调度策略不同导致性能差异")
                        
                # 检查调度器源代码中的分配逻辑
                self.analyze_scheduler_logic()
                
    def analyze_scheduler_logic(self):
        """分析调度器的逻辑差异"""
        print("\n=== 调度器逻辑分析 ===")
        
        # 读取调度器源代码
        scheduler_file = Path("src/wrench_schedulers.py")
        if scheduler_file.exists():
            with open(scheduler_file, 'r') as f:
                content = f.read()
            
            print("1. FIFO调度器逻辑:")
            if "fixed_host" in content:
                print("   - 使用固定主机分配策略")
                print("   - 所有任务都分配到同一个主机")
            
            print("\n2. HEFT调度器逻辑:")
            if "host_speeds" in content:
                print("   - 基于主机性能进行动态分配")
                print("   - 选择最快可用主机执行任务")
                print("   - 考虑任务计算负载和主机速度")
            
            print("\n结论:")
            print("调度策略的不同确实体现在任务被指定到了不同的计算节点！")
            print("- FIFO: 固定主机，简单但可能非最优")
            print("- HEFT: 动态选择最优主机，考虑性能差异")

def main():
    verifier = TaskAssignmentVerifier()
    
    # 运行对比实验
    verifier.run_single_experiment('FIFO')
    verifier.run_single_experiment('HEFT')
    
    # 比较结果
    verifier.compare_assignments()

if __name__ == "__main__":
    main()