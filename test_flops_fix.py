#!/usr/bin/env python3
"""
测试FLOPS值修复的脚本
"""

import json
import tempfile
import os
from pathlib import Path
from src.utils import WrenchExperimentRunner
from src.wrench_schedulers import FIFOScheduler, HEFTScheduler

def create_test_workflow_with_different_flops():
    """创建具有不同FLOPS值的工作流进行测试"""
    
    # 测试用例1: 低FLOPS值
    workflow_low_flops = {
        "metadata": {
            "name": "test_low_flops",
            "description": "测试低FLOPS值的工作流",
            "generated_at": "2024-01-01T00:00:00Z"
        },
        "workflow": {
            "tasks": [
                {
                    "id": "task_0",
                    "name": "low_flops_task",
                    "runtime": 10.0,
                    "cores": 1,
                    "flops": 1000000,  # 1MFLOP
                    "memory": 1024,
                    "dependencies": [],
                    "input_files": [],
                    "output_files": []
                }
            ],
            "files": []
        }
    }
    
    # 测试用例2: 高FLOPS值  
    workflow_high_flops = {
        "metadata": {
            "name": "test_high_flops", 
            "description": "测试高FLOPS值的工作流",
            "generated_at": "2024-01-01T00:00:00Z"
        },
        "workflow": {
            "tasks": [
                {
                    "id": "task_0",
                    "name": "high_flops_task",
                    "runtime": 10.0,
                    "cores": 1,
                    "flops": 10000000000,  # 10GFLOP
                    "memory": 1024,
                    "dependencies": [],
                    "input_files": [],
                    "output_files": []
                }
            ],
            "files": []
        }
    }
    
    return workflow_low_flops, workflow_high_flops

def test_flops_impact():
    """测试FLOPS值对执行时间的影响"""
    
    print("=== 测试FLOPS值修复效果 ===")
    
    # 创建临时工作流文件
    workflow_low, workflow_high = create_test_workflow_with_different_flops()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 保存工作流文件
        low_flops_file = temp_path / "low_flops.json"
        high_flops_file = temp_path / "high_flops.json"
        
        with open(low_flops_file, 'w') as f:
            json.dump(workflow_low, f, indent=2)
        
        with open(high_flops_file, 'w') as f:
            json.dump(workflow_high, f, indent=2)
        
        print(f"低FLOPS工作流文件: {low_flops_file}")
        print(f"高FLOPS工作流文件: {high_flops_file}")
        
        # 使用现有的平台配置
        platform_xml = """<?xml version="4.1" encoding="UTF-8"?>
<!DOCTYPE platform SYSTEM "http://simgrid.gforge.inria.fr/simgrid/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="ControllerHost" speed="1Gf" core="1" />
    <host id="StorageHost" speed="1Gf" core="1" />
    <host id="ComputeHost1" speed="1Gf" core="1" />
    <host id="ComputeHost2" speed="5Gf" core="1" />
    <host id="ComputeHost3" speed="2Gf" core="1" />
    <host id="ComputeHost4" speed="10Gf" core="1" />
    <link id="link1" bandwidth="1GBps" latency="1us" />
    <link id="link2" bandwidth="1GBps" latency="1us" />
    <link id="link3" bandwidth="1GBps" latency="1us" />
    <link id="link4" bandwidth="1GBps" latency="1us" />
    <link id="link5" bandwidth="1GBps" latency="1us" />
    <route src="ControllerHost" dst="StorageHost">
      <link_ctn id="link1"/>
    </route>
    <route src="ControllerHost" dst="ComputeHost1">
      <link_ctn id="link2"/>
    </route>
    <route src="ControllerHost" dst="ComputeHost2">
      <link_ctn id="link3"/>
    </route>
    <route src="ControllerHost" dst="ComputeHost3">
      <link_ctn id="link4"/>
    </route>
    <route src="ControllerHost" dst="ComputeHost4">
      <link_ctn id="link5"/>
    </route>
  </zone>
</platform>"""
        
        platform_file = temp_path / "platform.xml"
        with open(platform_file, 'w') as f:
            f.write(platform_xml)
        
        # 创建实验配置
        config = {
            "platform_file": str(platform_file),
            "workflow_dir": str(temp_path),
            "output_dir": str(temp_path / "results"),
            "workflow_sizes": ["low_flops", "high_flops"],
            "repetitions": 1
        }
        
        # 创建调度器映射
        schedulers = {
            "FIFO": lambda sim, services, hosts: FIFOScheduler(sim, services, hosts)
        }
        
        # 创建实验运行器
        runner = WrenchExperimentRunner(schedulers, config)
        
        print("\n运行低FLOPS工作流测试...")
        result_low = runner._run_single_simulation("FIFO", schedulers["FIFO"], low_flops_file)
        
        print("\n运行高FLOPS工作流测试...")
        result_high = runner._run_single_simulation("FIFO", schedulers["FIFO"], high_flops_file)
        
        # 分析结果
        print("\n=== 测试结果分析 ===")
        print(f"低FLOPS工作流 (1MFLOP): {result_low}")
        print(f"高FLOPS工作流 (10GFLOP): {result_high}")
        
        if result_low['status'] == 'success' and result_high['status'] == 'success':
            low_makespan = result_low['makespan']
            high_makespan = result_high['makespan']
            
            print(f"\n低FLOPS执行时间: {low_makespan:.6f} 秒")
            print(f"高FLOPS执行时间: {high_makespan:.6f} 秒")
            print(f"时间比率: {high_makespan / low_makespan:.2f}")
            
            # 验证FLOPS值是否生效
            expected_ratio = 10000  # 10GFLOP / 1MFLOP = 10000
            actual_ratio = high_makespan / low_makespan
            
            if actual_ratio > 100:  # 期望至少100倍差异
                print("\n✅ FLOPS值修复成功！执行时间随FLOPS值增加而增加")
            else:
                print(f"\n❌ FLOPS值可能未生效，期望比率约{expected_ratio}，实际比率{actual_ratio:.2f}")
        else:
            print("\n❌ 测试失败，请检查错误信息")
            print(f"低FLOPS结果: {result_low}")
            print(f"高FLOPS结果: {result_high}")

if __name__ == "__main__":
    test_flops_impact()