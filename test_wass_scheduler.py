#!/usr/bin/env python3
"""
测试WASSHeuristicScheduler与FIFO和HEFT的区别
"""
import json
import tempfile
import os
from src.wrench_schedulers import FIFOScheduler, HEFTScheduler, WASSHeuristicScheduler
from src.utils import WrenchExperimentRunner

def create_test_workflow():
    """创建一个简单的测试工作流"""
    return {
        "metadata": {
            "name": "wass_test_workflow",
            "description": "测试WASS调度器的工作流",
            "generated_at": "2024-01-01T00:00:00Z"
        },
        "workflow": {
            "tasks": [
                {
                    "id": "task_0",
                    "name": "data_generation",
                    "runtime": 10.0,
                    "cores": 1,
                    "flops": 1000000000000,  # 1 GFLOP
                    "memory": 100,
                    "dependencies": [],
                    "input_files": [],
                    "output_files": ["large_dataset"]
                },
                {
                    "id": "task_1", 
                    "name": "data_processing_1",
                    "runtime": 20.0,
                    "cores": 2,
                    "flops": 5000000000000,  # 5 GFLOP
                    "memory": 200,
                    "dependencies": ["task_0"],
                    "input_files": ["large_dataset"],
                    "output_files": ["result_1"]
                },
                {
                    "id": "task_2",
                    "name": "data_processing_2", 
                    "runtime": 15.0,
                    "cores": 1,
                    "flops": 3000000000000,  # 3 GFLOP
                    "memory": 150,
                    "dependencies": ["task_0"],
                    "input_files": ["large_dataset"],
                    "output_files": ["result_2"]
                },
                {
                    "id": "task_3",
                    "name": "final_analysis",
                    "runtime": 25.0,
                    "cores": 1,
                    "flops": 8000000000000,  # 8 GFLOP
                    "memory": 250,
                    "dependencies": ["task_1", "task_2"],
                    "input_files": ["result_1", "result_2"],
                    "output_files": ["final_output"]
                }
            ],
            "files": [
                {
                    "id": "large_dataset",
                    "name": "large_dataset",
                    "size": 1000000000  # 1GB
                },
                {
                    "id": "result_1",
                    "name": "result_1", 
                    "size": 500000000  # 500MB
                },
                {
                    "id": "result_2",
                    "name": "result_2",
                    "size": 300000000  # 300MB
                },
                {
                    "id": "final_output",
                    "name": "final_output",
                    "size": 100000000  # 100MB
                }
            ]
        }
    }

def create_test_platform():
    """创建测试平台配置"""
    return """<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="ControllerHost" speed="1Gf" core="1"/>
    <host id="StorageHost" speed="1Gf" core="1">
      <disk id="storage_disk" read_bw="150MBps" write_bw="150MBps">
        <prop id="size" value="1000GB"/>
        <prop id="mount" value="/storage"/>
      </disk>
    </host>
    <host id="ComputeHost1" speed="1Gf" core="4">
      <disk id="local_disk1" read_bw="200MBps" write_bw="200MBps">
        <prop id="size" value="100GB"/>
        <prop id="mount" value="/scratch"/>
      </disk>
    </host>
    <host id="ComputeHost2" speed="5Gf" core="8">
      <disk id="local_disk2" read_bw="250MBps" write_bw="250MBps">
        <prop id="size" value="200GB"/>
        <prop id="mount" value="/scratch"/>
      </disk>
    </host>
    <host id="ComputeHost3" speed="2Gf" core="6">
      <disk id="local_disk3" read_bw="220MBps" write_bw="220MBps">
        <prop id="size" value="150GB"/>
        <prop id="mount" value="/scratch"/>
      </disk>
    </host>
    <host id="ComputeHost4" speed="10Gf" core="16">
      <disk id="local_disk4" read_bw="300MBps" write_bw="300MBps">
        <prop id="size" value="500GB"/>
        <prop id="mount" value="/scratch"/>
      </disk>
    </host>
    <link id="network_link" bandwidth="1GBps" latency="1ms"/>
    <route src="ControllerHost" dst="StorageHost">
      <link_ctn id="network_link"/>
    </route>
    <route src="ControllerHost" dst="ComputeHost1">
      <link_ctn id="network_link"/>
    </route>
    <route src="ControllerHost" dst="ComputeHost2">
      <link_ctn id="network_link"/>
    </route>
    <route src="ControllerHost" dst="ComputeHost3">
      <link_ctn id="network_link"/>
    </route>
    <route src="ControllerHost" dst="ComputeHost4">
      <link_ctn id="network_link"/>
    </route>
    <route src="StorageHost" dst="ComputeHost1">
      <link_ctn id="network_link"/>
    </route>
    <route src="StorageHost" dst="ComputeHost2">
      <link_ctn id="network_link"/>
    </route>
    <route src="StorageHost" dst="ComputeHost3">
      <link_ctn id="network_link"/>
    </route>
    <route src="StorageHost" dst="ComputeHost4">
      <link_ctn id="network_link"/>
    </route>
  </zone>
</platform>"""

def test_schedulers():
    """测试三种调度器"""
    print("=== WASS调度器测试开始 ===\n")
    
    # 创建工作流和平台文件
    workflow_data = create_test_workflow()
    platform_xml = create_test_platform()
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(workflow_data, f, indent=2)
        workflow_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(platform_xml)
        platform_file = f.name
    
    try:
        # 定义调度器配置
        schedulers = {
            'FIFO': FIFOScheduler,
            'HEFT': HEFTScheduler, 
            'WASS': WASSHeuristicScheduler
        }
        
        results = {}
        
        for scheduler_name, scheduler_class in schedulers.items():
            print(f"\n--- 测试 {scheduler_name} 调度器 ---")
            
            # 配置实验
            config = {
                'platform_file': platform_file,
                'workflow_dir': os.path.dirname(workflow_file),
                'output_dir': tempfile.mkdtemp(),
                'reference_flop_rate': '1Mf',
                'ignore_machine_specs': False
            }
            
            # 运行实验
            runner = WrenchExperimentRunner(
                schedulers={'test': scheduler_class},
                config=config
            )
            
            # 运行单个仿真
            result = runner._run_single_simulation(
                scheduler_name,
                scheduler_class,
                workflow_file
            )
            
            makespan = result['makespan']
            results[scheduler_name] = makespan
            print(f"{scheduler_name} Makespan: {makespan:.2f} seconds")
            if result['status'] == 'failed':
                print(f"  ⚠️ 仿真失败: {result.get('error', '未知错误')}")
        
        # 比较结果
        print(f"\n=== 测试结果比较 ===")
        print(f"FIFO Makespan: {results['FIFO']:.2f}s")
        print(f"HEFT Makespan: {results['HEFT']:.2f}s")
        print(f"WASS Makespan: {results['WASS']:.2f}s")
        
        # 计算改进
        if results['FIFO'] != float('inf') and results['WASS'] != float('inf'):
            wass_improvement = ((results['FIFO'] - results['WASS']) / results['FIFO']) * 100
            print(f"WASS相比FIFO改进: {wass_improvement:.1f}%")
        
        if results['HEFT'] != float('inf') and results['WASS'] != float('inf'):
            wass_vs_heft = ((results['HEFT'] - results['WASS']) / results['HEFT']) * 100
            print(f"WASS相比HEFT改进: {wass_vs_heft:.1f}%")
        
        # 验证WASS与FIFO不同
        if abs(results['WASS'] - results['FIFO']) < 0.01:
            print("\n⚠️ 警告: WASS和FIFO结果几乎相同，可能实现有问题！")
        else:
            print(f"\n✅ WASS与FIFO有明显差异，调度器工作正常！")
            
    finally:
        # 清理临时文件
        if os.path.exists(workflow_file):
            os.unlink(workflow_file)
        if os.path.exists(platform_file):
            os.unlink(platform_file)

if __name__ == "__main__":
    test_schedulers()