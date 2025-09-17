#!/usr/bin/env python3
"""
最终放大效果测试 - 包含WASS调度器
对比FIFO、HEFT和WASS三种调度器的性能差异
"""
import json
import tempfile
import os
from src.wrench_schedulers import FIFOScheduler, HEFTScheduler, WASSHeuristicScheduler
from src.utils import WrenchExperimentRunner

def create_amplification_platform():
    """创建高差异平台配置"""
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

def create_amplification_workflow():
    """创建放大效果测试工作流 - 高FLOPS值"""
    return {
        "metadata": {
            "name": "amplification_test_workflow",
            "description": "FLOPS放大效果测试工作流",
            "generated_at": "2024-01-01T00:00:00Z"
        },
        "workflow": {
            "tasks": [
                {
                    "id": "task_0",
                    "name": "data_generation",
                    "runtime": 10.0,
                    "cores": 1,
                    "flops": 100000000000000,  # 100 GFLOP
                    "memory": 100,
                    "dependencies": [],
                    "input_files": [],
                    "output_files": ["large_dataset"]
                },
                {
                    "id": "task_1",
                    "name": "heavy_computation_1",
                    "runtime": 50.0,
                    "cores": 2,
                    "flops": 500000000000000,  # 500 GFLOP
                    "memory": 300,
                    "dependencies": ["task_0"],
                    "input_files": ["large_dataset"],
                    "output_files": ["result_1"]
                },
                {
                    "id": "task_2",
                    "name": "heavy_computation_2",
                    "runtime": 30.0,
                    "cores": 1,
                    "flops": 300000000000000,  # 300 GFLOP
                    "memory": 200,
                    "dependencies": ["task_0"],
                    "input_files": ["large_dataset"],
                    "output_files": ["result_2"]
                },
                {
                    "id": "task_3",
                    "name": "final_processing",
                    "runtime": 40.0,
                    "cores": 1,
                    "flops": 400000000000000,  # 400 GFLOP
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
                    "size": 2000000000  # 2GB
                },
                {
                    "id": "result_1",
                    "name": "result_1",
                    "size": 1000000000  # 1GB
                },
                {
                    "id": "result_2",
                    "name": "result_2",
                    "size": 800000000  # 800MB
                },
                {
                    "id": "final_output",
                    "name": "final_output",
                    "size": 500000000  # 500MB
                }
            ]
        }
    }

def test_three_schedulers():
    """测试三种调度器的放大效果"""
    print("=== 三调度器FLOPS放大效果测试 ===\n")
    
    # 创建工作流和平台文件
    workflow_data = create_amplification_workflow()
    platform_xml = create_amplification_platform()
    
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
        detailed_results = []
        
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
            
            # 运行多次以获得稳定结果
            makespans = []
            for i in range(3):
                print(f"  运行第 {i+1} 次...")
                result = runner._run_single_simulation(
                    f"{scheduler_name}_run_{i+1}",
                    scheduler_class,
                    workflow_file
                )
                
                if result['status'] == 'success':
                    makespans.append(result['makespan'])
                    print(f"    第 {i+1} 次 Makespan: {result['makespan']:.2f}s")
                else:
                    print(f"    第 {i+1} 次失败: {result.get('error', '未知错误')}")
            
            if makespans:
                avg_makespan = sum(makespans) / len(makespans)
                results[scheduler_name] = avg_makespan
                detailed_results.append({
                    'scheduler': scheduler_name,
                    'makespans': makespans,
                    'average': avg_makespan,
                    'min': min(makespans),
                    'max': max(makespans)
                })
                print(f"  {scheduler_name} 平均 Makespan: {avg_makespan:.2f}s")
            else:
                results[scheduler_name] = float('inf')
                print(f"  {scheduler_name} 所有运行都失败")
        
        # 保存详细结果
        results_data = {
            'summary': {},
            'detailed_results': detailed_results,
            'experiments': []
        }
        
        # 比较结果并计算改进
        print(f"\n{'='*60}")
        print(f"📊 三调度器性能对比分析")
        print(f"{'='*60}")
        
        if 'FIFO' in results and results['FIFO'] != float('inf'):
            print(f"FIFO 平均 Makespan: {results['FIFO']:.2f}s")
            results_data['summary']['FIFO'] = results['FIFO']
            
            for scheduler in ['HEFT', 'WASS']:
                if scheduler in results and results[scheduler] != float('inf'):
                    improvement = ((results['FIFO'] - results[scheduler]) / results['FIFO']) * 100
                    print(f"{scheduler} 平均 Makespan: {results[scheduler]:.2f}s")
                    print(f"{scheduler} 相比FIFO改进: {improvement:.1f}%")
                    results_data['summary'][scheduler] = results[scheduler]
                    results_data['summary'][f'{scheduler}_vs_FIFO_improvement'] = improvement
        
        # 保存结果到文件
        with open('three_scheduler_amplification_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 结果已保存到: three_scheduler_amplification_results.json")
        
        # 验证差异
        if len(results) >= 2:
            min_result = min(results.values())
            max_result = max(results.values())
            if max_result > min_result * 1.1:  # 至少10%差异
                print(f"\n🎯 成功验证：三调度器之间存在显著性能差异！")
                print(f"   最佳: {min(results, key=results.get)} ({min_result:.2f}s)")
                print(f"   最差: {max(results, key=results.get)} ({max_result:.2f}s)")
                print(f"   差异倍数: {max_result/min_result:.1f}x")
            else:
                print(f"\n⚠️ 警告：调度器之间差异较小，可能需要调整测试参数")
                
    finally:
        # 清理临时文件
        if os.path.exists(workflow_file):
            os.unlink(workflow_file)
        if os.path.exists(platform_file):
            os.unlink(platform_file)

if __name__ == "__main__":
    test_three_schedulers()