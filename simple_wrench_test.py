#!/usr/bin/env python3
"""
简化版本的wrench测试，直接使用wrench库
"""

import sys
from pathlib import Path
import json
import wrench

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

def create_simple_workflow():
    """创建简单测试工作流"""
    return {
        "metadata": {
            "name": "simple_test_workflow",
            "description": "简单测试工作流",
            "generated_at": "2024-01-01T00:00:00Z"
        },
        "workflow": {
            "tasks": [
                {
                    "id": "task_0",
                    "name": "task_0",
                    "runtime": 10.0,
                    "cores": 1,
                    "flops": 1000000000,  # 1 GFLOP
                    "memory": 100,
                    "dependencies": [],
                    "input_files": [],
                    "output_files": ["file_0"]
                },
                {
                    "id": "task_1", 
                    "name": "task_1",
                    "runtime": 20.0,
                    "cores": 1,
                    "flops": 2000000000,  # 2 GFLOP
                    "memory": 200,
                    "dependencies": ["task_0"],
                    "input_files": ["file_0"],
                    "output_files": ["file_1"]
                },
                {
                    "id": "task_2",
                    "name": "task_2", 
                    "runtime": 15.0,
                    "cores": 1,
                    "flops": 1500000000,  # 1.5 GFLOP
                    "memory": 150,
                    "dependencies": ["task_0"],
                    "input_files": ["file_0"],
                    "output_files": ["file_2"]
                }
            ],
            "files": [
                {"id": "file_0", "name": "file_0", "size": 100},
                {"id": "file_1", "name": "file_1", "size": 200},
                {"id": "file_2", "name": "file_2", "size": 150}
            ]
        }
    }

def test_wrench_basic():
    """测试基本的wrench功能"""
    print("测试基本的wrench功能...")
    
    # 读取平台文件
    platform_file = "test_platform.xml"
    with open(platform_file, "r") as f:
        platform_xml = f.read()
    
    print(f"平台XML长度: {len(platform_xml)} 字符")
    
    try:
        # 测试FIFO调度
        print("\n--- 测试FIFO调度 ---")
        fifo_result = test_scheduler_with_new_sim(platform_xml, "FIFO")
        
        # 测试HEFT调度
        print("\n--- 测试HEFT调度 ---")
        heft_result = test_scheduler_with_new_sim(platform_xml, "HEFT")
        
        # 对比结果
        compare_results(fifo_result, heft_result)
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scheduler_with_new_sim(platform_xml, scheduler_type):
    """使用新的模拟实例测试调度器"""
    
    # 创建新的模拟
    simulation = wrench.Simulation()
    controller_host = "ControllerHost"
    
    # 启动仿真
    simulation.start(platform_xml, controller_host)
    print(f"✓ {scheduler_type} 仿真启动成功")
    
    # 获取所有主机名
    all_hostnames = simulation.get_all_hostnames()
    compute_hosts = [host for host in all_hostnames if host not in [controller_host, "StorageHost"]]
    print(f"✓ 主机列表: {compute_hosts}")
    
    # 创建存储服务
    storage_service = simulation.create_simple_storage_service("StorageHost", ["/storage"])
    
    # 创建计算服务
    compute_services = {}
    for host in compute_hosts:
        compute_services[host] = simulation.create_bare_metal_compute_service(
            host, 
            {host: (-1, -1)},  # 只有该主机的核心
            "/scratch", 
            {}, 
            {}
        )
    
    # 创建工作流
    workflow_data = create_simple_workflow()
    
    # 转换为WfCommons格式
    wfcommons_data = {
        'name': workflow_data['metadata']['name'],
        'workflow_name': workflow_data['metadata']['name'],
        'description': workflow_data['metadata']['description'],
        'schemaVersion': '1.5',
        'author': {
            'name': 'WRENCH Experiment',
            'email': 'wrench@example.com'
        },
        'createdAt': workflow_data['metadata'].get('generated_at', '2024-01-01T00:00:00Z'),
        'workflow': {
            'specification': {
                'tasks': [],
                'files': []
            },
            'execution': {
                'tasks': []
            }
        }
    }
    
    # 转换任务
    task_children = {}
    for task in workflow_data['workflow']['tasks']:
        task_children[task['id']] = []
    for task in workflow_data['workflow']['tasks']:
        for parent_id in task.get('dependencies', []):
            if parent_id in task_children:
                task_children[parent_id].append(task['id'])
    
    for task in workflow_data['workflow']['tasks']:
        wfcommons_task = {
            'name': task['name'],
            'id': task['id'],
            'children': task_children[task['id']],
            'parents': task.get('dependencies', []),
            'inputFiles': task.get('input_files', []),
            'outputFiles': task.get('output_files', []),
            'flops': task.get('flops', 0),
            'memory': task.get('memory', 0)
        }
        wfcommons_data['workflow']['specification']['tasks'].append(wfcommons_task)
        
        execution_task = {
            'id': task['id'],
            'runtimeInSeconds': task.get('runtime', 1.0),
            'cores': 1,
            'avgCPU': 1.0
        }
        wfcommons_data['workflow']['execution']['tasks'].append(execution_task)
    
    # 转换文件
    for file in workflow_data['workflow']['files']:
        wfcommons_file = {
            'id': file['id'],
            'name': file['name'],
            'sizeInBytes': file.get('size', 0)
        }
        wfcommons_data['workflow']['specification']['files'].append(wfcommons_file)
    
    # 创建工作流
    workflow = simulation.create_workflow_from_json(
        wfcommons_data,
        reference_flop_rate='10Mf',
        ignore_machine_specs=True,
        redundant_dependencies=False,
        ignore_cycle_creating_dependencies=False,
        min_cores_per_task=1,
        max_cores_per_task=1,
        enforce_num_cores=True,
        ignore_avg_cpu=True,
        show_warnings=False
    )
    
    print(f"✓ 工作流创建成功，任务数: {len(workflow.get_tasks())}")
    
    # 创建文件副本
    for file in workflow.get_input_files():
        storage_service.create_file_copy(file)
    
    # 根据调度器类型选择调度器
    if scheduler_type == "FIFO":
        from src.wrench_schedulers import FIFOScheduler
        hosts_dict = {name: {} for name in compute_services.keys()}
        scheduler = FIFOScheduler(simulation, compute_services, hosts_dict)
        result_func = test_fifo_scheduler
    else:  # HEFT
        from src.wrench_schedulers import HEFTScheduler
        hosts_dict = {name: {} for name in compute_services.keys()}
        scheduler = HEFTScheduler(simulation, compute_services, hosts_dict)
        result_func = test_heft_scheduler
    
    # 运行调度
    result = result_func(simulation, workflow, compute_services, storage_service)
    
    # 终止仿真
    simulation.terminate()
    
    return result

def test_fifo_scheduler(simulation, workflow, compute_services, storage_service):
    """测试FIFO调度器"""
    from src.wrench_schedulers import FIFOScheduler
    
    hosts_dict = {name: {} for name in compute_services.keys()}
    scheduler = FIFOScheduler(simulation, compute_services, hosts_dict)
    
    print("FIFO调度器初始化完成")
    
    # 开始调度
    if hasattr(scheduler, 'schedule_ready_tasks'):
        scheduler.schedule_ready_tasks(workflow, storage_service)
    
    # 运行仿真循环
    while not workflow.is_done():
        event = simulation.wait_for_next_event()
        
        if event["event_type"] == "standard_job_completion":
            job = event["standard_job"]
            if hasattr(scheduler, 'handle_completion'):
                for task in job.get_tasks():
                    scheduler.handle_completion(task)
            # 调度新的就绪任务
            if hasattr(scheduler, 'schedule_ready_tasks'):
                scheduler.schedule_ready_tasks(workflow, storage_service)
        elif event["event_type"] == "simulation_termination":
            break
    
    # 获取makespan
    makespan = simulation.get_simulated_time()
    
    print(f"FIFO调度完成，Makespan: {makespan:.2f} 秒")
    
    return {
        'scheduler': 'FIFO',
        'makespan': makespan,
        'assignments': scheduler.task_assignments
    }

def test_heft_scheduler(simulation, workflow, compute_services, storage_service):
    """测试HEFT调度器"""
    from src.wrench_schedulers import HEFTScheduler
    
    hosts_dict = {name: {} for name in compute_services.keys()}
    scheduler = HEFTScheduler(simulation, compute_services, hosts_dict)
    
    print("HEFT调度器初始化完成")
    
    # 开始调度
    if hasattr(scheduler, 'schedule_ready_tasks'):
        scheduler.schedule_ready_tasks(workflow, storage_service)
    
    # 运行仿真循环
    while not workflow.is_done():
        event = simulation.wait_for_next_event()
        
        if event["event_type"] == "standard_job_completion":
            job = event["standard_job"]
            if hasattr(scheduler, 'handle_completion'):
                for task in job.get_tasks():
                    scheduler.handle_completion(task)
            # 调度新的就绪任务
            if hasattr(scheduler, 'schedule_ready_tasks'):
                scheduler.schedule_ready_tasks(workflow, storage_service)
        elif event["event_type"] == "simulation_termination":
            break
    
    # 获取makespan
    makespan = simulation.get_simulated_time()
    
    print(f"HEFT调度完成，Makespan: {makespan:.2f} 秒")
    
    return {
        'scheduler': 'HEFT',
        'makespan': makespan,
        'assignments': scheduler.task_assignments
    }

def compare_results(fifo_result, heft_result):
    """对比结果"""
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    
    fifo_makespan = fifo_result['makespan']
    heft_makespan = heft_result['makespan']
    
    print(f"FIFO Makespan: {fifo_makespan:.2f} 秒")
    print(f"HEFT Makespan: {heft_makespan:.2f} 秒")
    
    difference = abs(fifo_makespan - heft_makespan)
    improvement = ((fifo_makespan - heft_makespan) / fifo_makespan) * 100
    
    print(f"绝对差异: {difference:.2f} 秒")
    print(f"相对改进: {improvement:.2f}%")
    
    if difference > 1.0:
        print("✓ 差异明显放大")
    elif difference > 0.1:
        print("~ 差异有所放大")
    else:
        print("✗ 差异仍然很小")
    
    # 分析主机选择
    print(f"\n主机选择对比:")
    
    fifo_hosts = defaultdict(int)
    heft_hosts = defaultdict(int)
    
    for task, host in fifo_result['assignments'].items():
        fifo_hosts[host] += 1
    
    for task, host in heft_result['assignments'].items():
        heft_hosts[host] += 1
    
    all_hosts = set(fifo_hosts.keys()) | set(heft_hosts.keys())
    for host in sorted(all_hosts):
        fifo_count = fifo_hosts.get(host, 0)
        heft_count = heft_hosts.get(host, 0)
        print(f"  {host}: FIFO={fifo_count}, HEFT={heft_count}")

def main():
    """主函数"""
    print("简化wrench测试")
    print("=" * 60)
    
    success = test_wrench_basic()
    
    if success:
        print("\n✓ 测试完成")
    else:
        print("\n✗ 测试失败")

if __name__ == "__main__":
    from collections import defaultdict
    main()