"""工具函数、日志设置和最终版的实验运行器。"""
from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Generator, List
from pathlib import Path
import sys
import wrench
import pandas as pd
import numpy as np
import json

def get_logger(name, level=logging.INFO):
    """获取一个已配置的日志器。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# --- 最终版实验运行器 ---
class WrenchExperimentRunner:
    """处理多个WRENCH模拟的执行，以比较不同的调度器。"""
    def __init__(self, schedulers: Dict[str, Any], config: Dict[str, Any]):
        self.schedulers_map = schedulers
        self.config = config
        self.platform_file = config.get("platform_file")
        self.workflow_dir = Path(config.get("workflow_dir", "workflows"))
        self.workflow_sizes = config.get("workflow_sizes", [20, 50, 100])
        self.repetitions = config.get("repetitions", 3)
        self.output_dir = Path(config.get("output_dir", "results/final_experiments"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_single_simulation(self, scheduler_name: str, scheduler_impl: Any, workflow_file: str) -> Dict[str, Any]:
        """运行单个WRENCH模拟并返回结果。"""
        try:
            # 读取平台文件内容
            with open(self.platform_file, "r") as f:
                platform_xml = f.read()
            
            # 创建模拟对象
            simulation = wrench.Simulation()
            
            # 启动仿真，指定控制器主机
            controller_host = "ControllerHost"
            simulation.start(platform_xml, controller_host)
            
            # 获取所有主机名
            all_hostnames = simulation.get_all_hostnames()
            
            # 过滤出计算主机（排除控制器主机和存储主机）
            compute_hosts = [host for host in all_hostnames if host not in [controller_host, "StorageHost"]]
            
            # 创建存储服务（在StorageHost上，挂载点为/storage）
            storage_service = simulation.create_simple_storage_service("StorageHost", ["/storage"])
            
            # 为每个计算主机创建独立的计算服务
            compute_services = {}
            for host in compute_hosts:
                compute_services[host] = simulation.create_bare_metal_compute_service(
                    host, 
                    {host: (-1, -1)},  # 只有该主机的核心
                    "/scratch", 
                    {}, 
                    {}
                )
            
            # 创建主机信息字典
            hosts_dict = {name: {} for name in compute_hosts}
            
            # 实例化调度器
            if isinstance(scheduler_impl, str):
                # 字符串形式的调度器类名，需要导入对应的类
                import importlib
                module_name, class_name = scheduler_impl.rsplit('.', 1) if '.' in scheduler_impl else ('wrench_schedulers', scheduler_impl)
                try:
                    module = importlib.import_module(module_name)
                    scheduler_class = getattr(module, class_name)
                    scheduler_instance = scheduler_class(simulation, compute_services, hosts_dict)
                except (ImportError, AttributeError) as e:
                    print(f"导入调度器类失败: {scheduler_impl}, 错误: {e}")
                    # 回退到基础调度器
                    from wrench_schedulers import BaseScheduler
                    scheduler_instance = BaseScheduler(simulation, compute_services, hosts_dict)
            elif callable(scheduler_impl) and not isinstance(scheduler_impl, type):
                # 工厂函数形式的调度器
                scheduler_instance = scheduler_impl(simulation, compute_services, hosts_dict)
            else:
                # 类形式的调度器
                scheduler_instance = scheduler_impl(simulation, compute_services, hosts_dict)
            
            # 从JSON文件创建工作流
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)
            
            # 转换工作流数据为WfCommons格式
            print(f"转换工作流数据为WfCommons格式...")
            
            # 构建任务依赖关系映射
            task_children = {}
            for task in workflow_data['workflow']['tasks']:
                task_children[task['id']] = []
            for task in workflow_data['workflow']['tasks']:
                for parent_id in task.get('dependencies', []):
                    if parent_id in task_children:
                        task_children[parent_id].append(task['id'])
            
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
                
                # 添加执行信息
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
            
            # 创建工作流 - 使用直接方法
            print(f"创建工作流，名称: {wfcommons_data['name']}")
            
            # 创建工作流 - 确保包含workflow_name字段
            if 'workflow_name' not in wfcommons_data:
                wfcommons_data['workflow_name'] = wfcommons_data.get('name', 'unknown')
            
            # 使用直接的create_workflow_from_json方法创建工作流
            # 修复：设置合适的reference_flop_rate值让FLOPS值生效
            # 使用较大的reference_flop_rate值避免小FLOPS值被缩放为0
            workflow = simulation.create_workflow_from_json(
                wfcommons_data,
                reference_flop_rate=str(task.get('flops', 0)/1000000)+'Mf',  # 使用较小的参考值，让实际FLOPS值更有意义
                ignore_machine_specs=False,  # 改为False，让机器规格影响执行时间
                redundant_dependencies=False,
                ignore_cycle_creating_dependencies=False,
                min_cores_per_task=1,
                max_cores_per_task=1,
                enforce_num_cores=True,
                ignore_avg_cpu=True,
                show_warnings=False
            )
            
            print(f"工作流创建成功！工作流名称: {workflow.get_name()}")
            print(f"工作流任务数: {len(workflow.get_tasks())}")
            
            # 创建工作流中的所有文件副本
            try:
                for file in workflow.get_input_files():
                    storage_service.create_file_copy(file)
            except Exception as e:
                print(f"文件副本创建失败: {e}")
                # 跳过文件副本创建，继续执行
            
            # 开始调度
            if hasattr(scheduler_instance, 'schedule_ready_tasks'):
                scheduler_instance.schedule_ready_tasks(workflow, storage_service)
            
            # 运行仿真循环
            while not workflow.is_done():
                # 等待下一个事件
                event = simulation.wait_for_next_event()
                
                # 处理任务完成事件
                if event["event_type"] == "standard_job_completion":
                    job = event["standard_job"]
                    if hasattr(scheduler_instance, 'handle_completion'):
                        for task in job.get_tasks():
                            scheduler_instance.handle_completion(task)
                    # 调度新的就绪任务
                    if hasattr(scheduler_instance, 'schedule_ready_tasks'):
                        scheduler_instance.schedule_ready_tasks(workflow, storage_service)
                elif event["event_type"] == "simulation_termination":
                    break
            
            # 获取makespan
            makespan = simulation.get_simulated_time()
            
            # 终止仿真
            simulation.terminate()
            
            # 修复: 使用Path对象来获取文件名
            workflow_filename = Path(workflow_file).name
            return {"scheduler": scheduler_name, "workflow": workflow_filename, "makespan": makespan, "status": "success"}
        except Exception as e:
            # 修复: 使用Path对象来获取文件名
            workflow_filename = Path(workflow_file).name
            import traceback
            print(f"ERROR running {scheduler_name} on {workflow_filename}: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return {"scheduler": scheduler_name, "workflow": workflow_filename, "makespan": float('inf'), "status": "failed"}

    def run_all(self) -> List[Dict[str, Any]]:
        """运行所有配置的实验。"""
        results = []
        total_exps = len(self.schedulers_map) * len(self.workflow_sizes) * self.repetitions
        print(f"总实验数: {total_exps}")
        
        exp_count = 0
        for name, sched_impl in self.schedulers_map.items():
            for size in self.workflow_sizes:
                # [FIX] 智能扫描工作流文件，而不是依赖固定名称
                matching_files = list(self.workflow_dir.glob(f'*_{size}_*.json')) + \
                                 list(self.workflow_dir.glob(f'*_{size}.json')) + \
                                 list(self.workflow_dir.glob(f'*{size}-tasks-wf.json'))
                
                if not matching_files:
                    print(f"[警告] 在 {self.workflow_dir} 中未找到大小为 {size} 的工作流文件，跳过...")
                    continue
                
                # 优先选择compute_intensive_serial工作流，其次serial，再次highly_parallel，最后回退到第一个匹配文件
                compute_intensive_files = [f for f in matching_files if 'compute_intensive_serial' in f.name]
                if compute_intensive_files:
                    workflow_file_to_run = compute_intensive_files[0]
                    print(f"[信息] 优先使用计算密集型串行工作流: {workflow_file_to_run.name}")
                else:
                    serial_files = [f for f in matching_files if 'serial' in f.name]
                    if serial_files:
                        workflow_file_to_run = serial_files[0]
                        print(f"[信息] 优先使用串行工作流: {workflow_file_to_run.name}")
                    else:
                        highly_parallel_files = [f for f in matching_files if 'highly_parallel' in f.name]
                        if highly_parallel_files:
                            workflow_file_to_run = highly_parallel_files[0]
                            print(f"[信息] 优先使用高度并行工作流: {workflow_file_to_run.name}")
                        else:
                            workflow_file_to_run = matching_files[0]
                
                for rep in range(self.repetitions):
                    exp_count += 1
                    print(f"运行实验 [{exp_count}/{total_exps}]: {name} on {workflow_file_to_run.name}")
                    result = self._run_single_simulation(name, sched_impl, workflow_file_to_run)
                    results.append(result)
        return results

    def analyze_results(self, results: List[Dict[str, Any]]):
        """分析、打印并保存实验结果摘要。"""
        if not results:
            print("没有可供分析的实验结果。"); return

        df = pd.DataFrame(results)
        
        # 保存详细的原始结果
        detailed_csv_path = self.output_dir / "detailed_results.csv"
        df.to_csv(detailed_csv_path, index=False)
        print(f"✅ 详细实验结果已保存到: {detailed_csv_path}")

        summary = df.groupby('scheduler')['makespan'].agg(['mean', 'std', 'min', 'count']).reset_index()
        summary = summary.rename(columns={'scheduler': '调度器', 'mean': '平均Makespan', 'std': '标准差', 'min': '最佳', 'count': '实验次数'})

        print("\n" + "="*60); print("📈 实验结果分析:"); print(summary.to_string(index=False)); print("="*60 + "\n")
        
        # [FIX] 保存最终的摘要结果
        summary_csv_path = self.output_dir / "summary_results.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"✅ 实验结果摘要已保存到: {summary_csv_path}")