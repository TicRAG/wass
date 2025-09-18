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
import importlib 

reference_flop_rate = str(300000000)+'Mf'

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

    def run_single_seeding_simulation(self, scheduler_class: Any, workflow_file: str) -> tuple[float, list]:
        """运行一次用于知识库“播种”的模拟。"""
        workflow_filename = Path(workflow_file).name
        try:
            with open(self.platform_file, "r") as f:
                platform_xml = f.read()
            simulation = wrench.Simulation()
            controller_host = "ControllerHost"
            simulation.start(platform_xml, controller_host)
            all_hostnames = simulation.get_all_hostnames()
            compute_hosts = [host for host in all_hostnames if host not in [controller_host, "StorageHost"]]
            storage_service = simulation.create_simple_storage_service("StorageHost", ["/storage"])
            compute_services = {}
            for host in compute_hosts:
                compute_services[host] = simulation.create_bare_metal_compute_service(
                    host, {host: (-1, -1)}, "/scratch", {}, {})
            hosts_properties = {}
            for name, service in compute_services.items():
                flop_rates = service.get_core_flop_rates()
                speed = list(flop_rates.values())[0] if flop_rates else 0.0
                hosts_properties[name] = {"speed": speed}
            
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)
            
            # 为 seeder 注入 workflow 对象 (保持接口一致性)
            workflow = self.create_workflow_from_json_data(simulation, workflow_data, workflow_file)
            
            scheduler_args = {
                "simulation": simulation,
                "compute_services": compute_services,
                "hosts": hosts_properties,
                "workflow_obj": workflow
            }
            scheduler_instance = scheduler_class(**scheduler_args)
            
            for file in workflow.get_input_files():
                storage_service.create_file_copy(file)
            if hasattr(scheduler_instance, 'schedule_ready_tasks'):
                scheduler_instance.schedule_ready_tasks(workflow, storage_service)
            while not workflow.is_done():
                event = simulation.wait_for_next_event()
                if event["event_type"] == "standard_job_completion":
                    job = event["standard_job"]
                    if hasattr(scheduler_instance, 'handle_completion'):
                        for task in job.get_tasks():
                            scheduler_instance.handle_completion(task)
                    if hasattr(scheduler_instance, 'schedule_ready_tasks'):
                        scheduler_instance.schedule_ready_tasks(workflow, storage_service)
                elif event["event_type"] == "simulation_termination":
                    break
            makespan = simulation.get_simulated_time()
            decisions = []
            if hasattr(scheduler_instance, 'get_recorded_decisions'):
                decisions = scheduler_instance.get_recorded_decisions()
            simulation.terminate()
            return makespan, decisions
        except Exception as e:
            import traceback
            print(f"ERROR running seeding simulation on {workflow_filename}: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return -1.0, []

    def create_workflow_from_json_data(self, simulation, workflow_data, workflow_file_path):
        """从JSON数据创建WRENCH工作流对象的辅助函数。"""
        task_children = {}
        for task in workflow_data['workflow']['tasks']:
            task_children[task['id']] = []
        for task in workflow_data['workflow']['tasks']:
            for parent_id in task.get('dependencies', []):
                if parent_id in task_children:
                    task_children[parent_id].append(task['id'])
        wfcommons_data = {
            'name': workflow_data['metadata']['name'],
            'workflow_name': workflow_data['metadata']['name'],
            'description': workflow_data['metadata']['description'],
            'schemaVersion': '1.5',
            'author': {'name': 'WRENCH Experiment', 'email': 'wrench@example.com'},
            'createdAt': workflow_data['metadata'].get('generated_at', '2024-01-01T00:00:00Z'),
            'workflow': {'specification': {'tasks': [], 'files': []}, 'execution': {'tasks': []}}}
        for task in workflow_data['workflow']['tasks']:
            wfcommons_data['workflow']['specification']['tasks'].append({
                'name': task['name'], 'id': task['id'], 'children': task_children[task['id']],
                'parents': task.get('dependencies', []), 'inputFiles': task.get('input_files', []),
                'outputFiles': task.get('output_files', []), 'flops': task.get('flops', 0),
                'memory': task.get('memory', 0)})
            wfcommons_data['workflow']['execution']['tasks'].append({
                'id': task['id'], 'runtimeInSeconds': task.get('runtime', 1.0), 'cores': 1, 'avgCPU': 1.0})
        for file in workflow_data['workflow']['files']:
            wfcommons_data['workflow']['specification']['files'].append({
                'id': file['id'], 'name': file['name'], 'sizeInBytes': file.get('size', 0)})
        if 'workflow_name' not in wfcommons_data:
            wfcommons_data['workflow_name'] = wfcommons_data.get('name', 'unknown')
        return simulation.create_workflow_from_json(
            wfcommons_data, reference_flop_rate=reference_flop_rate,
            ignore_machine_specs=False, redundant_dependencies=False,
            ignore_cycle_creating_dependencies=False, min_cores_per_task=1,
            max_cores_per_task=1, enforce_num_cores=True,
            ignore_avg_cpu=False, show_warnings=False)

    def _run_single_simulation(self, scheduler_name: str, scheduler_impl: Any, workflow_file: str) -> Dict[str, Any]:
        """运行单个WRENCH模拟并返回结果。"""
        workflow_filename = Path(workflow_file).name
        try:
            with open(self.platform_file, "r") as f:
                platform_xml = f.read()

            simulation = wrench.Simulation()
            controller_host = "ControllerHost"
            simulation.start(platform_xml, controller_host)

            all_hostnames = simulation.get_all_hostnames()
            compute_hosts_names = [h for h in all_hostnames if h not in [controller_host, "StorageHost"]]
            storage_service = simulation.create_simple_storage_service("StorageHost", ["/storage"])

            compute_services = {}
            for host_name in compute_hosts_names:
                compute_services[host_name] = simulation.create_bare_metal_compute_service(
                    host_name, {host_name: (-1, -1)}, "/scratch", {}, {})

            hosts_properties = {}
            for name, service in compute_services.items():
                flop_rates = service.get_core_flop_rates()
                speed = list(flop_rates.values())[0] if flop_rates else 0.0
                hosts_properties[name] = {"speed": speed}
            
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)
            
            # 创建 workflow 对象
            workflow = self.create_workflow_from_json_data(simulation, workflow_data, workflow_file)
            
            # 准备调度器参数
            scheduler_args = {
                "simulation": simulation,
                "compute_services": compute_services,
                "hosts": hosts_properties,
                "workflow_obj": workflow # 传递 workflow 对象
            }
            if scheduler_name == "WASS_DRL":
                scheduler_args['workflow_file'] = workflow_file

            scheduler_instance = scheduler_impl(**scheduler_args)

            for file in workflow.get_input_files():
                storage_service.create_file_copy(file)
            
            if hasattr(scheduler_instance, 'schedule_ready_tasks'):
                scheduler_instance.schedule_ready_tasks(workflow, storage_service)
            
            while not workflow.is_done():
                event = simulation.wait_for_next_event()
                if event["event_type"] == "standard_job_completion":
                    job = event["standard_job"]
                    if hasattr(scheduler_instance, 'handle_completion'):
                        for task in job.get_tasks():
                            scheduler_instance.handle_completion(task)
                    if hasattr(scheduler_instance, 'schedule_ready_tasks'):
                        scheduler_instance.schedule_ready_tasks(workflow, storage_service)
                elif event["event_type"] == "simulation_termination":
                    break
            
            makespan = simulation.get_simulated_time()
            simulation.terminate()
            return {"scheduler": scheduler_name, "workflow": workflow_filename, "makespan": makespan, "status": "success"}
        except Exception as e:
            import traceback
            print(f"ERROR running {scheduler_name} on {workflow_filename}: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return {"scheduler": scheduler_name, "workflow": workflow_filename, "makespan": float('inf'), "status": "failed"}

    def analyze_results(self, results: List[Dict[str, Any]]):
        """分析、打印并保存实验结果摘要。"""
        if not results:
            print("没有可供分析的实验结果。"); return
        df = pd.DataFrame(results)
        detailed_csv_path = self.output_dir / "detailed_results.csv"
        df.to_csv(detailed_csv_path, index=False)
        print(f"✅ 详细实验结果已保存到: {detailed_csv_path}")
        summary = df.groupby('scheduler')['makespan'].agg(['mean', 'std', 'min', 'count']).reset_index()
        summary = summary.rename(columns={'scheduler': '调度器', 'mean': '平均Makespan', 'std': '标准差', 'min': '最佳', 'count': '实验次数'})
        print("\n" + "="*60); print("📈 实验结果分析:"); print(summary.to_string(index=False)); print("="*60 + "\n")
        summary_csv_path = self.output_dir / "summary_results.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"✅ 实验结果摘要已保存到: {summary_csv_path}")