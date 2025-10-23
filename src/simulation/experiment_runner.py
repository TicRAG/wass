# src/utils.py
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

reference_flop_rate = str(1000000000)+'Mf'

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
            
            workflow = self.create_workflow_from_json_data(simulation, workflow_data, workflow_file)
            
            scheduler_args = {
                "simulation": simulation,
                "compute_services": compute_services,
                "hosts": hosts_properties,
                "workflow_obj": workflow,
                "workflow_file": workflow_file
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
        """从 JSON 数据创建 WRENCH 工作流对象。

        支持两种格式:
          1. 项目内部生成格式: workflow.tasks + metadata + workflow.files
          2. wfcommons 转换格式: workflow.specification.tasks + workflow.specification.files + workflow.execution.tasks

        对于格式2，直接使用其任务并映射 parents->children 关系；对于格式1，构造一个 wfcommons 兼容对象。
        """
        wf_section = workflow_data.get('workflow', {})
        is_wfcommons_like = 'specification' in wf_section and 'tasks' in wf_section.get('specification', {})

        if is_wfcommons_like:
            # 已经是转换后的 wfcommons 格式 (scripts/0_convert_wfcommons.py 生成)
            spec = wf_section['specification']
            tasks = spec.get('tasks', [])
            # 构建 children 关系: 依据每个任务的 parents 列表
            task_children = {t.get('id'): [] for t in tasks if isinstance(t, dict)}
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                for parent_id in t.get('parents', []) or []:
                    if parent_id in task_children:
                        task_children[parent_id].append(t.get('id'))
            # 将 children 写入任务（不破坏原有结构）
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                tid = t.get('id')
                if tid in task_children:
                    t.setdefault('children', task_children[tid])
            wfcommons_data = workflow_data  # 直接使用
        else:
            # 原项目内部格式，需转换到 wfcommons 结构
            internal_tasks = wf_section.get('tasks', [])
            task_children = {t.get('id'): [] for t in internal_tasks if isinstance(t, dict)}
            for t in internal_tasks:
                if not isinstance(t, dict):
                    continue
                for parent_id in t.get('dependencies', []) or []:
                    if parent_id in task_children:
                        task_children[parent_id].append(t.get('id'))
            metadata = workflow_data.get('metadata', {})
            wfcommons_data = {
                'name': metadata.get('name', 'unknown'),
                'workflow_name': metadata.get('name', 'unknown'),
                'description': metadata.get('description', ''),
                'schemaVersion': '1.5',
                'author': {'name': 'WRENCH Experiment', 'email': 'wrench@example.com'},
                'createdAt': metadata.get('generated_at', '2024-01-01T00:00:00Z'),
                'workflow': {'specification': {'tasks': [], 'files': []}, 'execution': {'tasks': []}}}
            for t in internal_tasks:
                if not isinstance(t, dict):
                    continue
                tid = t.get('id')
                wfcommons_data['workflow']['specification']['tasks'].append({
                    'name': t.get('name', tid), 'id': tid, 'children': task_children.get(tid, []),
                    'parents': t.get('dependencies', []), 'inputFiles': t.get('input_files', []),
                    'outputFiles': t.get('output_files', []), 'flops': t.get('flops', 0),
                    'memory': t.get('memory', 0)})
                wfcommons_data['workflow']['execution']['tasks'].append({
                    'id': tid, 'runtimeInSeconds': t.get('runtime', 1.0), 'cores': 1, 'avgCPU': 1.0})
            for f in wf_section.get('files', []):
                if not isinstance(f, dict):
                    continue
                wfcommons_data['workflow']['specification']['files'].append({
                    'id': f.get('id'), 'name': f.get('name'), 'sizeInBytes': f.get('size', 0)})
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
            
            workflow = self.create_workflow_from_json_data(simulation, workflow_data, workflow_file)
            
            scheduler_args = {
                "simulation": simulation,
                "compute_services": compute_services,
                "hosts": hosts_properties,
                "workflow_obj": workflow
            }
            
            # --- 这是核心修复：确保所有DRL调度器都能收到 workflow_file ---
            if scheduler_name in ["WASS_DRL", "WASS_RAG"]:
                scheduler_args['workflow_file'] = workflow_file
            # --- 修复结束 ---

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
            wf_section = workflow_data.get('workflow', {})
            if 'tasks' in wf_section and isinstance(wf_section.get('tasks'), list):
                task_count_source = wf_section.get('tasks')
            else:
                spec_tasks = wf_section.get('specification', {}).get('tasks', [])
                task_count_source = spec_tasks if isinstance(spec_tasks, list) else []
            task_count = len(task_count_source)
            simulation.terminate()
            return {"scheduler": scheduler_name, "workflow": workflow_filename, "makespan": makespan, "status": "success", "task_count": task_count}
        except Exception as e:
            import traceback
            print(f"ERROR running {scheduler_name} on {workflow_filename}: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return {"scheduler": scheduler_name, "workflow": workflow_filename, "makespan": float('inf'), "status": "failed", "task_count": 0}

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