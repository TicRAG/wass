# src/utils.py
"""工具函数、日志设置和最终版的实验运行器。"""
from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Generator, List
from pathlib import Path
import sys
import random
import wrench
import pandas as pd
import numpy as np
import torch
import json
import importlib 

reference_flop_rate = '1Gf'  # or use '1000Mf' if preferred

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
        self.random_seeds = config.get("random_seeds")
        self.include_aug = bool(config.get("include_aug", False))

    def _load_workflows(self, workflow_dir: Path) -> List[str]:
        base = sorted(str(p) for p in workflow_dir.glob("*.json"))
        if not base:
            return []
        if self.include_aug:
            aug_dir = workflow_dir.parent / "training_aug"
            if aug_dir.exists():
                aug_files = sorted(str(p) for p in aug_dir.glob("*.json"))
                if aug_files:
                    base = list(dict.fromkeys(base + aug_files))
        return base

    def run_single_seeding_simulation(self, scheduler_class: Any, workflow_file: str, scheduler_kwargs: Dict[str, Any] | None = None) -> tuple[float, Dict[str, Any]]:
        """运行一次用于知识库“播种”的模拟。"""
        workflow_filename = Path(workflow_file).name
        scheduler_kwargs = scheduler_kwargs or {}
        try:
            with open(self.platform_file, "r") as f:
                platform_xml = f.read()
            simulation = wrench.Simulation()
            controller_host = "ControllerHost"
            simulation.start(platform_xml, controller_host)
            all_hostnames = simulation.get_all_hostnames()
            compute_hosts = [host for host in all_hostnames if host not in [controller_host, "StorageHost"]]
            storage_service = simulation.create_simple_storage_service("StorageHost", ["/storage"])
            min_host_speed = float(self.config.get("min_host_speed", 0.0) or 0.0)
            compute_services: Dict[str, Any] = {}
            hosts_properties: Dict[str, Dict[str, float]] = {}
            filtered_hosts: list[tuple[str, float]] = []
            for host in compute_hosts:
                service = simulation.create_bare_metal_compute_service(
                    host, {host: (-1, -1)}, "/scratch", {}, {})
                flop_rates = service.get_core_flop_rates()
                speed = list(flop_rates.values())[0] if flop_rates else 0.0
                speed_gf = speed / 1e9 if speed else 0.0
                if speed_gf < min_host_speed:
                    filtered_hosts.append((host, speed_gf))
                    continue
                compute_services[host] = service
                hosts_properties[host] = {"speed": speed}
            if not compute_services:
                raise RuntimeError(
                    "No compute hosts available after applying min_host_speed="
                    f"{min_host_speed}. Lower the threshold or adjust the platform configuration."
                )
            if filtered_hosts:
                skipped = ", ".join(f"{name}({speed:.2f} Gf/s)" for name, speed in filtered_hosts)
                print(f"⚖️  Filtered out slow hosts: {skipped}")
            
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
            scheduler_args.update(scheduler_kwargs)
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
            details: Dict[str, Any] = {}
            if hasattr(scheduler_instance, 'get_recorded_decisions'):
                details['decisions'] = scheduler_instance.get_recorded_decisions()
            if hasattr(scheduler_instance, 'get_knowledge_records'):
                try:
                    details['knowledge_records'] = scheduler_instance.get_knowledge_records(makespan)
                except TypeError:
                    # Fallback for implementations that do not accept makespan argument
                    details['knowledge_records'] = scheduler_instance.get_knowledge_records()
            if hasattr(scheduler_instance, 'get_potential_summary'):
                summary = scheduler_instance.get_potential_summary()
                if summary:
                    details['potential_summary'] = summary
            simulation.terminate()
            return makespan, details
        except Exception as e:
            import traceback
            print(f"ERROR running seeding simulation on {workflow_filename}: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return -1.0, {}

    def create_workflow_from_json_data(self, simulation, workflow_data, workflow_file_path):
        """从 JSON 数据创建 WRENCH 工作流对象。

        支持两种格式:
          1. 项目内部生成格式: workflow.tasks + metadata + workflow.files
          2. wfcommons 转换格式: workflow.specification.tasks + workflow.specification.files + workflow.execution.tasks

        对于格式2，直接使用其任务并映射 parents->children 关系；对于格式1，构造一个 wfcommons 兼容对象。
        """
        wf_section = workflow_data.get('workflow', {})
        # Ensure top-level workflow_name exists for WRENCH parser (some augmented files lack it)
        if 'workflow_name' not in workflow_data:
            workflow_data['workflow_name'] = workflow_data.get('name', workflow_file_path and Path(workflow_file_path).stem or 'unknown')
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

    def _run_single_simulation(self, scheduler_name: str, scheduler_impl: Any, workflow_file: str, seed: int | None = None, extra_kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """运行单个WRENCH模拟并返回结果。"""
        workflow_filename = Path(workflow_file).name
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            with open(self.platform_file, "r") as f:
                platform_xml = f.read()

            simulation = wrench.Simulation()
            controller_host = "ControllerHost"
            simulation.start(platform_xml, controller_host)

            all_hostnames = simulation.get_all_hostnames()
            compute_hosts_names = [h for h in all_hostnames if h not in [controller_host, "StorageHost"]]
            storage_service = simulation.create_simple_storage_service("StorageHost", ["/storage"])
            min_host_speed = float(self.config.get("min_host_speed", 0.0) or 0.0)
            compute_services: Dict[str, Any] = {}
            hosts_properties: Dict[str, Dict[str, float]] = {}
            filtered_hosts: list[tuple[str, float]] = []
            for host_name in compute_hosts_names:
                service = simulation.create_bare_metal_compute_service(
                    host_name, {host_name: (-1, -1)}, "/scratch", {}, {})
                flop_rates = service.get_core_flop_rates()
                speed = list(flop_rates.values())[0] if flop_rates else 0.0
                speed_gf = speed / 1e9 if speed else 0.0
                if speed_gf < min_host_speed:
                    filtered_hosts.append((host_name, speed_gf))
                    continue
                compute_services[host_name] = service
                hosts_properties[host_name] = {"speed": speed}
            if not compute_services:
                raise RuntimeError(
                    "No compute hosts available after applying min_host_speed="
                    f"{min_host_speed}. Lower the threshold or adjust the platform configuration."
                )
            if filtered_hosts:
                skipped = ", ".join(f"{name}({speed:.2f} Gf/s)" for name, speed in filtered_hosts)
                print(f"⚖️  Filtered out slow hosts: {skipped}")
            
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)
            
            workflow = self.create_workflow_from_json_data(simulation, workflow_data, workflow_file)
            
            scheduler_args = {
                "simulation": simulation,
                "compute_services": compute_services,
                "hosts": hosts_properties,
                "workflow_obj": workflow,
                "workflow_file": workflow_file,
            }

            if extra_kwargs:
                scheduler_args.update(extra_kwargs)
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
            result = {
                "scheduler": scheduler_name,
                "workflow": workflow_filename,
                "makespan": makespan,
                "status": "success",
                "task_count": task_count,
                "seed": seed,
            }
            if seed is not None:
                result["seed"] = seed
            return result
        except Exception as e:
            import traceback
            print(f"ERROR running {scheduler_name} on {workflow_filename}: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            result = {
                "scheduler": scheduler_name,
                "workflow": workflow_filename,
                "makespan": float('inf'),
                "status": "failed",
                "task_count": 0,
                "seed": seed,
            }
            return result

    def analyze_results(self, results: List[Dict[str, Any]]):
        """分析、打印并保存实验结果摘要。"""
        if not results:
            print("没有可供分析的实验结果。"); return
        df = pd.DataFrame(results)
        detailed_csv_path = self.output_dir / "detailed_results.csv"
        df.to_csv(detailed_csv_path, index=False)
        print(f"✅ 详细实验结果已保存到: {detailed_csv_path}")
        # 排除失败的运行，避免 inf 污染统计
        success_df = df[df['status'] == 'success'].copy()
        if success_df.empty:
            print("⚠️ 全部实验均失败，无法生成成功统计摘要。")
            summary = pd.DataFrame({
                '调度器': df['scheduler'].unique(),
                '平均Makespan': [float('inf')] * len(df['scheduler'].unique()),
                '标准差': [None] * len(df['scheduler'].unique()),
                '最佳': [float('inf')] * len(df['scheduler'].unique()),
                '实验次数': [0] * len(df['scheduler'].unique()),
                '成功次数': [0] * len(df['scheduler'].unique()),
                '失败次数': [len(df[df['scheduler'] == s]) for s in df['scheduler'].unique()],
            })
        else:
            summary = success_df.groupby('scheduler')['makespan'].agg(['mean', 'std', 'min', 'count']).reset_index()
            summary = summary.rename(columns={'scheduler': '调度器', 'mean': '平均Makespan', 'std': '标准差', 'min': '最佳', 'count': '实验次数'})
            # 附加成功/失败计数 (使用原始列名进行分组)
            success_counts = success_df.groupby('scheduler').size().to_dict()
            fail_counts = df[df['status'] != 'success'].groupby('scheduler').size().to_dict()
            summary['成功次数'] = summary['调度器'].map(success_counts).fillna(0).astype(int)
            summary['失败次数'] = summary['调度器'].map(fail_counts).fillna(0).astype(int)
        print("\n" + "="*60); print("📈 实验结果分析:"); print(summary.to_string(index=False)); print("="*60 + "\n")
        summary_csv_path = self.output_dir / "summary_results.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"✅ 实验结果摘要已保存到: {summary_csv_path}")