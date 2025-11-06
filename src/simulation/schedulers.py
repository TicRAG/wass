# src/wrench_schedulers.py
from __future__ import annotations
import wrench
from typing import Dict, Any, List, Tuple, Set
from pathlib import Path
import json
import torch
from torch_geometric.data import Data
import random
import numpy as np
import math
import time
from torch.distributions import Categorical

from src.drl.gnn_encoder import GNNEncoder
from src.drl.agent import ActorCritic
from src.drl.utils import workflow_json_to_pyg_data
from src.drl.replay_buffer import ReplayBuffer
from src.rag.teacher import KnowledgeableTeacher
from src.utils.config import load_training_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _resolve_model_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

try:
    _TRAINING_CONFIG = load_training_config()
except FileNotFoundError:
    _TRAINING_CONFIG = {}

_COMMON_CFG = _TRAINING_CONFIG.get("common", {})
_GNN_CFG = _COMMON_CFG.get("gnn", {})

DEFAULT_GNN_IN = _as_int(_GNN_CFG.get("in_channels"), 4)
DEFAULT_GNN_HIDDEN = _as_int(_GNN_CFG.get("hidden_channels"), 64)
DEFAULT_GNN_OUT = _as_int(_GNN_CFG.get("out_channels"), 32)

def _default_model_path(variant: str) -> Path:
    section_key = "rag_training" if variant == "rag" else "drl_training"
    section_cfg = _TRAINING_CONFIG.get(section_key, {})
    model_cfg = section_cfg.get("model", {})
    save_dir = model_cfg.get("save_dir", "models/saved_models")
    filename_default = "drl_agent.pth" if variant == "rag" else "drl_agent_no_rag.pth"
    filename = model_cfg.get("filename", filename_default)
    return _resolve_model_path(Path(save_dir) / filename)

_DEFAULT_MODEL_PATHS = {
    "rag": _default_model_path("rag"),
    "drl": _default_model_path("drl"),
}


def _populate_missing_file_sizes(
    file_sizes: Dict[str, float],
    tasks_section: List[dict[str, Any]] | None,
    default_per_file: float = 5e7,
) -> None:
    """Assign heuristic sizes to files that lack explicit metadata.

    Synthetic workflows often omit `sizeInBytes`, which disables any
    communication-aware behaviour in HEFT/MIN-MIN. This helper recovers a
    reasonable signal from runtime/compute hints so schedulers can
    differentiate host placement decisions.
    """
    if not tasks_section:
        return

    for task_entry in tasks_section:
        if not isinstance(task_entry, dict):
            continue
        output_files = list(task_entry.get("outputFiles") or [])
        if not output_files:
            continue

        runtime_candidates = []
        for key in ("runtime", "runtimeInSeconds"):
            value = task_entry.get(key)
            if value is not None:
                try:
                    runtime_candidates.append(float(value))
                except (TypeError, ValueError):
                    continue
        runtime_hint = max(runtime_candidates) if runtime_candidates else None

        flops_hint = task_entry.get("flops")
        if flops_hint is not None:
            try:
                flops_hint = float(flops_hint)
            except (TypeError, ValueError):
                flops_hint = None

        estimated_total_size = None
        if runtime_hint is not None and math.isfinite(runtime_hint) and runtime_hint > 0.0:
            estimated_total_size = runtime_hint * 5e7  # ~50 MB/s equivalent
        elif flops_hint is not None and math.isfinite(flops_hint) and flops_hint > 0.0:
            estimated_total_size = max(flops_hint / 20.0, default_per_file)

        if estimated_total_size is None or estimated_total_size <= 0.0:
            estimated_total_size = default_per_file * len(output_files)

        per_file_size = max(estimated_total_size / max(len(output_files), 1), default_per_file)
        for outfile in output_files:
            file_sizes.setdefault(outfile, per_file_size)


class KnowledgeRecordingMixin:
    STATUS_WAITING = 0.0
    STATUS_READY = 1.0
    STATUS_COMPLETED = 2.0

    def __init__(self, *args, **kwargs):
        self.knowledge_encoder = kwargs.pop('knowledge_encoder', None)
        self.feature_scaler = kwargs.pop('feature_scaler', None)
        super().__init__(*args, **kwargs)
        self._knowledge_records: list[dict[str, Any]] = []
        self._knowledge_graph: Data | None = None
        self._task_name_to_idx: dict[str, int] | None = None
        self.workflow_file = getattr(self, 'extra_args', {}).get('workflow_file')
        self._init_knowledge_graph()

    def _init_knowledge_graph(self) -> None:
        if not self.workflow_file or self.knowledge_encoder is None:
            return
        try:
            self._knowledge_graph = workflow_json_to_pyg_data(self.workflow_file, self.feature_scaler)
        except Exception as exc:
            print(f"[KnowledgeRecorder] Failed to build graph for {self.workflow_file}: {exc}")
            self._knowledge_graph = None

    def _prepare_graph_state(self, workflow: wrench.Workflow) -> Data | None:
        if self._knowledge_graph is None or self.knowledge_encoder is None:
            return None
        if self._task_name_to_idx is None:
            self._task_name_to_idx = {t.get_name(): i for i, t in enumerate(workflow.get_tasks().values())}
        ready_tasks = {t.get_name() for t in workflow.get_ready_tasks()}
        completed_tasks = {t.get_name() for t in getattr(self, 'completed_tasks', set())}
        for name, idx in self._task_name_to_idx.items():
            if name in completed_tasks:
                state_value = self.STATUS_COMPLETED
            elif name in ready_tasks:
                state_value = self.STATUS_READY
            else:
                state_value = self.STATUS_WAITING
            self._knowledge_graph.x[idx, 3] = state_value
        graph_snapshot = self._knowledge_graph.clone()
        graph_snapshot.x = self._knowledge_graph.x.clone()
        return graph_snapshot

    def _record_knowledge_state(
        self,
        workflow: wrench.Workflow,
        task: wrench.Task,
        decision_time: float,
        host_name: str,
        finish_time: float,
    ) -> None:
        graph_snapshot = self._prepare_graph_state(workflow)
        if graph_snapshot is None:
            return
        with torch.no_grad():
            embedding = self.knowledge_encoder(graph_snapshot).detach().cpu().numpy().flatten().astype(np.float32)
        host_props = self.hosts.get(host_name, {}) if isinstance(self.hosts, dict) else {}
        host_speed = float(host_props.get('speed', 0.0) or 0.0)
        task_flops = 0.0
        if hasattr(task, 'get_flops'):
            try:
                task_flops = float(task.get_flops() or 0.0)
            except (TypeError, ValueError):
                task_flops = 0.0
        compute_duration = max(float(finish_time) - float(decision_time), 0.0)
        host_type = host_props.get('type') if isinstance(host_props, dict) else None
        self._knowledge_records.append({
            'task_name': task.get_name(),
            'decision_time': float(decision_time),
            'host': host_name,
            'finish_time': float(finish_time),
            'embedding': embedding,
            'host_speed': host_speed,
            'task_flops': task_flops,
            'compute_duration': compute_duration,
            'host_type': host_type,
        })

    def get_knowledge_records(self, final_makespan: float | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in self._knowledge_records:
            remaining = None
            if final_makespan is not None:
                remaining = max(final_makespan - item['decision_time'], 0.0)
            record = {
                'task_name': item['task_name'],
                'decision_time': item['decision_time'],
                'host': item['host'],
                'finish_time': item['finish_time'],
                'embedding': item['embedding'].tolist(),
            }
            if remaining is not None:
                record['remaining_makespan'] = remaining
            host_speed = item.get('host_speed')
            if host_speed is not None:
                record['host_speed'] = float(host_speed)
            task_flops = item.get('task_flops')
            if task_flops is not None:
                record['task_flops'] = float(task_flops)
            compute_duration = item.get('compute_duration')
            if compute_duration is not None:
                record['compute_duration'] = float(compute_duration)
            host_type = item.get('host_type')
            if host_type is not None:
                record['host_type'] = host_type
            records.append(record)
        return records

class BaseScheduler:
    """A final, fully compatible BaseScheduler."""
    def __init__(self, simulation: wrench.Simulation, compute_services: Dict[str, Any], hosts: Dict[str, Any], workflow_obj: wrench.Workflow, **kwargs):
        self.simulation = simulation
        self.compute_services = compute_services
        self.hosts = hosts
        self.extra_args = kwargs
        self.completed_tasks = set()
        self._file_location_cache: Dict[Any, Any] = {}

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service: wrench.StorageService):
        ready_tasks = workflow.get_ready_tasks()
        for task in ready_tasks:
            if task in self.completed_tasks:
                continue
            host_name = self.get_scheduling_decision(task, workflow)
            if host_name:
                file_locations = self._prepare_file_locations(task, storage_service, host_name)
                job = self.simulation.create_standard_job([task], file_locations)
                if host_name in self.compute_services:
                    self.compute_services[host_name].submit_standard_job(job)
                else:
                    list(self.compute_services.values())[0].submit_standard_job(job)

    def handle_completion(self, task: wrench.Task):
        self.completed_tasks.add(task)

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        raise NotImplementedError

    def _prepare_file_locations(self, task: wrench.Task, storage_service: wrench.StorageService, host_name: str) -> Dict[Any, Any]:
        file_locations: Dict[Any, Any] = {}
        for file_obj in task.get_input_files():
            file_locations[file_obj] = storage_service
        for file_obj in task.get_output_files():
            file_locations[file_obj] = storage_service
        return file_locations

class FIFOScheduler(BaseScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host_queue = list(self.hosts.keys())
        self.next_host_idx = 0

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        host_name = self.host_queue[self.next_host_idx]
        self.next_host_idx = (self.next_host_idx + 1) % len(self.host_queue)
        return host_name

class HEFTScheduler(BaseScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host_speeds = {name: props['speed'] for name, props in self.hosts.items()}
        self.host_available_times = {name: 0.0 for name in self.hosts.keys()}
        self.scheduled_tasks: Set[str] = set()
        self.task_start_times: Dict[str, float] = {}
        self.task_finish_times: Dict[str, float] = {}
        self.task_host_assignment: Dict[str, str] = {}
        # Default to a conservative bandwidth (≈0.1 MB/s) so communication cost meaningfully
        # influences host selection unless the caller provides an explicit override.
        self.network_bandwidth_bytes = float(self.extra_args.get('network_bandwidth', 1e5))
        self.workflow_file_path = Path(self.extra_args.get('workflow_file', ''))
        self.noise_sigma = float(self.extra_args.get('noise_sigma', 0.0))
        self._cost_noise: Dict[tuple[str, str], float] = {}

        self.task_parents: Dict[str, List[str]] = {}
        self.task_children: Dict[str, List[str]] = {}
        self.task_input_files: Dict[str, List[str]] = {}
        self.task_output_files: Dict[str, List[str]] = {}
        self.task_flops: Dict[str, float] = {}
        self.task_runtime: Dict[str, float] = {}
        self.file_sizes: Dict[str, float] = {}
        self.edge_transfer_sizes: Dict[Tuple[str, str], float] = {}
        self._load_workflow_metadata()

        self.upward_rank: Dict[str, float] = {}
        for task_name in self.task_flops.keys():
            self._compute_upward_rank(task_name)

    def _load_workflow_metadata(self) -> None:
        if not self.workflow_file_path or not self.workflow_file_path.exists():
            return
        try:
            with self.workflow_file_path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
        except Exception:
            return

        spec = data.get('workflow', {}).get('specification', {})
        files_section = spec.get('files', []) or []
        for entry in files_section:
            fid = entry.get('id')
            if not fid:
                continue
            self.file_sizes[fid] = float(entry.get('sizeInBytes', 0.0))

        tasks_section = spec.get('tasks', []) or []
        for task_entry in tasks_section:
            task_id = task_entry.get('id')
            if not task_id:
                continue
            parents = list(task_entry.get('parents') or [])
            children = list(task_entry.get('children') or [])
            inputs = list(task_entry.get('inputFiles') or [])
            outputs = list(task_entry.get('outputFiles') or [])
            flops = float(task_entry.get('flops', 0.0))
            runtime = float(task_entry.get('runtime', task_entry.get('runtimeInSeconds', 0.0)))

            self.task_parents[task_id] = parents
            self.task_children[task_id] = children
            self.task_input_files[task_id] = inputs
            self.task_output_files[task_id] = outputs
            self.task_flops[task_id] = flops
            self.task_runtime[task_id] = runtime

        _populate_missing_file_sizes(self.file_sizes, tasks_section)

        for child, parents in self.task_parents.items():
            for parent in parents:
                transfer_size = 0.0
                parent_outputs = self.task_output_files.get(parent, [])
                child_inputs = set(self.task_input_files.get(child, []))
                for outfile in parent_outputs:
                    if outfile in child_inputs:
                        transfer_size += self.file_sizes.get(outfile, 0.0)
                if transfer_size == 0.0 and parent_outputs:
                    transfer_size = sum(self.file_sizes.get(ofile, 0.0) for ofile in parent_outputs)
                self.edge_transfer_sizes[(parent, child)] = transfer_size

    def _average_compute_cost(self, task_name: str) -> float:
        flops = self.task_flops.get(task_name, 0.0)
        if flops <= 0.0:
            runtime = self.task_runtime.get(task_name)
            return runtime if runtime is not None else 0.0
        valid_times = []
        for host_name, speed in self.host_speeds.items():
            if speed <= 0:
                continue
            base_time = flops / speed
            noise = self._get_noise_factor(task_name, host_name)
            valid_times.append(base_time * noise)
        if not valid_times:
            return 0.0
        return sum(valid_times) / len(valid_times)

    def _average_comm_cost(self, parent: str, child: str) -> float:
        size = self.edge_transfer_sizes.get((parent, child), 0.0)
        if size <= 0.0 or self.network_bandwidth_bytes <= 0.0:
            return 0.0
        return size / self.network_bandwidth_bytes

    def _compute_upward_rank(self, task_name: str) -> float:
        if task_name in self.upward_rank:
            return self.upward_rank[task_name]
        base_cost = self._average_compute_cost(task_name)
        children = self.task_children.get(task_name, [])
        if not children:
            rank_value = base_cost
        else:
            rank_value = base_cost + max(
                self._average_comm_cost(task_name, child) + self._compute_upward_rank(child)
                for child in children
            )
        self.upward_rank[task_name] = rank_value
        return rank_value

    def _get_comm_time(self, parent: str, child: str, host_name: str) -> float:
        parent_host = self.task_host_assignment.get(parent)
        if not parent_host or parent_host == host_name:
            return 0.0
        size = self.edge_transfer_sizes.get((parent, child), 0.0)
        if size <= 0.0 or self.network_bandwidth_bytes <= 0.0:
            return 0.0
        return size / self.network_bandwidth_bytes

    def _get_noise_factor(self, task_name: str, host_name: str) -> float:
        if self.noise_sigma <= 0.0:
            return 1.0
        key = (task_name, host_name)
        factor = self._cost_noise.get(key)
        if factor is None:
            sample = random.gauss(1.0, self.noise_sigma)
            factor = max(sample, 0.1)
            self._cost_noise[key] = factor
        return factor

    def _compute_time(self, task: wrench.Task, speed: float, host_name: str) -> float:
        flops = task.get_flops()
        if speed <= 0:
            return float('inf')
        if flops <= 0:
            runtime = self.task_runtime.get(task.get_name(), 0.0)
            return runtime
        base = flops / speed
        noise = self._get_noise_factor(task.get_name(), host_name)
        return base * noise

    def _select_host_for_task(self, task: wrench.Task) -> tuple[str | None, float, float]:
        task_name = task.get_name()
        best_host = None
        best_finish = float('inf')
        best_start = 0.0
        for host_name, speed in self.host_speeds.items():
            compute_time = self._compute_time(task, speed, host_name)
            if compute_time == float('inf'):
                continue
            parent_ready = 0.0
            for parent in self.task_parents.get(task_name, []):
                finish = self.task_finish_times.get(parent, 0.0)
                transfer = self._get_comm_time(parent, task_name, host_name)
                parent_ready = max(parent_ready, finish + transfer)
            earliest_start = max(self.host_available_times[host_name], parent_ready)
            finish_time = earliest_start + compute_time
            if finish_time < best_finish or (
                finish_time == best_finish and earliest_start < best_start
            ):
                best_finish = finish_time
                best_start = earliest_start
                best_host = host_name
        return best_host, best_start, best_finish

    def _on_task_assigned(
        self,
        workflow: wrench.Workflow,
        task: wrench.Task,
        host_name: str,
        start_time: float,
        finish_time: float,
    ) -> None:
        # Subclasses can override to record scheduling decisions.
        return

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service: wrench.StorageService):
        ready_tasks = [t for t in workflow.get_ready_tasks() if t.get_name() not in self.scheduled_tasks]
        if not ready_tasks:
            return
        ready_tasks.sort(key=lambda t: self.upward_rank.get(t.get_name(), 0.0), reverse=True)

        for task in ready_tasks:
            host_name, start_time, finish_time = self._select_host_for_task(task)
            if host_name is None:
                continue
            file_locations = self._prepare_file_locations(task, storage_service, host_name)
            job = self.simulation.create_standard_job([task], file_locations)
            if host_name in self.compute_services:
                self.compute_services[host_name].submit_standard_job(job)
            else:
                list(self.compute_services.values())[0].submit_standard_job(job)
            task_name = task.get_name()
            self.host_available_times[host_name] = finish_time
            self.task_start_times[task_name] = start_time
            self.task_finish_times[task_name] = finish_time
            self.task_host_assignment[task_name] = host_name
            self.scheduled_tasks.add(task_name)
            self._on_task_assigned(workflow, task, host_name, start_time, finish_time)

    def handle_completion(self, task: wrench.Task):
        super().handle_completion(task)
        self.scheduled_tasks.discard(task.get_name())


class RecordingHEFTScheduler(KnowledgeRecordingMixin, HEFTScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions = {}

    def _on_task_assigned(
        self,
        workflow: wrench.Workflow,
        task: wrench.Task,
        host_name: str,
        start_time: float,
        finish_time: float,
    ) -> None:
        self.decisions[task.get_name()] = {
            'host': host_name,
            'finish_time': finish_time,
            'decision_time': start_time,
        }
        if self.knowledge_encoder is not None:
            self._record_knowledge_state(workflow, task, start_time, host_name, finish_time)

    def get_recorded_decisions(self):
        return self.decisions

class RecordingRandomScheduler(KnowledgeRecordingMixin, BaseScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host_list = list(self.hosts.keys())
        self.host_speeds = {name: props['speed'] for name, props in self.hosts.items()}
        self.decisions = {}

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        decision_time = self.simulation.get_simulated_time()
        chosen_host = random.choice(self.host_list)
        speed = self.host_speeds.get(chosen_host, 1.0)
        compute_time = task.get_flops() / speed if speed > 0 else float('inf')
        finish_time = decision_time + compute_time
        self.decisions[task.get_name()] = {
            'host': chosen_host,
            'finish_time': finish_time,
            'decision_time': decision_time,
        }
        if self.knowledge_encoder is not None:
            self._record_knowledge_state(workflow, task, decision_time, chosen_host, finish_time)
        return chosen_host

    def get_recorded_decisions(self):
        return self.decisions


class MinMinScheduler(BaseScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host_speeds = {name: props['speed'] for name, props in self.hosts.items()}
        self.host_available_times = {name: 0.0 for name in self.hosts.keys()}
        self.host_assignment_counts = {name: 0 for name in self.hosts.keys()}
        self._inflight_assignments: dict[str, str] = {}
        self.task_host_assignment: dict[str, str] = {}
        self.completed_task_hosts: dict[str, str] = {}
        self.task_finish_times: dict[str, float] = {}
        self.workflow_file_path = Path(self.extra_args.get('workflow_file', ''))
        self.network_bandwidth_bytes = float(self.extra_args.get('network_bandwidth', 1e5))
        self.communication_scale = float(self.extra_args.get('communication_scale', 1.0))
        self.default_remote_penalty = float(self.extra_args.get('default_remote_penalty', 0.0))
        self.balance_weight = float(self.extra_args.get('balance_weight', 0.0))
        self.availability_weight = float(self.extra_args.get('availability_weight', 0.0))
        self.task_parents: Dict[str, List[str]] = {}
        self.task_output_files: Dict[str, List[str]] = {}
        self.task_input_files: Dict[str, List[str]] = {}
        self.file_sizes: Dict[str, float] = {}
        self.edge_transfer_sizes: Dict[Tuple[str, str], float] = {}
        self._load_workflow_metadata()

    def _load_workflow_metadata(self) -> None:
        if not self.workflow_file_path or not self.workflow_file_path.exists():
            return
        try:
            with self.workflow_file_path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
        except Exception:
            return

        spec = data.get('workflow', {}).get('specification', {})
        for entry in spec.get('files', []) or []:
            fid = entry.get('id')
            if not fid:
                continue
            self.file_sizes[fid] = float(entry.get('sizeInBytes', 0.0))

        tasks_section = spec.get('tasks', []) or []
        for task_entry in tasks_section:
            task_id = task_entry.get('id')
            if not task_id:
                continue
            self.task_parents[task_id] = list(task_entry.get('parents') or [])
            self.task_output_files[task_id] = list(task_entry.get('outputFiles') or [])
            self.task_input_files[task_id] = list(task_entry.get('inputFiles') or [])

        _populate_missing_file_sizes(self.file_sizes, tasks_section)

        for child, parents in self.task_parents.items():
            for parent in parents:
                transfer_size = 0.0
                parent_outputs = self.task_output_files.get(parent, [])
                child_inputs = set(self.task_input_files.get(child, []))
                for outfile in parent_outputs:
                    if outfile in child_inputs:
                        transfer_size += self.file_sizes.get(outfile, 0.0)
                if transfer_size == 0.0 and parent_outputs:
                    transfer_size = sum(self.file_sizes.get(ofile, 0.0) for ofile in parent_outputs)
                self.edge_transfer_sizes[(parent, child)] = transfer_size

    def _estimate_comm_penalty(self, task: wrench.Task, host_name: str) -> float:
        if not self.task_parents:
            return 0.0
        task_name = task.get_name()
        total_penalty = 0.0
        for parent in self.task_parents.get(task_name, []):
            parent_host = self.completed_task_hosts.get(parent) or self.task_host_assignment.get(parent)
            if not parent_host or parent_host == host_name:
                continue
            transfer_size = self.edge_transfer_sizes.get((parent, task_name), 0.0)
            if transfer_size > 0.0 and self.network_bandwidth_bytes > 0.0:
                total_penalty += (transfer_size / self.network_bandwidth_bytes) * self.communication_scale
            elif self.default_remote_penalty > 0.0:
                total_penalty += self.default_remote_penalty
        return total_penalty

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service: wrench.StorageService):
        ready_tasks = list(workflow.get_ready_tasks())
        if not ready_tasks:
            return

        current_time = self.simulation.get_simulated_time()
        assignments = []
        temp_available = self.host_available_times.copy()
        temp_assignment_counts = self.host_assignment_counts.copy()
        remaining_tasks = ready_tasks.copy()
        while remaining_tasks:
            best_assignment = None
            for task in remaining_tasks:
                task_flops = task.get_flops()
                for host_name, speed in self.host_speeds.items():
                    compute_time = task_flops / speed if speed > 0 else float('inf')
                    start_time = max(temp_available[host_name], current_time)
                    finish_time = start_time + compute_time + self._estimate_comm_penalty(task, host_name)
                    queue_penalty = self.balance_weight * temp_assignment_counts[host_name]
                    availability_penalty = self.availability_weight * max(temp_available[host_name] - current_time, 0.0)
                    score = finish_time + queue_penalty + availability_penalty
                    # Prefer hosts that can start earlier when finish times tie to avoid queueing on the first entry.
                    if (
                        best_assignment is None
                        or score < best_assignment['score']
                        or (
                            score == best_assignment['score']
                            and finish_time < best_assignment['finish_time']
                        )
                        or (
                            score == best_assignment['score']
                            and finish_time == best_assignment['finish_time']
                            and start_time < best_assignment['start_time']
                        )
                        or (
                            score == best_assignment['score']
                            and finish_time == best_assignment['finish_time']
                            and start_time == best_assignment['start_time']
                            and compute_time < best_assignment['compute_time']
                        )
                        or (
                            score == best_assignment['score']
                            and finish_time == best_assignment['finish_time']
                            and start_time == best_assignment['start_time']
                            and compute_time == best_assignment['compute_time']
                            and temp_assignment_counts[host_name] < temp_assignment_counts[best_assignment['host']]
                        )
                    ):
                        best_assignment = {
                            'task': task,
                            'host': host_name,
                            'compute_time': compute_time,
                            'start_time': start_time,
                            'finish_time': finish_time,
                            'score': score,
                        }
            if best_assignment is None:
                break
            assignments.append(best_assignment)
            temp_available[best_assignment['host']] = best_assignment['finish_time']
            temp_assignment_counts[best_assignment['host']] += 1
            remaining_tasks.remove(best_assignment['task'])

        for assignment in assignments:
            task = assignment['task']
            host_name = assignment['host']
            file_locations = self._prepare_file_locations(task, storage_service, host_name)
            job = self.simulation.create_standard_job([task], file_locations)
            if host_name in self.compute_services:
                self.compute_services[host_name].submit_standard_job(job)
            else:
                list(self.compute_services.values())[0].submit_standard_job(job)
            self.host_available_times[host_name] = assignment['finish_time']
            self.host_assignment_counts[host_name] += 1
            task_name = task.get_name()
            self.task_host_assignment[task_name] = host_name
            self.task_finish_times[task_name] = assignment['finish_time']
            self._inflight_assignments[task.get_name()] = host_name

    def handle_completion(self, task: wrench.Task):
        super().handle_completion(task)
        # Decrement outstanding assignments so future ties favour less loaded hosts.
        host = self._inflight_assignments.pop(task.get_name(), None)
        if host in self.host_assignment_counts:
            self.host_assignment_counts[host] = max(0, self.host_assignment_counts[host] - 1)
        if host:
            self.completed_task_hosts[task.get_name()] = host


class RecordingMinMinScheduler(KnowledgeRecordingMixin, MinMinScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions = {}

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service: wrench.StorageService):
        ready_tasks = list(workflow.get_ready_tasks())
        if not ready_tasks:
            return

        current_time = self.simulation.get_simulated_time()
        temp_available = self.host_available_times.copy()
        temp_assignment_counts = self.host_assignment_counts.copy()
        assignments = []
        remaining_tasks = ready_tasks.copy()

        while remaining_tasks:
            best_assignment = None
            for task in remaining_tasks:
                task_flops = task.get_flops()
                for host_name, speed in self.host_speeds.items():
                    compute_time = task_flops / speed if speed > 0 else float('inf')
                    start_time = max(temp_available[host_name], current_time)
                    finish_time = start_time + compute_time + self._estimate_comm_penalty(task, host_name)
                    queue_penalty = self.balance_weight * temp_assignment_counts[host_name]
                    availability_penalty = self.availability_weight * max(temp_available[host_name] - current_time, 0.0)
                    score = finish_time + queue_penalty + availability_penalty
                    if (
                        best_assignment is None
                        or score < best_assignment['score']
                        or (
                            score == best_assignment['score']
                            and finish_time < best_assignment['finish_time']
                        )
                        or (
                            score == best_assignment['score']
                            and finish_time == best_assignment['finish_time']
                            and start_time < best_assignment['start_time']
                        )
                        or (
                            score == best_assignment['score']
                            and finish_time == best_assignment['finish_time']
                            and start_time == best_assignment['start_time']
                            and compute_time < best_assignment['compute_time']
                        )
                        or (
                            score == best_assignment['score']
                            and finish_time == best_assignment['finish_time']
                            and start_time == best_assignment['start_time']
                            and compute_time == best_assignment['compute_time']
                            and temp_assignment_counts[host_name] < temp_assignment_counts[best_assignment['host']]
                        )
                    ):
                        best_assignment = {
                            'task': task,
                            'host': host_name,
                            'compute_time': compute_time,
                            'start_time': start_time,
                            'finish_time': finish_time,
                            'score': score,
                        }
            if best_assignment is None:
                break
            assignments.append(best_assignment)
            temp_available[best_assignment['host']] = best_assignment['finish_time']
            temp_assignment_counts[best_assignment['host']] += 1
            remaining_tasks.remove(best_assignment['task'])

        for assignment in assignments:
            task = assignment['task']
            host_name = assignment['host']
            task_name = task.get_name()
            file_locations = self._prepare_file_locations(task, storage_service, host_name)
            job = self.simulation.create_standard_job([task], file_locations)
            if host_name in self.compute_services:
                self.compute_services[host_name].submit_standard_job(job)
            else:
                list(self.compute_services.values())[0].submit_standard_job(job)
            self.host_available_times[host_name] = assignment['finish_time']
            self.decisions[task_name] = {
                'host': host_name,
                'finish_time': assignment['finish_time'],
                'decision_time': assignment['start_time'],
            }
            self.host_assignment_counts[host_name] += 1
            self.task_host_assignment[task_name] = host_name
            self.task_finish_times[task_name] = assignment['finish_time']
            self._inflight_assignments[task_name] = host_name
            if self.knowledge_encoder is not None:
                self._record_knowledge_state(
                    workflow,
                    task,
                    assignment['start_time'],
                    host_name,
                    assignment['finish_time'],
                )

    def get_recorded_decisions(self):
        return self.decisions

class WASS_DRL_Scheduler_Inference(BaseScheduler):
    def __init__(self, simulation: wrench.Simulation, compute_services: Dict[str, Any], hosts: Dict[str, Any], workflow_obj: wrench.Workflow, **kwargs):
        super().__init__(simulation, compute_services, hosts, workflow_obj, **kwargs)
        self.workflow_file = self.extra_args.get("workflow_file")
        if not self.workflow_file:
            raise ValueError("WASS_DRL_Scheduler_Inference requires a 'workflow_file' argument.")

        variant = self.extra_args.get("variant", "rag")
        model_override = self.extra_args.get("model_path")
        fallback_path = _DEFAULT_MODEL_PATHS.get(variant, _DEFAULT_MODEL_PATHS.get("rag"))
        model_path = _resolve_model_path(model_override) if model_override else fallback_path
        try:
            state_dict = torch.load(str(model_path), map_location=torch.device('cpu'))
        except Exception as e:
            raise RuntimeError(f"Failed to load DRL agent weights from {model_path}") from e

        # Dynamically infer actor output dimension by scanning keys for the final Linear layer weight.
        # Pattern: actor.*.weight where corresponding module is nn.Linear(out_features, ...)
        actor_weight_keys = [k for k in state_dict.keys() if k.startswith('actor.') and k.endswith('.weight')]
        if not actor_weight_keys:
            raise RuntimeError(f"Checkpoint at {model_path} contains no actor layer weights; cannot infer action dimension.")
        # Heuristic: the last sorted key is usually the final Linear before Softmax in nn.Sequential.
        actor_weight_keys.sort()
        final_actor_weight_key = actor_weight_keys[-1]
        expected_actions = state_dict[final_actor_weight_key]
        expected_action_dim = expected_actions.shape[0]

        action_dim = len(self.hosts)
        if expected_action_dim != action_dim:
            raise RuntimeError(
                "DRL agent action dimension mismatch: "
                f"checkpoint actor final layer out_features={expected_action_dim} but platform exposes hosts={action_dim}. "
                "Match host count used during training or retrain the agent with the new platform."
            )
        # (Optional future extension) Could allow smaller actor outputs and map only a subset of hosts.

        # Build full model including GNN so embeddings are consistent with training
        self.agent = ActorCritic(
            state_dim=DEFAULT_GNN_OUT,
            action_dim=action_dim,
            gnn_encoder=GNNEncoder(DEFAULT_GNN_IN, DEFAULT_GNN_HIDDEN, DEFAULT_GNN_OUT)
        )
        try:
            self.agent.load_state_dict(state_dict, strict=True)
            self.agent.eval()
            print(f"[Inference] Loaded full model (GNN+Actor+Critic) from {model_path}")
        except Exception as e:
            print(f"[Inference][ERROR] Strict load failed for {model_path}: {e}")
            missing = set(self.agent.state_dict().keys()) - set(state_dict.keys())
            unexpected = set(state_dict.keys()) - set(self.agent.state_dict().keys())
            print(f"  Missing keys: {missing}")
            print(f"  Unexpected keys: {unexpected}")
            raise RuntimeError("Failed to load full DRL agent with GNN weights.") from e
        # Load feature scaler for consistent preprocessing (fallback to None)
        scaler = None
        import joblib
        scaler_override = self.extra_args.get("feature_scaler")  # allow caller to inject
        if scaler_override is not None:
            scaler = scaler_override
        else:
            scaler_path = PROJECT_ROOT / "models/saved_models/feature_scaler.joblib"
            if scaler_path.exists():
                try:
                    scaler = joblib.load(scaler_path)
                    print(f"[Inference] Loaded feature scaler from {scaler_path}")
                except Exception as e:
                    print(f"[Inference][WARN] Failed to load scaler ({e}); proceeding without scaling.")
            else:
                print(f"[Inference] No feature scaler found at {scaler_path}; proceeding without scaling.")
        self.feature_scaler = scaler
        self.pyg_data = workflow_json_to_pyg_data(self.workflow_file, scaler)
        self.task_name_to_idx = {t.get_name(): i for i, t in enumerate(workflow_obj.get_tasks().values())}
        self.STATUS_WAITING, self.STATUS_READY, self.STATUS_COMPLETED = 0.0, 1.0, 2.0
        self.host_speeds = {name: props['speed'] for name, props in self.hosts.items()}
        # Debug instrumentation to inspect the exact host ordering assumed by the actor head.
        self._host_order = list(self.hosts.keys())
        print(f"[Inference][Debug] Host order ({len(self._host_order)}): {self._host_order}")
        self.teacher = self.extra_args.get("teacher")
        self.run_label = self.extra_args.get("run_label")
        self.seed = self.extra_args.get("seed")
        self.repeat_index = self.extra_args.get("repeat_index")
        self.strategy_label = self.extra_args.get("strategy_label")
        self.stochastic_tie_break = bool(self.extra_args.get("stochastic_tie_break", False))
        self.temperature = float(self.extra_args.get("temperature", 1.0))
        if self.temperature <= 0.0:
            self.temperature = 1.0
        self.greedy_threshold = float(self.extra_args.get("greedy_threshold", 1.1))
        self.epsilon = max(float(self.extra_args.get("epsilon", 0.0)), 0.0)
        sample_top_k = self.extra_args.get("sample_top_k")
        self.sample_top_k = int(sample_top_k) if sample_top_k is not None else None
        if self.sample_top_k is not None and self.sample_top_k < 1:
            self.sample_top_k = None
        self.pending_rewards: dict[str, dict[str, Any]] = {}
        self._latest_workflow = None
        self._potential_min = float('inf')
        self._potential_max = float('-inf')
        self._delta_min = float('inf')
        self._delta_max = float('-inf')
        self._potential_samples: list[dict[str, float | str]] = []
        self._max_potential_samples = 10
        self._debug_action_logs = 0  # limit action-probability prints to avoid log spam
        self._last_action_probs: list[float] | None = None
        if self.teacher is not None and hasattr(self.teacher, "set_trace_context"):
            self.teacher.set_trace_context(
                workflow_file=Path(self.workflow_file).name,
                run_label=self.run_label,
                seed=self.seed,
                repeat=self.repeat_index,
                strategy=self.strategy_label,
                trace_path=self.extra_args.get("trace_output_path"),
            )
        print(
            "[Inference][Debug] policy_flags: temperature=%.3f epsilon=%.3f greedy_threshold=%.3f sample_top_k=%s stochastic=%s"
            % (
                self.temperature,
                self.epsilon,
                self.greedy_threshold,
                str(self.sample_top_k),
                self.stochastic_tie_break,
            )
        )
    
    def _update_state(self, workflow: wrench.Workflow):
        self._latest_workflow = workflow
        ready_tasks = {t.get_name() for t in workflow.get_ready_tasks()}
        completed_tasks = {t.get_name() for t in self.completed_tasks}
        for name, idx in self.task_name_to_idx.items():
            if name in completed_tasks: self.pyg_data.x[idx, 3] = self.STATUS_COMPLETED
            elif name in ready_tasks: self.pyg_data.x[idx, 3] = self.STATUS_READY
            else: self.pyg_data.x[idx, 3] = self.STATUS_WAITING

    def _build_rag_graph(self) -> Data:
        rag_graph = self.pyg_data.clone()
        rag_graph.x = self.pyg_data.x.clone()
        return rag_graph

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service: wrench.StorageService):
        self._update_state(workflow)
        super().schedule_ready_tasks(workflow, storage_service)

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        # Use the agent's internal (trained) GNN encoder
        with torch.no_grad():
            state_embedding = self.agent.gnn_encoder(self.pyg_data)
            action_index = self._select_action(state_embedding)
        chosen_host_name = list(self.hosts.keys())[action_index]
        if self._last_action_probs is not None and self._debug_action_logs <= 5:
            prob_value = None
            if 0 <= action_index < len(self._last_action_probs):
                prob_value = self._last_action_probs[action_index]
            prob_str = f"{prob_value:.4f}" if isinstance(prob_value, (int, float)) else "n/a"
            print(
                f"[Inference][Debug] action_choice#{self._debug_action_logs}: "
                f"host={chosen_host_name} (idx={action_index}) prob={prob_str}"
            )
        if self.teacher is None:
            return chosen_host_name
        host_speed = self.host_speeds.get(chosen_host_name, 1.0)
        compute_time = task.get_flops() / host_speed if host_speed > 0 else float('inf')
        agent_eft = self.simulation.get_simulated_time() + compute_time
        rag_graph = self._build_rag_graph()
        current_potential = self.teacher.calculate_potential(rag_graph)
        self._potential_min = min(self._potential_min, current_potential)
        self._potential_max = max(self._potential_max, current_potential)
        last_payload = getattr(self.teacher, "_last_trace_payload", None)
        trace_snapshot = dict(last_payload) if isinstance(last_payload, dict) else None
        action_probs_snapshot = list(self._last_action_probs) if isinstance(self._last_action_probs, list) else None
        pending_entry: dict[str, Any] = {
            "potential_before": float(current_potential),
            "agent_eft": float(agent_eft),
            "host_name": chosen_host_name,
            "trace_payload": trace_snapshot,
        }
        if action_probs_snapshot is not None:
            pending_entry.update({
                "action_probs": action_probs_snapshot,
                "action_choice_index": int(action_index),
                "host_order": list(self._host_order),
            })
        self.pending_rewards[task.get_name()] = pending_entry
        return chosen_host_name

    def _select_action(self, state_embedding: torch.Tensor) -> int:
        action_probs = self.agent.actor(state_embedding).view(-1)
        if action_probs.numel() == 0:
            raise RuntimeError("Actor network produced empty action probabilities.")
        action_probs = torch.clamp(action_probs, min=1e-9)
        action_probs = action_probs / action_probs.sum()

        if self.temperature != 1.0:
            adjusted = torch.pow(action_probs, 1.0 / self.temperature)
            action_probs = adjusted / adjusted.sum()

        if self.sample_top_k is not None and self.sample_top_k < action_probs.numel():
            top_values, top_indices = torch.topk(action_probs, self.sample_top_k)
            masked = torch.zeros_like(action_probs)
            masked[top_indices] = top_values
            if masked.sum() > 0:
                action_probs = masked / masked.sum()

        if self.epsilon > 0.0:
            uniform = torch.full_like(action_probs, 1.0 / action_probs.numel())
            action_probs = (1.0 - self.epsilon) * action_probs + self.epsilon * uniform
            action_probs = action_probs / action_probs.sum()

        probs_snapshot = action_probs.detach().cpu().tolist()
        self._last_action_probs = probs_snapshot

        if self._debug_action_logs < 5:
            debug_entries = []
            for idx, (host, prob) in enumerate(zip(self._host_order, probs_snapshot)):
                speed = self.host_speeds.get(host)
                speed_str = f"{speed:.3f}" if isinstance(speed, (int, float)) else str(speed)
                debug_entries.append(f"{idx}:{host}@{speed_str}:{prob:.4f}")
            log_idx = self._debug_action_logs + 1
            print(f"[Inference][Debug] action_probs#{log_idx}: {', '.join(debug_entries)}")
        self._debug_action_logs += 1

        max_prob, max_idx = torch.max(action_probs, dim=0)
        if max_prob >= self.greedy_threshold:
            return int(max_idx.item())

        if self.stochastic_tie_break:
            dist = Categorical(action_probs)
            return int(dist.sample().item())

        return int(max_idx.item())

    def handle_completion(self, task: wrench.Task):
        super().handle_completion(task)
        if self.teacher is None:
            return
        pending = self.pending_rewards.pop(task.get_name(), None)
        if pending is None:
            return
        workflow = self._latest_workflow
        if workflow is None:
            return
        self._update_state(workflow)
        rag_graph_next = self._build_rag_graph()
        next_potential = self.teacher.calculate_potential(rag_graph_next)
        self._potential_min = min(self._potential_min, next_potential)
        self._potential_max = max(self._potential_max, next_potential)
        delta = next_potential - pending["potential_before"]
        self._delta_min = min(self._delta_min, delta)
        self._delta_max = max(self._delta_max, delta)
        shaped_reward = -self.teacher.lambda_scale * (
            self.teacher.gamma * next_potential - pending["potential_before"]
        )
        if len(self._potential_samples) < self._max_potential_samples:
            self._potential_samples.append({
                "task_name": task.get_name(),
                "phi_before": float(pending["potential_before"]),
                "phi_after": float(next_potential),
                "delta": float(delta),
                "reward": float(shaped_reward),
            })
        if getattr(self.teacher, "trace_logger", None) is not None:
            trace_payload = dict(pending.get("trace_payload") or {})
            trace_payload.update({
                "timestamp": time.time(),
                "task_name": task.get_name(),
                "decision_host": pending.get("host_name"),
                "agent_eft": float(pending.get("agent_eft", 0.0)),
                "potential_before": float(pending["potential_before"]),
                "potential_after": float(next_potential),
                "potential_delta": float(delta),
                "shaped_reward": float(shaped_reward),
                "lambda": float(self.teacher.lambda_scale),
                "gamma": float(self.teacher.gamma),
            })
            if "action_probs" in pending:
                trace_payload.update({
                    "action_probs": pending["action_probs"],
                    "action_choice_index": pending.get("action_choice_index"),
                    "host_order": pending.get("host_order"),
                })
            self.teacher.trace_logger.log(trace_payload)

    def get_potential_summary(self) -> dict[str, float | list[dict[str, float | str]]]:
        if self.teacher is None:
            return {}
        phi_min = None if self._potential_min == float('inf') else self._potential_min
        phi_max = None if self._potential_max == float('-inf') else self._potential_max
        delta_min = None if self._delta_min == float('inf') else self._delta_min
        delta_max = None if self._delta_max == float('-inf') else self._delta_max
        return {
            'phi_min': phi_min,
            'phi_max': phi_max,
            'delta_min': delta_min,
            'delta_max': delta_max,
            'samples': self._potential_samples,
        }

class WASS_RAG_Scheduler_Trainable(BaseScheduler):
    def __init__(
        self,
        simulation: wrench.Simulation,
        compute_services: Dict[str, Any],
        hosts: Dict[str, Any],
        workflow_obj: wrench.Workflow,
        **kwargs
    ):
        super().__init__(simulation, compute_services, hosts, workflow_obj, **kwargs)
        self.agent = self.extra_args['agent']
        self.teacher = self.extra_args['teacher']
        self.replay_buffer = self.extra_args['replay_buffer']
        # Dual encoders: policy (trainable) and rag (frozen). Fallback to policy if rag not provided.
        self.policy_gnn_encoder = self.extra_args.get('policy_gnn_encoder') or self.extra_args.get('gnn_encoder')
        self.rag_gnn_encoder = self.extra_args.get('rag_gnn_encoder')  # may be None
        self.workflow_file = self.extra_args.get("workflow_file")
        self.feature_scaler = self.extra_args.get('feature_scaler')
        if self.teacher is not None and hasattr(self.teacher, "set_trace_context"):
            self.teacher.set_trace_context(workflow_file=self.workflow_file)
        self.host_speeds = {name: props['speed'] for name, props in self.hosts.items()}
        # Debug instrumentation mirroring inference scheduler host ordering.
        self._host_order = list(self.hosts.keys())
        print(f"[Training][Debug] Host order ({len(self._host_order)}): {self._host_order}")
        self.pyg_data = workflow_json_to_pyg_data(self.workflow_file, self.feature_scaler)
        self.task_name_to_idx = None
        self.STATUS_WAITING, self.STATUS_READY, self.STATUS_COMPLETED = 0.0, 1.0, 2.0
        self.pending_rewards = {}
        self._latest_workflow = None
        self._potential_min = float('inf')
        self._potential_max = float('-inf')
        self._delta_min = float('inf')
        self._delta_max = float('-inf')
        self._potential_samples = []
        self._max_potential_samples = 10
        self._debug_action_logs = 0
        self._last_action_probs: list[float] | None = None

    def _update_and_get_state(self, workflow: wrench.Workflow) -> Data:
        self._latest_workflow = workflow
        if self.task_name_to_idx is None:
            self.task_name_to_idx = {t.get_name(): i for i, t in enumerate(workflow.get_tasks().values())}
        ready_tasks = {t.get_name() for t in workflow.get_ready_tasks()}
        completed_tasks = {t.get_name() for t in self.completed_tasks}
        for name, idx in self.task_name_to_idx.items():
            if name in completed_tasks: self.pyg_data.x[idx, 3] = self.STATUS_COMPLETED
            elif name in ready_tasks: self.pyg_data.x[idx, 3] = self.STATUS_READY
            else: self.pyg_data.x[idx, 3] = self.STATUS_WAITING
        return self.pyg_data

    def _build_rag_graph(self, graph_state: Data) -> Data:
        # Use the live graph snapshot directly so retrieval encoder observes task-progress markers.
        rag_graph = graph_state.clone()
        rag_graph.x = graph_state.x.clone()
        return rag_graph

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        graph_state = self._update_and_get_state(workflow)
        # Clone graph to produce embedding while keeping original for storage
        graph_for_embed = graph_state.clone()
        graph_for_embed.x = graph_state.x.clone()
        policy_emb = self.policy_gnn_encoder(graph_for_embed)
        action_probs = self.agent.actor(policy_emb).view(-1)
        probs_snapshot = action_probs.detach().cpu().tolist()
        self._last_action_probs = probs_snapshot
        if self._debug_action_logs < 5:
            debug_entries = []
            for idx, (host, prob) in enumerate(zip(self._host_order, probs_snapshot)):
                speed = self.host_speeds.get(host)
                speed_str = f"{speed:.3f}" if isinstance(speed, (int, float)) else str(speed)
                debug_entries.append(f"{idx}:{host}@{speed_str}:{prob:.4f}")
            log_idx = self._debug_action_logs + 1
            print(f"[Training][Debug] action_probs#{log_idx}: {', '.join(debug_entries)}")
        self._debug_action_logs += 1
        action_index, action_logprob = self.agent.act(policy_emb, deterministic=False)
        logprob_value = float(action_logprob.item()) if torch.is_tensor(action_logprob) else float(action_logprob)
        chosen_host_name = list(self.hosts.keys())[action_index]
        if self._last_action_probs is not None and self._debug_action_logs <= 5:
            prob_value = None
            if 0 <= action_index < len(self._last_action_probs):
                prob_value = self._last_action_probs[action_index]
            prob_str = f"{prob_value:.4f}" if isinstance(prob_value, (int, float)) else "n/a"
            print(
                f"[Training][Debug] action_choice#{self._debug_action_logs}: "
                f"host={chosen_host_name} (idx={action_index}) prob={prob_str} logprob={logprob_value:.4f}"
            )
        host_speed = self.host_speeds.get(chosen_host_name, 1.0)
        compute_time = task.get_flops() / host_speed if host_speed > 0 else float('inf')
        agent_eft = self.simulation.get_simulated_time() + compute_time
        # Potential-based reward will be finalized on completion; capture current potential now
        current_potential = 0.0
        trace_snapshot = None
        if self.teacher is not None:
            rag_graph = self._build_rag_graph(graph_state)
            current_potential = self.teacher.calculate_potential(rag_graph)
            self._potential_min = min(self._potential_min, current_potential)
            self._potential_max = max(self._potential_max, current_potential)
            last_payload = getattr(self.teacher, "_last_trace_payload", None)
            if isinstance(last_payload, dict):
                trace_snapshot = dict(last_payload)
        # Store raw graph state (not embedding) to allow PPO re-encoding with current GNN weights
        graph_snapshot = graph_state.clone()
        graph_snapshot.x = graph_state.x.clone()
        self.replay_buffer.add(graph_snapshot, action_index, action_logprob, reward=torch.tensor(0.0))
        buffer_index = len(self.replay_buffer.rewards) - 1
        pending_entry = {
            'buffer_index': buffer_index,
            'potential_before': current_potential,
            'agent_eft': float(agent_eft),
            'host_name': chosen_host_name,
            'trace_payload': trace_snapshot,
        }
        if self._last_action_probs is not None:
            pending_entry.update({
                'action_probs': list(self._last_action_probs),
                'action_choice_index': int(action_index),
                'action_logprob': logprob_value,
                'host_order': list(self._host_order),
            })
        self.pending_rewards[task.get_name()] = pending_entry
        return chosen_host_name

    def handle_completion(self, task: wrench.Task):
        super().handle_completion(task)
        pending = self.pending_rewards.pop(task.get_name(), None)
        if pending is None or self.teacher is None:
            return
        workflow = self._latest_workflow
        if workflow is None:
            return
        graph_state = self._update_and_get_state(workflow)
        rag_graph_next = self._build_rag_graph(graph_state)
        next_potential = self.teacher.calculate_potential(rag_graph_next)
        self._potential_min = min(self._potential_min, next_potential)
        self._potential_max = max(self._potential_max, next_potential)
        delta = next_potential - pending['potential_before']
        self._delta_min = min(self._delta_min, delta)
        self._delta_max = max(self._delta_max, delta)
        reward = -self.teacher.lambda_scale * (
            self.teacher.gamma * next_potential - pending['potential_before']
        )
        buffer_index = pending['buffer_index']
        if 0 <= buffer_index < len(self.replay_buffer.rewards):
            self.replay_buffer.rewards[buffer_index] = torch.tensor(reward, dtype=torch.float32)
        if len(self._potential_samples) < self._max_potential_samples:
            self._potential_samples.append({
                'task_name': task.get_name(),
                'phi_before': float(pending['potential_before']),
                'phi_after': float(next_potential),
                'delta': float(delta),
                'reward': float(reward),
            })
        if self.teacher is not None and getattr(self.teacher, 'trace_logger', None) is not None:
            base_payload = pending.get('trace_payload')
            trace_payload = dict(base_payload) if isinstance(base_payload, dict) else {}
            trace_payload.update({
                'timestamp': time.time(),
                'task_name': task.get_name(),
                'decision_host': pending.get('host_name'),
                'agent_eft': float(pending.get('agent_eft', 0.0)),
                'potential_before': float(pending['potential_before']),
                'potential_after': float(next_potential),
                'potential_delta': float(delta),
                'shaped_reward': float(reward),
                'lambda': float(self.teacher.lambda_scale),
                'gamma': float(self.teacher.gamma),
            })
            if 'action_probs' in pending:
                trace_payload.update({
                    'action_probs': pending['action_probs'],
                    'action_choice_index': pending.get('action_choice_index'),
                    'action_logprob': pending.get('action_logprob'),
                    'host_order': pending.get('host_order'),
                })
            self.teacher.trace_logger.log(trace_payload)

    def get_potential_summary(self) -> dict[str, float | list[dict[str, float | str]]]:
        if self.teacher is None:
            return {}
        phi_min = None if self._potential_min == float('inf') else self._potential_min
        phi_max = None if self._potential_max == float('-inf') else self._potential_max
        delta_min = None if self._delta_min == float('inf') else self._delta_min
        delta_max = None if self._delta_max == float('-inf') else self._delta_max
        return {
            'phi_min': phi_min,
            'phi_max': phi_max,
            'delta_min': delta_min,
            'delta_max': delta_max,
            'samples': self._potential_samples,
        }