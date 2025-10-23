# src/wrench_schedulers.py
from __future__ import annotations
import wrench
from typing import Dict, Any
from pathlib import Path
import torch
from torch_geometric.data import Data
import random
import json

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

class BaseScheduler:
    """A final, fully compatible BaseScheduler."""
    def __init__(self, simulation: wrench.Simulation, compute_services: Dict[str, Any], hosts: Dict[str, Any], workflow_obj: wrench.Workflow, **kwargs):
        self.simulation = simulation
        self.compute_services = compute_services
        self.hosts = hosts
        self.extra_args = kwargs
        self.completed_tasks = set()

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service: wrench.StorageService):
        ready_tasks = workflow.get_ready_tasks()
        for task in ready_tasks:
            if task in self.completed_tasks:
                continue
            host_name = self.get_scheduling_decision(task, workflow)
            if host_name:
                file_locations = {}
                for f in task.get_input_files(): file_locations[f] = storage_service
                for f in task.get_output_files(): file_locations[f] = storage_service
                job = self.simulation.create_standard_job([task], file_locations)
                if host_name in self.compute_services:
                    self.compute_services[host_name].submit_standard_job(job)
                else:
                    list(self.compute_services.values())[0].submit_standard_job(job)

    def handle_completion(self, task: wrench.Task):
        self.completed_tasks.add(task)

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        raise NotImplementedError

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

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        best_host, earliest_finish_time = None, float('inf')
        for host_name, speed in self.host_speeds.items():
            compute_time = task.get_flops() / speed if speed > 0 else float('inf')
            finish_time = self.simulation.get_simulated_time() + compute_time
            if finish_time < earliest_finish_time:
                earliest_finish_time, best_host = finish_time, host_name
        return best_host

class RecordingHEFTScheduler(HEFTScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions = {}

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        best_host = super().get_scheduling_decision(task, workflow)
        speed = self.host_speeds.get(best_host, 1.0)
        compute_time = task.get_flops() / speed if speed > 0 else float('inf')
        finish_time = self.simulation.get_simulated_time() + compute_time
        self.decisions[task.get_name()] = {'host': best_host, 'finish_time': finish_time}
        return best_host

    def get_recorded_decisions(self):
        return self.decisions

class RecordingRandomScheduler(BaseScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host_list = list(self.hosts.keys())
        self.host_speeds = {name: props['speed'] for name, props in self.hosts.items()}
        self.decisions = {}

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        chosen_host = random.choice(self.host_list)
        speed = self.host_speeds.get(chosen_host, 1.0)
        compute_time = task.get_flops() / speed if speed > 0 else float('inf')
        finish_time = self.simulation.get_simulated_time() + compute_time
        self.decisions[task.get_name()] = {'host': chosen_host, 'finish_time': finish_time}
        return chosen_host

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

        expected_actions = state_dict.get('actor.4.weight')
        if expected_actions is None:
            raise RuntimeError(
                f"Checkpoint at {model_path} is missing 'actor.4.weight'; cannot infer required action dimension."
            )

        action_dim = len(self.hosts)
        expected_action_dim = expected_actions.shape[0]
        if expected_action_dim != action_dim:
            raise RuntimeError(
                "DRL agent action dimension mismatch: "
                f"checkpoint expects {expected_action_dim} compute hosts but platform exposes {action_dim}. "
                "Select a platform XML with the same host count used during training or retrain the agent."
            )

        self.gnn_encoder = GNNEncoder(DEFAULT_GNN_IN, DEFAULT_GNN_HIDDEN, DEFAULT_GNN_OUT)
        self.agent = ActorCritic(state_dim=DEFAULT_GNN_OUT, action_dim=action_dim)
        try:
            self.agent.load_state_dict(state_dict)
            self.agent.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load DRL agent model from {model_path}") from e
        self.pyg_data = workflow_json_to_pyg_data(self.workflow_file)
        self.task_name_to_idx = {t.get_name(): i for i, t in enumerate(workflow_obj.get_tasks().values())}
        self.STATUS_WAITING, self.STATUS_READY, self.STATUS_COMPLETED = 0.0, 1.0, 2.0
    
    def _update_state(self, workflow: wrench.Workflow):
        ready_tasks = {t.get_name() for t in workflow.get_ready_tasks()}
        completed_tasks = {t.get_name() for t in self.completed_tasks}
        for name, idx in self.task_name_to_idx.items():
            if name in completed_tasks: self.pyg_data.x[idx, 3] = self.STATUS_COMPLETED
            elif name in ready_tasks: self.pyg_data.x[idx, 3] = self.STATUS_READY
            else: self.pyg_data.x[idx, 3] = self.STATUS_WAITING

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service: wrench.StorageService):
        self._update_state(workflow)
        super().schedule_ready_tasks(workflow, storage_service)

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        state_embedding = self.gnn_encoder(self.pyg_data)
        action_index, _ = self.agent.act(state_embedding, deterministic=True)
        return list(self.hosts.keys())[action_index]

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
        self.host_speeds = {name: props['speed'] for name, props in self.hosts.items()}
        self.pyg_data = workflow_json_to_pyg_data(self.workflow_file, self.feature_scaler)
        self.task_name_to_idx = None
        self.STATUS_WAITING, self.STATUS_READY, self.STATUS_COMPLETED = 0.0, 1.0, 2.0

    def _update_and_get_state(self, workflow: wrench.Workflow) -> torch.Tensor:
        if self.task_name_to_idx is None:
            self.task_name_to_idx = {t.get_name(): i for i, t in enumerate(workflow.get_tasks().values())}
        ready_tasks = {t.get_name() for t in workflow.get_ready_tasks()}
        completed_tasks = {t.get_name() for t in self.completed_tasks}
        for name, idx in self.task_name_to_idx.items():
            if name in completed_tasks: self.pyg_data.x[idx, 3] = self.STATUS_COMPLETED
            elif name in ready_tasks: self.pyg_data.x[idx, 3] = self.STATUS_READY
            else: self.pyg_data.x[idx, 3] = self.STATUS_WAITING
        return self.policy_gnn_encoder(self.pyg_data)

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        state = self._update_and_get_state(workflow)
        action_index, action_logprob = self.agent.act(state, deterministic=False)
        chosen_host_name = list(self.hosts.keys())[action_index]
        host_speed = self.host_speeds.get(chosen_host_name, 1.0)
        compute_time = task.get_flops() / host_speed if host_speed > 0 else float('inf')
        agent_eft = self.simulation.get_simulated_time() + compute_time
        # Use frozen rag encoder embedding for teacher if provided; else reuse policy embedding
        if self.teacher:
            rag_state = self.rag_gnn_encoder(self.pyg_data) if self.rag_gnn_encoder else state
            reward = self.teacher.generate_rag_reward(rag_state, agent_eft, task.get_name())
        else:
            reward = 0.0
        self.replay_buffer.add(state, action_index, action_logprob, reward=reward)
        return chosen_host_name