# src/wrench_schedulers.py
from __future__ import annotations
import wrench
from typing import Dict, Any
from pathlib import Path
import torch
from torch_geometric.data import Data
import random
import json
import numpy as np

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
        self._knowledge_records.append({
            'task_name': task.get_name(),
            'decision_time': float(decision_time),
            'host': host_name,
            'finish_time': float(finish_time),
            'embedding': embedding,
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

class RecordingHEFTScheduler(KnowledgeRecordingMixin, HEFTScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions = {}

    def get_scheduling_decision(self, task: wrench.Task, workflow: wrench.Workflow) -> str:
        decision_time = self.simulation.get_simulated_time()
        best_host = super().get_scheduling_decision(task, workflow)
        speed = self.host_speeds.get(best_host, 1.0)
        compute_time = task.get_flops() / speed if speed > 0 else float('inf')
        finish_time = decision_time + compute_time
        self.decisions[task.get_name()] = {
            'host': best_host,
            'finish_time': finish_time,
            'decision_time': decision_time,
        }
        if self.knowledge_encoder is not None:
            self._record_knowledge_state(workflow, task, decision_time, best_host, finish_time)
        return best_host

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

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service: wrench.StorageService):
        ready_tasks = list(workflow.get_ready_tasks())
        if not ready_tasks:
            return

        current_time = self.simulation.get_simulated_time()
        temp_available = self.host_available_times.copy()
        assignments = []

        remaining_tasks = ready_tasks.copy()
        while remaining_tasks:
            best_assignment = None
            for task in remaining_tasks:
                task_flops = task.get_flops()
                for host_name, speed in self.host_speeds.items():
                    compute_time = task_flops / speed if speed > 0 else float('inf')
                    start_time = max(temp_available[host_name], current_time)
                    finish_time = start_time + compute_time
                    if best_assignment is None or finish_time < best_assignment['finish_time']:
                        best_assignment = {
                            'task': task,
                            'host': host_name,
                            'compute_time': compute_time,
                            'start_time': start_time,
                            'finish_time': finish_time,
                        }
            if best_assignment is None:
                break
            assignments.append(best_assignment)
            temp_available[best_assignment['host']] = best_assignment['finish_time']
            remaining_tasks.remove(best_assignment['task'])

        for assignment in assignments:
            task = assignment['task']
            host_name = assignment['host']
            file_locations = {}
            for f in task.get_input_files():
                file_locations[f] = storage_service
            for f in task.get_output_files():
                file_locations[f] = storage_service
            job = self.simulation.create_standard_job([task], file_locations)
            if host_name in self.compute_services:
                self.compute_services[host_name].submit_standard_job(job)
            else:
                list(self.compute_services.values())[0].submit_standard_job(job)
            self.host_available_times[host_name] = assignment['finish_time']


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
        assignments = []
        remaining_tasks = ready_tasks.copy()

        while remaining_tasks:
            best_assignment = None
            for task in remaining_tasks:
                task_flops = task.get_flops()
                for host_name, speed in self.host_speeds.items():
                    compute_time = task_flops / speed if speed > 0 else float('inf')
                    start_time = max(temp_available[host_name], current_time)
                    finish_time = start_time + compute_time
                    if best_assignment is None or finish_time < best_assignment['finish_time']:
                        best_assignment = {
                            'task': task,
                            'host': host_name,
                            'compute_time': compute_time,
                            'start_time': start_time,
                            'finish_time': finish_time,
                        }
            if best_assignment is None:
                break
            assignments.append(best_assignment)
            temp_available[best_assignment['host']] = best_assignment['finish_time']
            remaining_tasks.remove(best_assignment['task'])

        for assignment in assignments:
            task = assignment['task']
            host_name = assignment['host']
            task_name = task.get_name()
            file_locations = {}
            for f in task.get_input_files():
                file_locations[f] = storage_service
            for f in task.get_output_files():
                file_locations[f] = storage_service
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
        self.pyg_data = workflow_json_to_pyg_data(self.workflow_file, scaler)
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
        # Use the agent's internal (trained) GNN encoder
        state_embedding = self.agent.gnn_encoder(self.pyg_data)
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
        self.pending_rewards = {}
        self._latest_workflow = None
        self._potential_min = float('inf')
        self._potential_max = float('-inf')
        self._delta_min = float('inf')
        self._delta_max = float('-inf')
        self._potential_samples = []
        self._max_potential_samples = 10

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
        action_index, action_logprob = self.agent.act(policy_emb, deterministic=False)
        chosen_host_name = list(self.hosts.keys())[action_index]
        host_speed = self.host_speeds.get(chosen_host_name, 1.0)
        compute_time = task.get_flops() / host_speed if host_speed > 0 else float('inf')
        agent_eft = self.simulation.get_simulated_time() + compute_time
        # Potential-based reward will be finalized on completion; capture current potential now
        current_potential = 0.0
        if self.teacher is not None:
            rag_graph = self._build_rag_graph(graph_state)
            current_potential = self.teacher.calculate_potential(rag_graph)
            self._potential_min = min(self._potential_min, current_potential)
            self._potential_max = max(self._potential_max, current_potential)
        # Store raw graph state (not embedding) to allow PPO re-encoding with current GNN weights
        graph_snapshot = graph_state.clone()
        graph_snapshot.x = graph_state.x.clone()
        self.replay_buffer.add(graph_snapshot, action_index, action_logprob, reward=torch.tensor(0.0))
        buffer_index = len(self.replay_buffer.rewards) - 1
        self.pending_rewards[task.get_name()] = {
            'buffer_index': buffer_index,
            'potential_before': current_potential,
        }
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
        reward = self.teacher.lambda_scale * (self.teacher.gamma * next_potential - pending['potential_before'])
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