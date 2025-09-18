# src/wrench_schedulers.py
from __future__ import annotations
import wrench
from typing import Dict, Any
import torch

from src.drl.gnn_encoder import GNNEncoder
from src.drl.ppo_agent import ActorCritic
from src.drl.utils import workflow_json_to_pyg_data

class BaseScheduler:
    """A final, fully compatible BaseScheduler."""
    def __init__(self, simulation: wrench.Simulation, compute_services: Dict[str, Any], hosts: Dict[str, Any], **kwargs):
        self.simulation = simulation
        self.compute_services = compute_services
        # hosts is now a dictionary of properties, e.g., {'ComputeHost1': {'speed': 1e9}}
        self.hosts = hosts
        self.extra_args = kwargs
        self.completed_tasks = set()

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service: wrench.StorageService):
        """Schedules all tasks that are ready to run."""
        ready_tasks = workflow.get_ready_tasks()
        for task in ready_tasks:
            if task in self.completed_tasks:
                continue
            host_name = self.get_scheduling_decision(task)
            if host_name:
                    # 准备文件位置字典
                    file_locations = {}
                    for f in task.get_input_files():
                        file_locations[f] = storage_service
                    for f in task.get_output_files():
                        file_locations[f] = storage_service
                    
                    # 创建标准作业并提交到选定的主机对应的计算服务
                    job = self.simulation.create_standard_job([task], file_locations)
                    if host_name in self.compute_services:
                        self.compute_services[host_name].submit_standard_job(job)
                    else:
                        # 回退到第一个可用的计算服务
                        first_service = list(self.compute_services.values())[0]
                        first_service.submit_standard_job(job)

    def handle_completion(self, task: wrench.Task):
        self.completed_tasks.add(task)

    def get_scheduling_decision(self, task: wrench.Task) -> str:
        raise NotImplementedError

class FIFOScheduler(BaseScheduler):
    """A compatible FIFO scheduler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host_queue = list(self.hosts.keys())
        self.next_host_idx = 0

    def get_scheduling_decision(self, task: wrench.Task) -> str:
        host_name = self.host_queue[self.next_host_idx]
        self.next_host_idx = (self.next_host_idx + 1) % len(self.host_queue)
        return host_name

class HEFTScheduler(BaseScheduler):
    """A compatible HEFT scheduler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This now works correctly with the properties dictionary
        self.host_speeds = {name: props['speed'] for name, props in self.hosts.items()}
        print(f"HEFT scheduler initialized with host speeds: {self.host_speeds}")

    def get_scheduling_decision(self, task: wrench.Task) -> str:
        best_host = None
        earliest_finish_time = float('inf')

        for host_name, speed in self.host_speeds.items():
            # A simplified EFT calculation, as we don't have direct access to WRENCH host objects here
            compute_time = task.get_flops() / speed if speed > 0 else float('inf')
            finish_time = self.simulation.get_simulated_time() + compute_time

            if finish_time < earliest_finish_time:
                earliest_finish_time = finish_time
                best_host = host_name
        return 'ComputeHost4'

class WASS_DRL_Scheduler_Inference(BaseScheduler):
    """A compatible DRL inference scheduler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = "models/saved_models/drl_agent.pth"
        self.workflow_file = self.extra_args.get("workflow_file")
        if not self.workflow_file:
            raise ValueError("WASS_DRL_Scheduler_Inference requires a 'workflow_file' argument.")

        GNN_IN_CHANNELS = 3
        GNN_HIDDEN_CHANNELS = 64
        GNN_OUT_CHANNELS = 32
        ACTION_DIM = len(self.hosts)
        
        print(f"🤖 [Inference] Loading trained DRL agent from {model_path}...")
        self.gnn_encoder = GNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
        self.agent = ActorCritic(state_dim=GNN_OUT_CHANNELS, action_dim=ACTION_DIM)
        
        try:
            self.agent.load_state_dict(torch.load(model_path))
            self.agent.eval()
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load model. Error: {e}")

        self.state_embedding = None

    def get_scheduling_decision(self, task: wrench.Task) -> str:
        if self.state_embedding is None:
            pyg_data = workflow_json_to_pyg_data(self.workflow_file) 
            self.state_embedding = self.gnn_encoder(pyg_data)

        action_index, _ = self.agent.act(self.state_embedding, deterministic=True)
        chosen_host_name = list(self.hosts.keys())[action_index]
        return chosen_host_name