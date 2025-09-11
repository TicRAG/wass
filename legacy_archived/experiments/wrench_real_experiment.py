"""LEGACY COPY

This is the archived original experiment driver (copied verbatim from previous location).
It remains for reference; new experiments should use:
  - scripts/train_wass_paper_aligned.py
  - scripts/evaluate_paper_methods.py

"""

# --- Original Content Below (unchanged) ---
#!/usr/bin/env python3
"""
基于WRENCH的真实WASS-RAG实验框架
使用训练好的模型在真实WRENCH环境中进行性能对比实验
"""

import sys
import os
import json
import time
import random
import numpy as np
import torch
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import yaml

try:
    import wrench
except ImportError:
    print("Error: WRENCH not available. Please install wrench-python-api.")
    sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, str(parent_dir))

def load_config(cfg_path: str) -> Dict:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if 'include' in cfg:
        base_dir = os.path.dirname(cfg_path)
        for include_file in cfg['include']:
            include_path = os.path.join(base_dir, include_file)
            if os.path.exists(include_path):
                with open(include_path, 'r', encoding='utf-8') as f:
                    include_cfg = yaml.safe_load(f) or {}
                    for key, value in include_cfg.items():
                        if key not in cfg:
                            cfg[key] = value
    return cfg

@dataclass
class WRENCHExperimentResult:
    scheduler_name: str
    workflow_id: str
    task_count: int
    dependency_count: int
    makespan: float
    cpu_utilization: Dict[str, float]
    task_execution_times: Dict[str, float]
    scheduling_decisions: List[Dict[str, Any]]
    experiment_metadata: Dict[str, Any]

class WRENCHScheduler:
    def __init__(self, name: str):
        self.name = name
    def schedule_task(self, task, available_nodes: List[str], node_capacities: Dict, node_loads: Dict, compute_service) -> str:
        raise NotImplementedError

class FIFOScheduler(WRENCHScheduler):
    def __init__(self):
        super().__init__("FIFO")
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        return min(available_nodes, key=lambda x: node_loads.get(x, 0))

class HEFTScheduler(WRENCHScheduler):
    def __init__(self):
        super().__init__("HEFT")
    def schedule_task(self, task, available_nodes, node_capacities, node_loads, compute_service):
        best_node = None; best_finish_time = float('inf')
        for node in available_nodes:
            capacity = node_capacities.get(node, 1.0)
            load = node_loads.get(node, 0.0)
            exec_time = task.get_flops() / (capacity * 1e9)
            finish_time = load + exec_time
            if finish_time < best_finish_time:
                best_finish_time = finish_time; best_node = node
        return best_node or available_nodes[0]

# (Truncated legacy remainder for brevity in archive; original logic retained in active history if needed.)
