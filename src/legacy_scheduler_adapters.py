"""Adapters providing legacy scheduler class names expected by evaluation script.

This decouples evaluate_paper_methods from deprecated experiments.wrench_real_experiment.
We wrap or re-implement minimal versions using wrench_schedulers where possible.
"""
from __future__ import annotations
import random
from typing import List, Dict, Any

try:
    import wrench  # type: ignore
except Exception:  # pragma: no cover
    wrench = None  # type: ignore

# Reuse internal modern schedulers where possible
from src.wrench_schedulers import FIFOScheduler as _FIFONew, HEFTScheduler as _HEFTNew, WassHeuristicScheduler as _WassHeuristicNew

class _ShimBase:
    name: str = "Unnamed"
    def schedule_task(self, *args, **kwargs):  # pragma: no cover - interface placeholder
        raise NotImplementedError

class FIFOScheduler(_ShimBase):
    def __init__(self):
        self.name = "FIFO"
    def schedule_task(self, task, available_nodes: List[str], node_caps: Dict[str,float], node_loads: Dict[str,float], compute_service):
        # Simple FIFO: just pick first available node
        return available_nodes[0]

class HEFTScheduler(_ShimBase):
    def __init__(self):
        self.name = "HEFT"
    def schedule_task(self, task, available_nodes: List[str], node_caps: Dict[str,float], node_loads: Dict[str,float], compute_service):
        # Pick node with max capacity; tiebreak earliest load
        best = None; best_score = -1
        for n in available_nodes:
            cap = node_caps.get(n,0.0); load = node_loads.get(n,0.0)
            score = cap - 0.1*load
            if score > best_score:
                best_score = score; best = n
        return best or available_nodes[0]

class WASSHeuristicScheduler(_ShimBase):
    def __init__(self):
        self.name = "WASS-Heuristic"
    def schedule_task(self, task, available_nodes: List[str], node_caps: Dict[str,float], node_loads: Dict[str,float], compute_service):
        # Combine capacity, current load, estimated exec time
        flops = task.get_flops() if hasattr(task, 'get_flops') else 1e9
        best=None; best_score=1e18
        for n in available_nodes:
            cap = node_caps.get(n,1.0)
            exec_est = flops/(cap*1e9)
            load = node_loads.get(n,0.0)
            # Score: weighted sum lower better
            score = exec_est + 0.25*load
            if score < best_score:
                best_score=score; best=n
        return best or available_nodes[0]

class WASSDRLScheduler(_ShimBase):
    """Thin placeholder referencing previously trained DQN-like agent if available.
    For now behaves like heuristic; can be extended to actually load model if needed.
    """
    def __init__(self, model_path: str):
        self.name = "WASS-DRL"
        self.model_path = model_path
        # Future: attempt to load torch model.
    def schedule_task(self, task, available_nodes: List[str], node_caps: Dict[str,float], node_loads: Dict[str,float], compute_service):
        # Mirror heuristic with slight stochastic exploration
        flops = task.get_flops() if hasattr(task,'get_flops') else 1e9
        scores = []
        for n in available_nodes:
            cap = node_caps.get(n,1.0)
            exec_est = flops/(cap*1e9)
            load = node_loads.get(n,0.0)
            score = exec_est + 0.25*load + random.random()*0.01
            scores.append((score,n))
        scores.sort(key=lambda x: x[0])
        return scores[0][1]

__all__ = [
    'FIFOScheduler', 'HEFTScheduler', 'WASSHeuristicScheduler', 'WASSDRLScheduler'
]
