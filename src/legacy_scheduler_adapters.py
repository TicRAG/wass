"""Adapters providing legacy scheduler class names expected by evaluation script.

This decouples evaluate_paper_methods from deprecated experiments.wrench_real_experiment.
We wrap or re-implement minimal versions using wrench_schedulers where possible.
"""
from __future__ import annotations
import random
from typing import List, Dict, Any, Tuple

try:
    import wrench  # type: ignore
except Exception:  # pragma: no cover
    wrench = None  # type: ignore

# Reuse internal modern schedulers where possible
from src.wrench_schedulers import FIFOScheduler as _FIFONew, HEFTScheduler as _HEFTNew, WassHeuristicScheduler as _WassHeuristicNew

class _ShimBase:
    name: str = "Unnamed"
    def schedule_task(self, *args, **kwargs):  # Simple (task already chosen)
        raise NotImplementedError
    def choose(self, ready_tasks, available_nodes, node_caps, node_loads, compute_service):
        """Optional extended interface: return (task, node). Fallback uses first ready + schedule_task."""
        if not ready_tasks:
            return None, None
        t = ready_tasks[0]
        node = self.schedule_task(t, available_nodes, node_caps, node_loads, compute_service)
        return t, node
    def set_adjacency(self, adjacency):  # optional
        self._adjacency = adjacency
    def notify_task_completion(self, task_name: str, node: str):  # optional hook
        pass

class FIFOScheduler(_ShimBase):
    def __init__(self):
        self.name = "FIFO"
    def schedule_task(self, task, available_nodes: List[str], node_caps: Dict[str,float], node_loads: Dict[str,float], compute_service):
        # Simple FIFO: just pick first available node
        return available_nodes[0]

class HEFTScheduler(_ShimBase):
    def __init__(self):
        self.name = "HEFT"
        self._ranks: Dict[str, float] = {}
        self.data_location: Dict[str, str] = {}
        self.bandwidth = 2e8  # lower to amplify transfer cost

    def _average_exec_time(self, task, node_caps: Dict[str, float]):
        flops = getattr(task, 'get_flops', lambda: 1e9)()
        avg_cap = sum(node_caps.values()) / max(1, len(node_caps))
        return flops / max(1e-6, avg_cap * 1e9)

    def _compute_upward_ranks(self, tasks, node_caps):
        task_map = {t.get_name(): t for t in tasks}
        ranks: Dict[str, float] = {}

        def children_of(t):
            if hasattr(self, '_adjacency') and getattr(self, '_adjacency') and 'children' in self._adjacency:
                return [task_map[c] for c in self._adjacency['children'].get(t.get_name(), []) if c in task_map]
            return []

        def rank(name: str):
            if name in ranks:
                return ranks[name]
            t = task_map[name]
            child_max = 0.0
            for c in children_of(t):
                child_max = max(child_max, rank(c.get_name()))
            val = self._average_exec_time(t, node_caps) + child_max
            ranks[name] = val
            return val

        for t in tasks:
            try:
                rank(t.get_name())
            except Exception:
                ranks[t.get_name()] = self._average_exec_time(t, node_caps)
        return ranks

    def _eft(self, task, node, node_caps, node_loads):
        flops = getattr(task, 'get_flops', lambda: 1e9)()
        cap = node_caps.get(node, 1.0)
        exec_time = flops / max(1e-6, cap * 1e9)
        load = node_loads.get(node, 0.0)
        transfer = 0.0
        try:
            for f in task.get_input_files():
                fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                loc = self.data_location.get(fid)
                if loc is None or loc != node:
                    transfer += f.get_size() / self.bandwidth
        except Exception:
            pass
        return load + exec_time + transfer

    def choose(self, ready_tasks, available_nodes, node_caps, node_loads, compute_service):
        if not ready_tasks:
            return None, None
        all_tasks = list(ready_tasks)
        try:
            wf = getattr(ready_tasks[0], 'get_workflow', lambda: None)()
            if wf is not None and hasattr(wf, 'get_tasks'):
                cand = wf.get_tasks()
                if cand and not isinstance(cand[0], str):
                    all_tasks = cand
        except Exception:
            pass
        self._ranks = self._compute_upward_ranks(all_tasks, node_caps)
        chosen_task = max(ready_tasks, key=lambda t: self._ranks.get(t.get_name(), 0.0))
        best_node = None
        best_eft = 1e18
        for n in available_nodes:
            eft = self._eft(chosen_task, n, node_caps, node_loads)
            if eft < best_eft:
                best_eft = eft
                best_node = n
        return chosen_task, best_node or available_nodes[0]

    def schedule_task(self, task, available_nodes, node_caps, node_loads, compute_service):
        best_node = None
        best_eft = 1e18
        for n in available_nodes:
            eft = self._eft(task, n, node_caps, node_loads)
            if eft < best_eft:
                best_eft = eft
                best_node = n
        return best_node or available_nodes[0]

    def notify_task_completion(self, task_name: str, node: str):  # pragma: no cover
        pass

class WASSHeuristicScheduler(_ShimBase):
    def __init__(self, alpha: float=0.55, beta: float=0.20, gamma: float=0.15, delta: float=0.10, bandwidth: float=2e8):
        self.name = "WASS-Heuristic"
        self.alpha = alpha      # EFT weight
        self.beta = beta        # locality (transfer) weight
        self.gamma = gamma      # load balance weight
        self.delta = delta      # critical path reward weight
        self.bandwidth = bandwidth
        self.data_location: Dict[str,str] = {}
        self._ranks: Dict[str,float] = {}
        self._max_rank: float = 1.0

    # ---- Internal helpers ----
    def _transfer_time(self, task, node):
        max_t = 0.0
        try:
            for f in task.get_input_files():
                fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                loc = self.data_location.get(fid)
                if loc is None or loc != node:
                    max_t = max(max_t, f.get_size()/self.bandwidth)
        except Exception:
            pass
        return max_t

    def _eft(self, task, node, node_caps, node_loads):
        flops = getattr(task,'get_flops',lambda:1e9)()
        cap = node_caps.get(node,1.0)
        exec_time = flops / max(1e-6, cap*1e9)
        return node_loads.get(node,0.0) + exec_time

    def _compute_upward_ranks(self, tasks, node_caps):
        task_map = {t.get_name(): t for t in tasks}
        ranks: Dict[str,float] = {}

        def children_of(t):
            if hasattr(self,'_adjacency') and getattr(self,'_adjacency') and 'children' in self._adjacency:
                return [task_map[c] for c in self._adjacency['children'].get(t.get_name(), []) if c in task_map]
            try:
                return list(t.get_children())
            except Exception:
                return []

        def avg_exec(t):
            flops = getattr(t,'get_flops',lambda:1e9)()
            avg_cap = sum(node_caps.values())/max(1,len(node_caps))
            return flops / max(1e-6, avg_cap*1e9)

        def rank(name: str):
            if name in ranks: return ranks[name]
            t = task_map[name]
            succ_max = 0.0
            for c in children_of(t):
                try:
                    succ_max = max(succ_max, rank(c.get_name()))
                except Exception:
                    continue
            val = avg_exec(t) + succ_max
            ranks[name] = val
            return val
        for t in tasks:
            try: rank(t.get_name())
            except Exception: continue
        return ranks

    def _score(self, task, node, node_caps, node_loads, load_std):
        eft = self._eft(task, node, node_caps, node_loads)
        transfer = self._transfer_time(task, node)
        r = self._ranks.get(task.get_name(), 0.0)
        rank_term = (r / self._max_rank) if self._max_rank > 0 else 0.0
        return self.alpha*eft + self.beta*transfer + self.gamma*load_std - self.delta*rank_term

    # ---- Public scheduling interface ----
    def choose(self, ready_tasks, available_nodes, node_caps, node_loads, compute_service):
        if not ready_tasks:
            return None, None
        if not self._ranks:
            all_tasks = list(ready_tasks)
            try:
                wf = getattr(ready_tasks[0], 'get_workflow', lambda: None)()
                if wf is not None and hasattr(wf, 'get_tasks'):
                    cand = wf.get_tasks()
                    if cand and not isinstance(cand[0], str):
                        all_tasks = cand
            except Exception:
                pass
            self._ranks = self._compute_upward_ranks(all_tasks, node_caps)
            if self._ranks:
                self._max_rank = max(self._ranks.values())
        # Balance term
        loads = list(node_loads.values()) or [0.0]
        import math
        mean_l = sum(loads)/len(loads)
        load_var = sum((l-mean_l)**2 for l in loads)/len(loads)
        load_std = math.sqrt(load_var)
        best = (1e18, None, None)
        for t in ready_tasks:
            for n in available_nodes:
                sc = self._score(t, n, node_caps, node_loads, load_std)
                if sc < best[0]:
                    best = (sc, t, n)
        return best[1], best[2]

    def schedule_task(self, task, available_nodes, node_caps, node_loads, compute_service):
        loads = list(node_loads.values()) or [0.0]
        import math
        mean_l = sum(loads)/len(loads)
        load_var = sum((l-mean_l)**2 for l in loads)/len(loads)
        load_std = math.sqrt(load_var)
        best_sc = 1e18; best_node = None
        for n in available_nodes:
            sc = self._score(task, n, node_caps, node_loads, load_std)
            if sc < best_sc:
                best_sc = sc; best_node = n
        return best_node or available_nodes[0]

    def notify_task_completion(self, task_name: str, node: str):
        # Placeholder for adaptive adjustments (not used currently)
        pass

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
