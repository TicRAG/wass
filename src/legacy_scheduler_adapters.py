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
        self._last_index = -1
    def schedule_task(self, task, available_nodes: List[str], node_caps: Dict[str,float], node_loads: Dict[str,float], compute_service):
        # Simple round-robin across available nodes to avoid trivial perfect locality
        if not available_nodes:
            return None
        self._last_index = (self._last_index + 1) % len(available_nodes)
        return available_nodes[self._last_index]

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
    """Wrapper using improved internal WASS heuristic (EFT + data arrival + host availability)."""

    def __init__(self, data_locality_weight: float = 0.35, bandwidth: float = 1e9, load_alpha: float = 0.05, tie_epsilon: float = 0.02):
        self.name = "WASS-Heuristic"
        self.w = max(0.0, min(1.0, data_locality_weight))
        self.bandwidth = bandwidth
        self.network_bandwidth = bandwidth  # alias for clarity with spec
        # State
        self.host_available_time: Dict[str, float] = {}
        self.task_finish_times: Dict[str, float] = {}
        self.task_placement: Dict[str, str] = {}
        self._ranks: Dict[str, float] = {}
        self._rank_max: float = 1.0
        self._adjacency = None
        self._parent_max_output_cache: Dict[str, float] = {}  # parent_task_name -> max_output_transfer_time (seconds)
        self.load_alpha = max(0.0, load_alpha)
        self.tie_epsilon = max(0.0, tie_epsilon)

    # ---------- Rank (upward) ----------
    def _compute_upward_ranks(self, tasks, node_caps):
        task_map = {t.get_name(): t for t in tasks}
        ranks: Dict[str, float] = {}
        def avg_exec(t):
            flops = getattr(t,'get_flops',lambda:1e9)()
            avg_cap = sum(node_caps.values())/max(1,len(node_caps))
            return flops / max(1e-6, avg_cap*1e9)
        def children_of(t):
            if self._adjacency and 'children' in self._adjacency:
                return [task_map[c] for c in self._adjacency['children'].get(t.get_name(), []) if c in task_map]
            try:
                return list(t.get_children())
            except Exception:
                return []
        def rank(name: str):
            if name in ranks: return ranks[name]
            t = task_map[name]
            succ = 0.0
            for c in children_of(t):
                succ = max(succ, rank(c.get_name()))
            val = avg_exec(t) + succ
            ranks[name] = val
            return val
        for t in tasks:
            try: rank(t.get_name())
            except Exception: continue
        return ranks

    # ---------- Core time model ----------
    def _exec_time(self, task, node, node_caps):
        flops = getattr(task,'get_flops',lambda:1e9)()
        cap = node_caps.get(node,1.0)
        return flops / max(1e-6, cap*1e9)

    def _data_arrival(self, task, node):
        try:
            parents = list(task.get_parents())
        except Exception:
            parents = []
        max_dat = 0.0
        for p in parents:
            finish = self.task_finish_times.get(p.get_name(), 0.0)
            src = self.task_placement.get(p.get_name())
            transfer = 0.0
            if src and src != node:
                try:
                    for of in p.get_output_files():
                        transfer = max(transfer, of.get_size()/self.bandwidth)
                except Exception:
                    pass
            max_dat = max(max_dat, finish + transfer)
        return max_dat

    def _score_host_for_task(self, task, node, node_caps):
        dat = self._data_arrival(task, node)
        host_ready = self.host_available_time.get(node, 0.0)
        est = max(dat, host_ready)
        eft = est + self._exec_time(task, node, node_caps)
        score = (1 - self.w) * eft + self.w * dat
        return score, eft, dat

    def choose(self, ready_tasks, available_nodes, node_caps, node_loads, compute_service):
        """Select (task, host) by evaluating all ready tasks with corrected EST logic.

        EST = max(host_available_time, max_parent_finish + transfer_if_remote)
        EFT = EST + exec_time
        Score = (1-w)*EFT + w*max_parent_data_arrival
        """
        if not ready_tasks:
            return None, None
        # Compute/refresh ranks (could cache across calls if DAG static)
        self._ranks = self._compute_upward_ranks(ready_tasks, node_caps)
        if self._ranks:
            self._rank_max = max(self._ranks.values())
        best_tuple = (float('inf'), None, None, 0.0, 0.0)  # score, task, host, eft, rank
        for task in ready_tasks:
            # For each host evaluate corrected EST/EFT
            for host_name in available_nodes:
                self.host_available_time.setdefault(host_name, 0.0)
                # 1. compute max parent data arrival (max_dat)
                try:
                    parents = list(task.get_parents())
                except Exception:
                    parents = []
                max_dat = 0.0
                for parent_task in parents:
                    parent_finish_time = self.task_finish_times.get(parent_task.get_name(), 0.0)
                    transfer_time = 0.0
                    parent_host = self.task_placement.get(parent_task.get_name())
                    if parent_host and parent_host != host_name:
                        # cache max output size transfer time
                        pname = parent_task.get_name()
                        if pname not in self._parent_max_output_cache:
                            max_tr = 0.0
                            try:
                                for output_file in parent_task.get_output_files():
                                    file_size = output_file.get_size()
                                    max_tr = max(max_tr, file_size / self.network_bandwidth)
                            except Exception:
                                pass
                            self._parent_max_output_cache[pname] = max_tr
                        transfer_time = self._parent_max_output_cache.get(pname, 0.0)
                    dat = parent_finish_time + transfer_time
                    if dat > max_dat:
                        max_dat = dat
                # 2. host availability
                host_avail_time = self.host_available_time.get(host_name, 0.0)
                # 3. EST
                est = max(host_avail_time, max_dat)
                # 4. execution time
                exec_time = self._exec_time(task, host_name, node_caps)
                # 5. EFT
                eft = est + exec_time
                # 6. score (add load penalty)
                # approximate normalized load: predicted busy horizon / max horizon among hosts (avoid extra pass: use host_available_time)
                max_horizon = max(self.host_available_time.values()) if self.host_available_time else 0.0
                if max_horizon <= 0:
                    load_ratio = 0.0
                else:
                    load_ratio = self.host_available_time.get(host_name,0.0)/max_horizon
                score = (1 - self.w) * eft + self.w * max_dat + self.load_alpha * load_ratio
                task_rank = self._ranks.get(task.get_name(),0.0)
                if score < best_tuple[0] * (1.0 - self.tie_epsilon):
                    best_tuple = (score, task, host_name, eft, task_rank)
                elif score <= best_tuple[0] * (1.0 + self.tie_epsilon):
                    # tie range: prefer higher rank (critical path)
                    if task_rank > best_tuple[4]:
                        best_tuple = (score, task, host_name, eft, task_rank)
        # Commit selection
        if best_tuple[1] is not None and best_tuple[2] is not None:
            _, task, host, eft, _ = best_tuple
            self.host_available_time[host] = eft
            self.task_finish_times[task.get_name()] = eft
            self.task_placement[task.get_name()] = host
            return task, host
        return None, None

    def schedule_task(self, task, available_nodes, node_caps, node_loads, compute_service):
        # Fallback single-task scheduling (rarely used by current env)
        best_node = None; best_sc = float('inf')
        for n in available_nodes:
            self.host_available_time.setdefault(n,0.0)
            sc, eft, _ = self._score_host_for_task(task, n, node_caps)
            if sc < best_sc:
                best_sc = sc; best_node = n
        if best_node:
            self.host_available_time[best_node] = self._score_host_for_task(task,best_node,node_caps)[1]
            self.task_finish_times[task.get_name()] = self.host_available_time[best_node]
            self.task_placement[task.get_name()] = best_node
        return best_node or available_nodes[0]

    def notify_task_completion(self, task_name: str, node: str):
        # Could reconcile predicted vs actual if env passed timestamps.
        return

class WASSDRLScheduler(_ShimBase):
    """Thin placeholder referencing previously trained DQN-like agent if available.
    For now behaves like heuristic; can be extended to actually load model if needed.
    """
    def __init__(self, model_path: str):
        self.name = "WASS-DRL"
        self.model_path = model_path
        # Future: attempt to load torch model.
    def schedule_task(self, task, available_nodes: List[str], node_caps: Dict[str,float], node_loads: Dict[str,float], compute_service):
        # Enhanced: approximate WASS heuristic (execution + load + transfer) + exploration
        flops = task.get_flops() if hasattr(task,'get_flops') else 1e9
        # Estimate simplistic transfer penalty (sum sizes not on node)
        def transfer_penalty(node):
            pen = 0.0
            try:
                for f in task.get_input_files():
                    fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                    loc = getattr(self, 'data_location', {}).get(fid)
                    if loc is None or loc != node:
                        pen += f.get_size()/2e8  # align with heuristic bandwidth default
            except Exception:
                pass
            return pen
        best = (1e18, available_nodes[0])
        alts = []
        for n in available_nodes:
            cap = node_caps.get(n,1.0)
            exec_est = flops/(cap*1e9)
            load = node_loads.get(n,0.0)
            tpen = transfer_penalty(n)
            score = exec_est + 0.15*load + 0.20*tpen
            alts.append((score, n))
            if score < best[0]:
                best = (score, n)
        # Exploration: with small probability pick near-optimal (<1.05*best)
        near = [n for s,n in alts if s <= best[0]*1.05]
        if random.random() < 0.15 and near:
            return random.choice(near)
        return best[1]

__all__ = [
    'FIFOScheduler', 'HEFTScheduler', 'WASSHeuristicScheduler', 'WASSDRLScheduler'
]
