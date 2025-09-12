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
    """Refined WASS heuristic.

    Step 1: Integrate transfer time directly into EFT (so locality directly shortens predicted completion time).
    Step 2: Add explicit locality gain term rewarding placing task on node holding most of its inputs.

    score = exec_eft + w_bal*load_ratio - w_rank*norm_rank - w_loc*locality_gain
    Lower score better.
    """
    def __init__(self, w_loc: float = 0.40, w_rank: float = 0.25, w_bal: float = 0.05, bandwidth: float = 1e8,
                 transfer_aggregate: str = "sum", overlap: bool = False, overlap_factor: float = 0.5,
                 eft_mode: str = "predicted"):
        self.name = "WASS-Heuristic"
        self.w_loc = w_loc
        self.w_rank = w_rank
        self.w_bal = w_bal
        self.bandwidth = bandwidth
        self.transfer_aggregate = transfer_aggregate
        self.overlap = overlap
        self.overlap_factor = max(0.0, min(1.0, overlap_factor))
        self.data_location: Dict[str, str] = {}
        self._ranks: Dict[str, float] = {}
        self._max_rank = 1.0
        self._node_available: Dict[str, float] = {}
        # 新增：跟踪真实/预测 EFT 所需状态
        self.host_available_time: Dict[str, float] = {}
        self.task_finish_times: Dict[str, float] = {}
        self.eft_mode = eft_mode  # 'predicted' currently; hook for 'actual'

    # ---- Internal helpers ----
    def _transfer_time(self, task, node):
        total = 0.0; max_single = 0.0
        try:
            for f in task.get_input_files():
                fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                loc = self.data_location.get(fid)
                if loc is None or loc != node:
                    dt = f.get_size()/self.bandwidth
                    total += dt
                    if dt > max_single:
                        max_single = dt
        except Exception:
            pass
        raw = total if self.transfer_aggregate == "sum" else max_single
        if self.overlap:
            return raw * (1.0 - self.overlap_factor)
        return raw

    def _eft(self, task, node, node_caps, node_loads):
        # 旧（简化）EFT 与 新（包含主机可用与数据到达）的统一接口
        flops = getattr(task, 'get_flops', lambda: 1e9)()
        cap = node_caps.get(node, 1.0)
        exec_time = flops / max(1e-6, cap * 1e9)
        transfer = self._transfer_time(task, node)
        # 如果启用新的 EFT 推理，使用 host_available_time + 数据到达时间
        if self.eft_mode:
            data_ready = 0.0
            try:
                ins = task.get_input_files()
            except Exception:
                ins = []
            # 基于父任务完成时间 + 传输估算数据就绪
            parents = []
            try:
                parents = list(task.get_parents())
            except Exception:
                pass
            parent_finish_max = 0.0
            for p in parents:
                parent_finish_max = max(parent_finish_max, self.task_finish_times.get(p.get_name(), 0.0))
            for f in ins:
                fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                loc = self.data_location.get(fid)
                if loc == node:
                    arrival = parent_finish_max
                else:
                    arrival = parent_finish_max + f.get_size()/self.bandwidth
                data_ready = max(data_ready, arrival)
            host_ready = self.host_available_time.get(node, 0.0)
            est = max(host_ready, data_ready)
            return est + exec_time
        # fallback legacy predicted queue
        avail = self._node_available.get(node, 0.0)
        return max(avail, 0.0) + transfer + exec_time

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

    def _locality_gain(self, task, node):
        # Fan-out weighted fraction of inputs located on this node.
        try:
            ins = task.get_input_files()
            if not ins:
                return 0.0
            total_weight = 0.0
            local_weight = 0.0
            # Precompute downstream reuse (fan-out) via adjacency children mapping if available
            downstream_counts: Dict[str,int] = {}
            if hasattr(self,'_adjacency') and getattr(self,'_adjacency') and 'producers' in self._adjacency:
                # _adjacency['producers'][file_id] -> producing task; we derive consumers from 'file_consumers' if exists
                consumers = self._adjacency.get('file_consumers', {}) if isinstance(self._adjacency, dict) else {}
                for f in ins:
                    fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                    fan = len(consumers.get(fid, [])) if consumers else 1
                    downstream_counts[fid] = max(1, fan)
            for f in ins:
                fid = f.get_name() if hasattr(f,'get_name') else getattr(f,'name', str(f))
                fan_out = downstream_counts.get(fid, 1)
                w = 1.0 + 0.25 * (fan_out - 1)  # amplify weight for widely reused data
                total_weight += w
                if self.data_location.get(fid) == node:
                    local_weight += w
            if total_weight <= 0:
                return 0.0
            return local_weight / total_weight
        except Exception:
            return 0.0

    def _score(self, task, node, node_caps, node_loads):
        eft = self._eft(task, node, node_caps, node_loads)
        rank_val = self._ranks.get(task.get_name(), 0.0)
        rank_term = (rank_val / self._max_rank) if self._max_rank > 0 else 0.0
        node_load = node_loads.get(node, 0.0)
        max_load = max(node_loads.values()) if node_loads else 1.0
        load_ratio = node_load / max(1e-6, max_load)
        loc_gain = self._locality_gain(task, node)
        # Lower better: eft + small load penalty - rewards
        return eft + self.w_bal * load_ratio - self.w_rank * rank_term - self.w_loc * loc_gain

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
        # Ensure node availability map includes all nodes
        for n in available_nodes:
            self._node_available.setdefault(n, 0.0)
        best = (1e18, None, None, 0.0)
        for n in available_nodes:
            self.host_available_time.setdefault(n, 0.0)
        for t in ready_tasks:
            for n in available_nodes:
                sc = self._score(t, n, node_caps, node_loads)
                if sc < best[0]:
                    eft = self._eft(t, n, node_caps, node_loads)
                    best = (sc, t, n, eft)
        if best[2] is not None:
            # 记录预测 finish（供后续任务数据就绪 & 资源占用）
            self.host_available_time[best[2]] = best[3]
            self.task_finish_times[best[1].get_name()] = best[3]
        return best[1], best[2]

    def schedule_task(self, task, available_nodes, node_caps, node_loads, compute_service):
        best_sc = 1e18; best_node = None
        for n in available_nodes:
            sc = self._score(task, n, node_caps, node_loads)
            if sc < best_sc:
                best_sc = sc; best_node = n
        return best_node or available_nodes[0]

    def notify_task_completion(self, task_name: str, node: str):
            # 若未来需要用实际完成时间更新，可在此钩子调整 host_available_time (当前模拟预测即用)
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
