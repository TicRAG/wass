"""Wrench adapter abstraction (placeholder).

Goal: unify training and evaluation environments to eliminate train-test skew.

Design Sketch:
- Provide a thin wrapper around the real WRENCH Python API.
- Expose methods to: load platform, submit workflow, query task states, advance simulation step, collect metrics.
- Later, replace calls in experiments/wrench_real_experiment.py that use the local _simulate_wrench_execution.
"""
from __future__ import annotations
from typing import Any, List, Dict, Optional, Tuple, Iterable
from dataclasses import dataclass
import time

try:
    # If wrench bindings become available later, we import conditionally
    import wrench  # type: ignore
except Exception:  # noqa: E722
    wrench = None

@dataclass
class AdapterTask:
    id: str
    parents: List[str]
    children: List[str]
    computation_size: float
    is_critical_path: bool = False
    submit_time: float = 0.0
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

@dataclass
class AdapterNode:
    id: str
    speed: float
    current_load: float = 0.0
    available_time: float = 0.0
    historical_busy: float = 0.0

@dataclass
class AdapterStateSnapshot:
    current_time: float
    ready_tasks: List[AdapterTask]
    running_tasks: List[AdapterTask]
    completed_tasks: List[AdapterTask]
    nodes: List[AdapterNode]

class AdapterNotInitialized(Exception):
    pass

class WrenchAdapter:
    """Adapter skeleton for unifying training/evaluation.

    Responsibilities:
    - Initialize platform (real or mock fallback)
    - Submit workflow DAG
    - Provide incremental simulation stepping
    - Expose state snapshot for RL (ready set, node loads, timing)
    - Offer helper to build StepContext (to be consumed by reward module)
    """

    def __init__(self, platform_file: str, mock: bool = True):
        self.platform_file = platform_file
        self.mock = mock or (wrench is None)
        self.initialized = False
        self.controller = None  # real wrench controller placeholder
        self._workflow_handle: Optional[str] = None
        self._tasks: Dict[str, AdapterTask] = {}
        self._nodes: Dict[str, AdapterNode] = {}
        self._current_time: float = 0.0
        self._ready: List[str] = []
        self._running: List[str] = []
        self._completed: List[str] = []

    # --- Initialization & Workflow ---
    def initialize(self, nodes: Optional[List[Tuple[str, float]]] = None):
        if self.initialized:
            return
        if self.mock:
            # minimal platform from provided node specs
            spec = nodes or [(f"ComputeHost{i+1}", 2.0 + i * 0.5) for i in range(4)]
            for nid, speed in spec:
                self._nodes[nid] = AdapterNode(id=nid, speed=speed)
        else:
            # TODO: integrate real wrench platform load
            pass
        self.initialized = True

    def submit_workflow(self, tasks: Iterable[Dict[str, Any]]) -> str:
        if not self.initialized:
            raise AdapterNotInitialized("Adapter not initialized")
        for t in tasks:
            task = AdapterTask(
                id=t['id'],
                parents=t.get('parents', []),
                children=t.get('children', []),
                computation_size=float(t.get('computation_size', 1e9)),
                is_critical_path=bool(t.get('is_critical_path', False)),
                submit_time=self._current_time,
            )
            self._tasks[task.id] = task
        self._ready = [tid for tid, tk in self._tasks.items() if not tk.parents]
        self._workflow_handle = f"wf_{int(time.time())}"
        return self._workflow_handle

    # --- State Access ---
    def get_ready_tasks(self) -> List[AdapterTask]:
        return [self._tasks[tid] for tid in self._ready]

    def get_nodes(self) -> List[AdapterNode]:
        return list(self._nodes.values())

    def snapshot(self) -> AdapterStateSnapshot:
        return AdapterStateSnapshot(
            current_time=self._current_time,
            ready_tasks=self.get_ready_tasks(),
            running_tasks=[self._tasks[tid] for tid in self._running],
            completed_tasks=[self._tasks[tid] for tid in self._completed],
            nodes=self.get_nodes(),
        )

    # --- Scheduling & Simulation ---
    def schedule_task(self, task_id: str, node_id: str):
        if task_id not in self._ready:
            return False
        task = self._tasks[task_id]
        node = self._nodes[node_id]
        task.start_time = self._current_time
        exec_time = task.computation_size / max(node.speed, 1e-6)
        node.available_time = max(node.available_time, self._current_time) + exec_time
        node.current_load += 1.0
        node.historical_busy += exec_time
        self._running.append(task_id)
        self._ready.remove(task_id)
        # store expected finish on task for stepping logic
        task.finish_time = node.available_time
        return True

    def _update_ready_set(self):
        for tid, task in self._tasks.items():
            if tid in self._completed or tid in self._running:
                continue
            if all(p in self._completed for p in task.parents) and tid not in self._ready:
                self._ready.append(tid)

    def advance(self, delta: Optional[float] = None):
        if delta is not None:
            self._current_time += delta
        else:
            # advance to next finishing task
            running_tasks = [self._tasks[tid] for tid in self._running]
            if not running_tasks:
                return
            next_finish = min(t.finish_time for t in running_tasks if t.finish_time is not None)
            self._current_time = next_finish
        # move finished tasks
        still_running = []
        for tid in self._running:
            t = self._tasks[tid]
            if t.finish_time and t.finish_time <= self._current_time:
                self._completed.append(tid)
            else:
                still_running.append(tid)
        self._running = still_running
        # update node loads (simple decay model)
        for node in self._nodes.values():
            node.current_load = max(0.0, node.current_load - 0.1)
        self._update_ready_set()

    def is_finished(self) -> bool:
        return len(self._completed) == len(self._tasks) and len(self._tasks) > 0

    # --- Metrics ---
    def collect_metrics(self) -> Dict[str, Any]:
        makespan = self._current_time if self.is_finished() else None
        utilization = {n.id: n.historical_busy / max(self._current_time, 1e-6) for n in self._nodes.values()}
        return {
            'makespan': makespan,
            'current_time': self._current_time,
            'completed': len(self._completed),
            'total_tasks': len(self._tasks),
            'utilization': utilization,
        }

    # --- Helper for reward StepContext (approx) ---
    def build_step_context(self) -> Dict[str, Any]:
        ready = self.get_ready_tasks()
        nodes = self.get_nodes()
        return {
            'ready_task_count': len(ready),
            'total_nodes': len(nodes),
            'avg_queue_wait': sum(n.available_time for n in nodes) / max(1, len(nodes)),
            'queue_wait_baseline': max(1.0, sum(n.available_time for n in nodes) / max(1, len(nodes))),
            'node_busy_times': {n.id: n.historical_busy for n in nodes},
            'critical_path_progress': 0.0,  # placeholder until CP analysis integrated
        }

    # --- Reset ---
    def reset(self):
        self._current_time = 0.0
        self._ready.clear()
        self._running.clear()
        self._completed.clear()
        for n in self._nodes.values():
            n.current_load = 0.0
            n.available_time = 0.0
            n.historical_busy = 0.0

__all__ = ["WrenchAdapter"]
