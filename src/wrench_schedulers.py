"""Custom Python schedulers (FIFO, HEFT) implemented on top of wrench-python-api primitives.
These schedulers assume a workflow created via simulation.create_workflow() and tasks with standard attributes.
"""
from __future__ import annotations
from typing import List, Dict, Any
import heapq
import wrench

class BaseScheduler:
    def __init__(self, simulation: 'wrench.Simulation', compute_service):
        self.sim = simulation
        self.cs = compute_service  # Could be bare metal or VM compute service

    def submit_ready_tasks(self, workflow):
        raise NotImplementedError

class FIFOScheduler(BaseScheduler):
    def __init__(self, simulation, compute_service):
        super().__init__(simulation, compute_service)
        self.queue: List[Any] = []

    def submit_ready_tasks(self, workflow):
        # Enqueue any runnable tasks not yet submitted
        for task in workflow.get_runnable_tasks():
            if task not in self.queue:
                self.queue.append(task)
        # Simple FIFO: submit tasks in order
        new_queue = []
        for task in self.queue:
            if task.get_state_as_string() == 'NOT_SUBMITTED':
                job = self.sim.create_standard_job([task], {})
                self.cs.submit_standard_job(job)
            else:
                new_queue.append(task)
        self.queue = new_queue

class HEFTScheduler(BaseScheduler):
    def __init__(self, simulation, compute_service, hosts: Dict[str, Any]):
        super().__init__(simulation, compute_service)
        self.ranks: Dict[str, float] = {}
        self.hosts = hosts
        self._initialized = False

    def _average_exec_time(self, task):
        flops = task.get_flops()
        avg_speed = 0.0
        for h in self.hosts:
            # assume (cores, speed) mapping as created when service started
            try:
                speed = self.hosts[h][1]
            except Exception:
                speed = 10.0
            avg_speed += speed
        avg_speed = avg_speed / max(1, len(self.hosts))
        return flops / max(1e-6, avg_speed)

    def _compute_upward_ranks(self, workflow):
        tasks = workflow.get_tasks()  # returns list
        task_map = {t.get_name(): t for t in tasks}
        ranks: Dict[str, float] = {}
        def rank(task_name: str):
            if task_name in ranks: return ranks[task_name]
            t = task_map[task_name]
            succs = [c.get_name() for c in t.get_children()]
            max_succ = 0.0
            for s in succs:
                max_succ = max(max_succ, rank(s))
            r = self._average_exec_time(t) + max_succ
            ranks[task_name] = r
            return r
        for t in tasks:
            rank(t.get_name())
        return ranks

    def submit_ready_tasks(self, workflow):
        if not self._initialized:
            # gather host meta from compute service if possible (not strictly needed)
            self.ranks = self._compute_upward_ranks(workflow)
            self._initialized = True
        ready = workflow.get_runnable_tasks()
        # Select highest rank first
        ready_sorted = sorted(ready, key=lambda t: self.ranks.get(t.get_name(), 0.0), reverse=True)
        for task in ready_sorted:
            if task.get_state_as_string() == 'NOT_SUBMITTED':
                job = self.sim.create_standard_job([task], {})
                self.cs.submit_standard_job(job)
