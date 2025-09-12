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

    def get_earliest_finish_time(self, task, host_name):
        """计算任务在指定主机上的最早完成时间"""
        try:
            flops = task.get_flops()
            host_speed = self.hosts.get(host_name, [1, 10.0])[1]
            exec_time = flops / max(1e-6, host_speed)
            # 简化：假设当前时间为0，实际应该考虑主机当前负载
            return exec_time
        except Exception:
            return 10.0

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


class WassHeuristicScheduler(BaseScheduler):
    """WASS启发式调度器 - 在HEFT基础上加入数据局部性考虑"""
    
    def __init__(self, simulation, compute_service, hosts: Dict[str, Any], data_locality_weight: float = 0.5,
                 bandwidth: float = 1e9):
        super().__init__(simulation, compute_service)
        self.ranks: Dict[str, float] = {}
        self.hosts = hosts
        self.data_locality_weight = data_locality_weight  # w参数，平衡计算效率和数据局部性
        self._initialized = False
        # 模拟数据分布 - 实际应用中应该从真实系统获取
        self.data_location: Dict[str, str] = {}  # file_id -> host_name
        class WassHeuristicScheduler(BaseScheduler):
            """Improved WASS heuristic with correct EFT (parents finish + transfer + host availability)."""

            def __init__(self, simulation, compute_service, hosts: Dict[str, Any], data_locality_weight: float = 0.5,
                         bandwidth: float = 1e9):
                super().__init__(simulation, compute_service)
                self.hosts = hosts
                self.data_locality_weight = data_locality_weight
                self.bandwidth = bandwidth  # bytes/sec
                self._initialized = False
                # Rank state
                self.ranks: Dict[str, float] = {}
                # Data placement (file_id -> host)
                self.data_location: Dict[str, str] = {}
                # Resource / timing prediction state
                self.host_available_time: Dict[str, float] = {h: 0.0 for h in hosts}
                self.task_finish_times: Dict[str, float] = {}
                self.task_placement: Dict[str, str] = {}

            # ---------------- Rank Computation -----------------
            def _average_exec_time(self, task) -> float:
                flops = task.get_flops()
                total = 0.0
                for h in self.hosts:
                    try:
                        total += self.hosts[h][1]
                    except Exception:
                        total += 10.0
                avg = total / max(1, len(self.hosts))
                return flops / max(1e-6, avg)

            def _compute_upward_ranks(self, workflow):
                tasks = workflow.get_tasks()
                tmap = {t.get_name(): t for t in tasks}
                ranks: Dict[str, float] = {}

                def rank(name: str):
                    if name in ranks:
                        return ranks[name]
                    t = tmap[name]
                    succ_max = 0.0
                    for c in t.get_children():
                        succ_max = max(succ_max, rank(c.get_name()))
                    val = self._average_exec_time(t) + succ_max
                    ranks[name] = val
                    return val

                for t in tasks:
                    try:
                        rank(t.get_name())
                    except Exception:
                        continue
                return ranks

            # ---------------- Core Time Models -----------------
            def _exec_time(self, task, host: str) -> float:
                try:
                    flops = task.get_flops()
                    speed = self.hosts.get(host, [1, 10.0])[1]
                    return flops / max(1e-6, speed)
                except Exception:
                    return 10.0

            def _select_best_host(self, task):
                # Parent list
                try:
                    parents = list(task.get_parents())
                except Exception:
                    parents = []
                best_host = None
                best_score = float('inf')
                best_eft = 0.0
                for host in self.hosts:
                    # Data arrival (latest parent finish + transfer if needed)
                    max_dat = 0.0
                    for p in parents:
                        p_finish = self.task_finish_times.get(p.get_name(), 0.0)
                        transfer_time = 0.0
                        src_host = self.task_placement.get(p.get_name())
                        if src_host and src_host != host:
                            try:
                                for of in p.get_output_files():
                                    transfer_time = max(transfer_time, of.get_size() / self.bandwidth)
                            except Exception:
                                pass
                        max_dat = max(max_dat, p_finish + transfer_time)
                    host_avail = self.host_available_time.get(host, 0.0)
                    est = max(host_avail, max_dat)
                    eft = est + self._exec_time(task, host)
                    w = self.data_locality_weight
                    score = (1 - w) * eft + w * max_dat
                    if score < best_score:
                        best_score = score
                        best_host = host
                        best_eft = eft
                return best_host, best_eft

            # ---------------- Public API -----------------
            def submit_ready_tasks(self, workflow):
                if not self._initialized:
                    self.ranks = self._compute_upward_ranks(workflow)
                    self._initialize_data_locations(workflow)
                    self._initialized = True
                ready = workflow.get_runnable_tasks()
                ready_sorted = sorted(ready, key=lambda t: self.ranks.get(t.get_name(), 0.0), reverse=True)
                for task in ready_sorted:
                    if task.get_state_as_string() != 'NOT_SUBMITTED':
                        continue
                    host, eft = self._select_best_host(task)
                    if host is None:
                        continue
                    # (Best-effort) annotate preference
                    try:
                        if hasattr(task, 'set_property'):
                            task.set_property('preferred_host', host)  # type: ignore
                    except Exception:
                        pass
                    job = self.sim.create_standard_job([task], {})
                    self.cs.submit_standard_job(job)
                    self.host_available_time[host] = eft
                    self.task_finish_times[task.get_name()] = eft
                    self.task_placement[task.get_name()] = host
                    self._update_data_locations_after_task(task, host)

    # (Deprecated legacy helpers removed by refactor)
