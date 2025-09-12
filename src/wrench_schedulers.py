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
        # 记录主机资源（核心数, 内存）供后续构造job时“暗示”执行主机；需要compute_service资源键与hosts一致
        try:
            self._host_resource_specs = {h: self.hosts[h] for h in self.hosts}
        except Exception:
            self._host_resource_specs = {}
        # 新增：跟踪主机可用时间与任务完成时间，实现真实EFT
        self.host_available_time: Dict[str, float] = {h: 0.0 for h in self.hosts}
        self.task_finish_times: Dict[str, float] = {}
        self.bandwidth = bandwidth  # bytes/sec for transfer estimation

    def _average_exec_time(self, task):
        """计算任务的平均执行时间"""
        flops = task.get_flops()
        avg_speed = 0.0
        for h in self.hosts:
            try:
                speed = self.hosts[h][1]
            except Exception:
                speed = 10.0
            avg_speed += speed
        avg_speed = avg_speed / max(1, len(self.hosts))
        return flops / max(1e-6, avg_speed)

    def _compute_upward_ranks(self, workflow):
        """计算向上排名 - 复用HEFT逻辑"""
        tasks = workflow.get_tasks()
        task_map = {t.get_name(): t for t in tasks}
        ranks: Dict[str, float] = {}
        
        def rank(task_name: str):
            if task_name in ranks: 
                return ranks[task_name]
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

    def _execution_time(self, task, host_name: str) -> float:
        """估算在指定主机上的纯执行时间。"""
        try:
            flops = task.get_flops()
            host_speed = self.hosts.get(host_name, [1, 10.0])[1]
            return flops / max(1e-6, host_speed)
        except Exception:
            return 10.0

    def _data_ready_time(self, task, host_name: str) -> float:
        """所有输入在目标主机就绪的时间 (父任务完成 + 传输)。"""
        ready = 0.0
        try:
            for f in task.get_input_files():
                file_id = f.get_id()
                # 生产者完成时间
                # 简化：通过文件位置反查其生产主机的任务完成时间不可行 => 使用父任务 finish time
                prod_finish = 0.0
                try:
                    # 获取父任务最大完成时间
                    parents = list(task.get_parents()) if hasattr(task, 'get_parents') else []
                except Exception:
                    parents = []
                parent_max = 0.0
                for p in parents:
                    parent_max = max(parent_max, self.task_finish_times.get(p.get_name(), 0.0))
                prod_finish = parent_max
                # 若数据已在该主机，无需传输
                loc = self.data_location.get(file_id)
                if loc == host_name:
                    arrival = prod_finish
                else:
                    size_b = f.get_size()
                    transfer = size_b / self.bandwidth
                    arrival = prod_finish + transfer
                ready = max(ready, arrival)
            return ready
        except Exception:
            return ready

    def _earliest_finish_time(self, task, host_name: str) -> float:
        data_ready = self._data_ready_time(task, host_name)
        host_ready = self.host_available_time.get(host_name, 0.0)
        est = max(data_ready, host_ready)
        return est + self._execution_time(task, host_name)

    def _compute_wass_score(self, task, host_name):
        """使用真实EFT与数据就绪时间的组合分数。"""
        eft = self._earliest_finish_time(task, host_name)
        # 提取数据就绪与主机可用的组成部分用于 locality 权衡
        data_ready = self._data_ready_time(task, host_name)
        w = self.data_locality_weight
        # 分数越低越好：EFT 主导 + locality (较小 data_ready 提升)
        return (1 - w) * eft + w * data_ready

    def _select_best_host(self, task):
        best = (float('inf'), None, 0.0)
        for host_name in self.hosts:
            score = self._compute_wass_score(task, host_name)
            eft = self._earliest_finish_time(task, host_name)
            if score < best[0]:
                best = (score, host_name, eft)
        return best  # (score, host, eft)

    def submit_ready_tasks(self, workflow):
        """提交就绪任务 - 使用WASS启发式选择主机"""
        if not self._initialized:
            self.ranks = self._compute_upward_ranks(workflow)
            self._initialize_data_locations(workflow)
            self._initialized = True
        
        ready = workflow.get_runnable_tasks()
        # 按向上排名排序（复用HEFT的任务优先级）
        ready_sorted = sorted(ready, key=lambda t: self.ranks.get(t.get_name(), 0.0), reverse=True)
        
        for task in ready_sorted:
            if task.get_state_as_string() == 'NOT_SUBMITTED':
                score, best_host, eft = self._select_best_host(task)
                if best_host is None:
                    continue
                try:
                    if hasattr(task, 'set_property'):
                        task.set_property('preferred_host', best_host)  # type: ignore
                except Exception:
                    pass
                job = self.sim.create_standard_job([task], {})
                self.cs.submit_standard_job(job)
                # 更新预测状态
                finish_time = eft
                self.host_available_time[best_host] = finish_time
                self.task_finish_times[task.get_name()] = finish_time
                self._update_data_locations_after_task(task, best_host)

    def _initialize_data_locations(self, workflow):
        """初始化数据位置 - 简化模拟"""
        import random
        
        # 获取所有文件
        all_files = set()
        for task in workflow.get_tasks():
            for file in task.get_input_files():
                all_files.add(file.get_id())
            for file in task.get_output_files():
                all_files.add(file.get_id())
        
        # 随机分配初始数据位置
        host_names = list(self.hosts.keys())
        for file_id in all_files:
            self.data_location[file_id] = random.choice(host_names)

    def _update_data_locations_after_task(self, task, host_name):
        """任务执行后更新数据位置"""
        # 假设任务的输出文件会存储在执行主机上
        for file in task.get_output_files():
            self.data_location[file.get_id()] = host_name
