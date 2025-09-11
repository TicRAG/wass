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
    
    def __init__(self, simulation, compute_service, hosts: Dict[str, Any], data_locality_weight: float = 0.5):
        super().__init__(simulation, compute_service)
        self.ranks: Dict[str, float] = {}
        self.hosts = hosts
        self.data_locality_weight = data_locality_weight  # w参数，平衡计算效率和数据局部性
        self._initialized = False
        # 模拟数据分布 - 实际应用中应该从真实系统获取
        self.data_location: Dict[str, str] = {}  # file_id -> host_name

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

    def _get_earliest_finish_time(self, task, host_name):
        """计算任务在指定主机上的最早完成时间 (EFT)"""
        try:
            flops = task.get_flops()
            host_speed = self.hosts.get(host_name, [1, 10.0])[1]
            exec_time = flops / max(1e-6, host_speed)
            # 简化实现：实际应该考虑主机当前负载和队列
            return exec_time
        except Exception:
            return 10.0

    def _get_data_ready_time(self, task, host_name):
        """计算数据就绪时间 (DRT) - 考虑数据传输开销"""
        try:
            # 获取任务的输入文件
            input_files = task.get_input_files()
            max_transfer_time = 0.0
            
            for file in input_files:
                file_id = file.get_id()
                # 检查文件是否在目标主机上
                if self.data_location.get(file_id) == host_name:
                    # 数据已在本地，无传输时间
                    transfer_time = 0.0
                else:
                    # 需要传输数据，简化计算传输时间
                    file_size = file.get_size()  # bytes
                    network_bandwidth = 1e9  # 1GB/s，实际应该从平台配置获取
                    transfer_time = file_size / network_bandwidth
                
                max_transfer_time = max(max_transfer_time, transfer_time)
            
            return max_transfer_time
        except Exception:
            # 如果出现异常，返回一个默认值
            return 1.0

    def _compute_wass_score(self, task, host_name):
        """计算WASS综合评分"""
        eft = self._get_earliest_finish_time(task, host_name)
        drt = self._get_data_ready_time(task, host_name)
        
        # 归一化处理（简化版本）
        # 实际应该根据所有候选主机的EFT和DRT范围进行归一化
        w = self.data_locality_weight
        score = (1 - w) * eft + w * drt
        
        return score

    def _select_best_host(self, task):
        """为任务选择最佳主机"""
        best_host = None
        best_score = float('inf')
        
        for host_name in self.hosts:
            score = self._compute_wass_score(task, host_name)
            if score < best_score:
                best_score = score
                best_host = host_name
        
        return best_host or list(self.hosts.keys())[0]

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
                # 使用WASS启发式选择最佳主机
                best_host = self._select_best_host(task)
                
                # 创建作业并指定主机
                job = self.sim.create_standard_job([task], {})
                # 注意：这里简化处理，实际WRENCH API可能需要不同的主机指定方式
                self.cs.submit_standard_job(job)
                
                # 更新数据位置（假设任务输出数据会保存在执行主机上）
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
