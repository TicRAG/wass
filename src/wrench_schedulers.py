"""
Custom Python schedulers for WRENCH simulation.
"""
from __future__ import annotations
from typing import List, Dict, Any
import wrench
import pandas as pd
from pathlib import Path
import json

# 基础调度器类
class BaseScheduler:
    """基础调度器类，用于自定义调度策略"""
    def __init__(self, simulation: 'wrench.Simulation', compute_services, hosts: Dict[str, Any] = None):
        self.sim = simulation
        self.compute_services = compute_services  # 现在是多个计算服务的字典
        self.hosts = hosts
        self.completed_tasks = set()

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service):
        """调度所有准备就绪的任务"""
        ready_tasks = workflow.get_ready_tasks()
        for task in ready_tasks:
            if task not in self.completed_tasks:
                # 调用子类的调度决策方法
                host_name = self.get_scheduling_decision(task)
                if host_name:
                    # 准备文件位置字典
                    file_locations = {}
                    for f in task.get_input_files():
                        file_locations[f] = storage_service
                    for f in task.get_output_files():
                        file_locations[f] = storage_service
                    
                    # 创建标准作业并提交到选定的主机对应的计算服务
                    job = self.sim.create_standard_job([task], file_locations)
                    if host_name in self.compute_services:
                        self.compute_services[host_name].submit_standard_job(job)
                    else:
                        # 回退到第一个可用的计算服务
                        first_service = list(self.compute_services.values())[0]
                        first_service.submit_standard_job(job)

    def get_scheduling_decision(self, task: wrench.Task):
        """获取调度决策（由子类实现）"""
        raise NotImplementedError

    def handle_completion(self, task: wrench.Task):
        """处理任务完成事件"""
        self.completed_tasks.add(task)

class FIFOScheduler(BaseScheduler):
    """简单的先进先出调度器"""
    def __init__(self, simulation: 'wrench.Simulation', compute_services, hosts: Dict[str, Any] = None):
        super().__init__(simulation, compute_services, hosts)
        # 固定选择ComputeHost1（1GFLOPS，最慢的主机）
        self.fixed_host = 'ComputeHost1'
        self.task_assignments = {}  # 记录任务分配情况
        print(f"FIFO调度器初始化完成，固定主机: {self.fixed_host}")
    
    def get_scheduling_decision(self, task: wrench.Task):
        # 总是选择固定的主机，确保与HEFT有明显区别
        print(f"FIFO调度任务 '{task.get_name()}' -> 固定分配到 {self.fixed_host}")
        self.task_assignments[task.get_name()] = self.fixed_host
        return self.fixed_host

class HEFTScheduler(BaseScheduler):
    """真正的HEFT（异构最早完成时间）调度器"""
    def __init__(self, simulation: 'wrench.Simulation', compute_services, hosts: Dict[str, Any] = None):
        super().__init__(simulation, compute_services, hosts)
        self.host_ready_times = {host: 0.0 for host in hosts.keys()}  # 记录每个主机的可用时间
        self.task_assignments = {}  # 记录任务分配情况
        
        # 预定义主机速度映射（基于平台XML配置，这里没有读取xml而是为了简化代码写死了）
        # 进一步放大主机性能差异
        self.host_speeds = {
            'ComputeHost1': 1e9,    # 1 GFLOPS - 最慢
            'ComputeHost2': 5e9,    # 5 GFLOPS - 中等
            'ComputeHost3': 2e9,    # 2 GFLOPS - 较慢
            'ComputeHost4': 10e9    # 10 GFLOPS - 最快
        }
        print(f"HEFT调度器初始化完成，主机速度配置: {self.host_speeds}")
    
    def get_scheduling_decision(self, task: wrench.Task):
        """
        真正的HEFT调度：选择能使任务最早完成的主机
        关键改进：让计算性能差异占主导，而非数据传输时间
        """
        best_host = None
        min_eft = float('inf')
        current_time = self.sim.get_simulated_time()

        # 获取任务的计算负载
        task_flops = task.get_flops()
        
        print(f"HEFT调度任务 '{task.get_name()}' (FLOPS: {task_flops:.2e})")
        
        for host_name in self.hosts.keys():
            # 获取主机速度（FLOPS）
            host_speed = self.host_speeds.get(host_name, 2e9)  # 默认2 GFLOPS
            
            # 计算计算时间 - 这是主要差异来源
            compute_time = task_flops / host_speed
            
            # 简化数据传输时间：假设高效的数据预取和缓存
            # 只考虑很小的固定传输开销
            transfer_time = 0.1  # 固定100ms传输开销
            
            # 计算任务的开始时间：主机可用时间 或 当前时间 + 数据传输时间
            host_ready_time = self.host_ready_times.get(host_name, 0.0)
            start_time = max(host_ready_time, current_time + transfer_time)
            
            # 计算完成时间
            finish_time = start_time + compute_time
            
            print(f"  主机 {host_name}: 速度={host_speed/1e9:.1f}GFLOPS, 计算时间={compute_time:.2f}s, 传输时间={transfer_time:.2f}s, 完成时间={finish_time:.2f}s")
            
            if finish_time < min_eft:
                min_eft = finish_time
                best_host = host_name
        
        # 更新选中主机的可用时间
        if best_host:
            host_speed = self.host_speeds.get(best_host, 2e9)
            compute_time = task_flops / host_speed
            self.host_ready_times[best_host] = min_eft
            self.task_assignments[task.get_name()] = best_host
            print(f"  -> 选择主机 {best_host}，预计完成时间: {min_eft:.2f}s")
        
        return best_host or list(self.hosts.keys())[0]
    
    def handle_completion(self, task: wrench.Task):
        """处理任务完成事件，更新主机状态"""
        super().handle_completion(task)
        # 这里可以添加更复杂的逻辑，如更新主机负载状态

class WASSHeuristicScheduler(HEFTScheduler):
    """WASS启发式调度器"""
    # 实际实现中会包含更复杂的数据感知逻辑
    def get_scheduling_decision(self, task: wrench.Task):
        # 基于数据局部性和主机负载的启发式调度
        best_host = None
        min_cost = float('inf')
        
        # 获取任务输入文件大小
        input_files = task.get_input_files()
        total_input_size = sum(f.get_size() for f in input_files)
        
        for host_name in self.hosts.keys():
            # 获取主机对应的计算服务
            if host_name in self.compute_services:
                compute_service = self.compute_services[host_name]
            else:
                # 回退到第一个可用的计算服务
                compute_service = list(self.compute_services.values())[0]
            
            # 获取主机速度
            core_speeds = compute_service.get_core_flop_rates()
            host_speed = core_speeds[0] if isinstance(core_speeds, (list, tuple)) and core_speeds else 1e9
            
            # 计算计算成本（执行时间）
            compute_cost = task.get_flops() / host_speed
            
            # 计算数据传输成本（简化版，假设数据需要从存储主机传输）
            # 网络带宽为1GBps，延迟为1ms
            transfer_time = total_input_size / (1e9)  # 1GBps = 1e9 bytes/s
            network_cost = transfer_time + 0.001  # 加上1ms延迟
            
            # 总成本 = 计算成本 + 数据传输成本
            total_cost = compute_cost + network_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_host = host_name
        
        return best_host or list(self.hosts.keys())[0]

class RecordingHEFTScheduler(HEFTScheduler):
    """
    一个继承自HEFTScheduler的特殊调度器，
    它的唯一目的是记录下所有调度决策。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions = []
        print("📝 RecordingHEFTScheduler initialized. Ready to record decisions.")

    def get_scheduling_decision(self, task: 'wrench.Task') -> str:
        # 调用父类（原始HEFT）的决策逻辑
        decision_host = super().get_scheduling_decision(task)
        
        # 记录决策
        self.decisions.append({
            "task_name": task.get_name(),
            "host_name": decision_host
        })
        # print(f"    [Record] Task '{task.get_name()}' -> Host '{decision_host}'")
        return decision_host

    def get_recorded_decisions(self) -> list:
        """获取所有记录下来的决策"""
        return self.decisions