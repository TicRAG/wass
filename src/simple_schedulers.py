import random
from typing import List, Dict
import wrench.task as w
from src.interfaces import Scheduler, SchedulingDecision
from src.wrench_schedulers import BaseScheduler


class SimpleRandomScheduler(Scheduler):
    """简单的随机调度器"""
    
    def __init__(self, name: str):
        self.name = name
        self.node_names = ['node1', 'node2', 'node3']  # 默认节点名
    
    def schedule(self, ready_tasks: List[w.Task], simulation) -> SchedulingDecision:
        if not ready_tasks:
            return {}
        
        # 调度所有就绪任务，而不是只调度第一个
        scheduling_decisions = {}
        for task in ready_tasks:
            chosen_node = random.choice(self.node_names)
            scheduling_decisions[chosen_node] = task
        
        return scheduling_decisions


class HeuristicScheduler(Scheduler):
    """启发式调度器"""
    
    def __init__(self, name: str):
        self.name = name
        self.node_names = ['node1', 'node2', 'node3']  # 默认节点名
    
    def schedule(self, ready_tasks: List[w.Task], simulation) -> SchedulingDecision:
        if not ready_tasks:
            return {}
        
        # 调度所有就绪任务，使用轮询方式分配节点
        scheduling_decisions = {}
        for i, task in enumerate(ready_tasks):
            chosen_node = self.node_names[i % len(self.node_names)]
            scheduling_decisions[chosen_node] = task
        
        return scheduling_decisions


class ImprovedHeuristicScheduler(Scheduler):
    """改进的启发式调度器"""
    
    def __init__(self, name: str):
        self.name = name
        self.node_names = ['node1', 'node2', 'node3']  # 默认节点名
    
    def schedule(self, ready_tasks: List[w.Task], simulation) -> SchedulingDecision:
        if not ready_tasks:
            return {}
        
        # 调度所有就绪任务，基于任务计算量分配节点
        scheduling_decisions = {}
        for task in ready_tasks:
            # 基于任务计算量选择节点（计算量大的任务分配给更快的节点）
            flops = task.get_flops() if hasattr(task, 'get_flops') else 1000
            
            if flops > 5000:
                chosen_node = self.node_names[0]  # 假设第一个节点最快
            elif flops > 2000:
                chosen_node = self.node_names[1]
            else:
                chosen_node = self.node_names[-1]  # 最后一个节点处理轻量任务
            
            scheduling_decisions[chosen_node] = task
        
        return scheduling_decisions