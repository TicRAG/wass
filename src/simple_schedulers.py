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
        
        task_to_schedule = ready_tasks[0]
        chosen_node = random.choice(self.node_names)
        return {chosen_node: task_to_schedule}


class HeuristicScheduler(Scheduler):
    """启发式调度器"""
    
    def __init__(self, name: str):
        self.name = name
        self.node_names = ['node1', 'node2', 'node3']  # 默认节点名
    
    def schedule(self, ready_tasks: List[w.Task], simulation) -> SchedulingDecision:
        if not ready_tasks:
            return {}
        
        task_to_schedule = ready_tasks[0]
        # 简单的启发式：选择第一个节点
        chosen_node = self.node_names[0]
        return {chosen_node: task_to_schedule}


class ImprovedHeuristicScheduler(Scheduler):
    """改进的启发式调度器"""
    
    def __init__(self, name: str):
        self.name = name
        self.node_names = ['node1', 'node2', 'node3']  # 默认节点名
    
    def schedule(self, ready_tasks: List[w.Task], simulation) -> SchedulingDecision:
        if not ready_tasks:
            return {}
        
        task_to_schedule = ready_tasks[0]
        # 改进的启发式：随机选择节点
        chosen_node = random.choice(self.node_names)
        return {chosen_node: task_to_schedule}