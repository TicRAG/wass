#!/usr/bin/env python3
"""
WASS WRENCH仿真器
简化版本，用于生成基础实验数据
"""

import numpy as np
import time
from typing import Dict, Any

class WassWRENCHSimulator:
    """WASS WRENCH仿真器"""
    
    def __init__(self):
        self.simulation_overhead = 0.001  # 1ms仿真开销
        
    def run_simulation(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """运行工作流仿真"""
        
        # 添加基于工作流的随机性
        workflow_seed = hash(workflow.get('name', 'default')) % 2**32
        np.random.seed(workflow_seed)
        
        # 模拟仿真时间
        time.sleep(self.simulation_overhead)
        
        # 获取工作流基本信息
        tasks = workflow.get('tasks', [])
        task_count = len(tasks)
        
        if task_count == 0:
            return self._default_result()
        
        # 计算总工作量
        total_flops = sum(task.get('flops', 1e9) for task in tasks)
        total_memory = sum(task.get('memory', 1e9) for task in tasks)
        
        # 基础执行时间（基于工作量）
        base_execution_time = total_flops / 1e9  # 假设1 GFlops/s的基础处理能力
        
        # 添加网络和I/O延迟
        network_delay = task_count * np.random.uniform(0.1, 0.5)  # 每个任务的网络延迟
        io_delay = total_memory / 1e8 * np.random.uniform(0.8, 1.2)  # I/O延迟
        
        # 计算总执行时间
        execution_time = base_execution_time + network_delay + io_delay
        
        # 添加系统变异性（模拟真实环境的不确定性）
        system_variation = np.random.normal(1.0, 0.15)  # ±15%的系统变异
        execution_time *= abs(system_variation)
        
        # 计算其他指标
        throughput = task_count / execution_time if execution_time > 0 else 0
        memory_usage = min(total_memory / 1e9, 2.0)  # 标准化内存使用
        
        return {
            'execution_time': execution_time,
            'makespan': execution_time,
            'throughput': throughput,
            'memory_usage': memory_usage,
            'task_count': task_count,
            'total_flops': total_flops,
            'network_delay': network_delay,
            'io_delay': io_delay
        }
    
    def _default_result(self) -> Dict[str, Any]:
        """默认结果（空工作流）"""
        return {
            'execution_time': 1.0,
            'makespan': 1.0,
            'throughput': 0.0,
            'memory_usage': 0.1,
            'task_count': 0,
            'total_flops': 0,
            'network_delay': 0,
            'io_delay': 0
        }
