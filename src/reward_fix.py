import numpy as np
import math
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class RewardFix:
    """修复后的奖励计算类"""
    
    def __init__(self):
        self.reward_history = []
        self.makespan_history = []
        
    def calculate_normalized_reward(self, teacher_makespan, student_makespan, task_scale):
        """
        计算归一化的奖励信号
        
        Args:
            teacher_makespan: 老师（预测器）建议的makespan
            student_makespan: 学生（Agent）选择的makespan
            task_scale: 任务规模，用于归一化
        
        Returns:
            归一化后的奖励值
        """
        # 1. 确保时间单位合理（转换为秒）
        teacher_makespan_sec = teacher_makespan / 1e9  # 假设原始是纳秒
        student_makespan_sec = student_makespan / 1e9
        
        # 2. 检查makespan值是否异常大，如果是则进行对数变换
        if teacher_makespan_sec > 1e6:
            teacher_makespan_sec = math.log(teacher_makespan_sec)
        
        if student_makespan_sec > 1e6:
            student_makespan_sec = math.log(student_makespan_sec)
        
        # 3. 确保makespan值不为零或负数
        teacher_makespan_sec = max(teacher_makespan_sec, 1e-8)
        student_makespan_sec = max(student_makespan_sec, 1e-8)
        
        # 4. 计算相对改进
        relative_improvement = (teacher_makespan_sec - student_makespan_sec) / teacher_makespan_sec
        
        # 5. 使用tanh函数将奖励限制在[-1, 1]范围内
        normalized_reward = math.tanh(relative_improvement * 2.0)  # 减小缩放因子
        
        # 6. 记录历史
        self.reward_history.append(normalized_reward)
        self.makespan_history.append(student_makespan_sec)
        
        return normalized_reward
    
    def calculate_multi_objective_reward(self, simulation, task, chosen_node, teacher_makespan, student_makespan):
        """
        计算多目标奖励，考虑多个性能指标
        
        Args:
            simulation: 仿真环境
            task: 当前调度的任务
            chosen_node: 选择的计算节点
            teacher_makespan: 老师（预测器）建议的makespan
            student_makespan: 学生（Agent）选择的makespan
        
        Returns:
            综合奖励值
        """
        # 1. 时间效率奖励（归一化的makespan改进）
        task_scale = max(task.computation_size, 1.0)
        time_reward = self.calculate_normalized_reward(
            teacher_makespan, student_makespan, task_scale
        )
        
        # 2. 资源利用率奖励
        node_utilization = self._get_node_utilization(simulation, chosen_node)
        utilization_reward = node_utilization * 0.2  # 权重0.2
        
        # 3. 负载均衡奖励
        load_std = self._get_load_std(simulation)
        balance_reward = -load_std * 0.3  # 权重0.3，负号表示希望负载均衡
        
        # 4. 任务完成率奖励
        completion_rate = len(simulation.completed_tasks) / len(simulation.workflow.tasks)
        completion_reward = completion_rate * 0.1  # 权重0.1
        
        # 5. 紧急任务奖励
        urgency = self._get_task_urgency(simulation, task)
        urgency_reward = urgency * 0.2  # 权重0.2
        
        # 6. 综合奖励
        total_reward = (
            time_reward * 0.5 +  # 时间效率权重0.5
            utilization_reward +
            balance_reward +
            completion_reward +
            urgency_reward
        )
        
        return total_reward
    
    def _get_node_utilization(self, simulation, node_name):
        """获取节点利用率"""
        # 简化实现，实际应根据仿真环境计算
        node = simulation.platform.get_node(node_name)
        if hasattr(node, 'utilization'):
            return node.utilization
        return 0.5  # 默认值
    
    def _get_load_std(self, simulation):
        """获取负载标准差"""
        # 简化实现，实际应根据仿真环境计算
        return 0.1  # 默认值
    
    def _get_task_urgency(self, simulation, task):
        """获取任务紧急程度"""
        # 简化实现，实际应根据任务deadline计算
        total_tasks = len(simulation.workflow.tasks)
        completed_tasks = len(simulation.completed_tasks)
        return completed_tasks / total_tasks  # 完成比例作为紧急程度
    
    def debug_reward_info(self, task_id, teacher_makespan, student_makespan, final_reward):
        """记录奖励调试信息"""
        # 转换为秒以便于阅读
        teacher_sec = teacher_makespan / 1e9
        student_sec = student_makespan / 1e9
        
        # 如果值很大，使用对数变换显示
        if teacher_sec > 1e6:
            teacher_display = f"log({teacher_sec:.2e})={math.log(teacher_sec):.2f}"
        else:
            teacher_display = f"{teacher_sec:.2f}s"
            
        if student_sec > 1e6:
            student_display = f"log({student_sec:.2e})={math.log(student_sec):.2f}"
        else:
            student_display = f"{student_sec:.2f}s"
        
        logger.info(f"Reward Debug - Task={task_id}, "
                   f"Teacher={teacher_display}, "
                   f"Student={student_display}, "
                   f"FinalReward={final_reward:.4f}")