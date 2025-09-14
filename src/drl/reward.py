from __future__ import annotations
"""Reward computation module (R-2.1 skeleton).

Provides step-level shaping rewards and final makespan-based sparse reward.
Implementation here is a placeholder; real integration requires environment
context supplying workflow structure, node stats, and timing metrics.
"""
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, TextIO
import math

@dataclass
class StepContext:
    completed_critical_path_tasks: int
    total_critical_path_tasks: int
    node_busy_times: Dict[str, float]
    ready_task_count: int
    total_nodes: int
    avg_queue_wait: float
    queue_wait_baseline: float
    # 新增字段
    node_utilization: Dict[str, float]  # 节点利用率
    avg_resource_utilization: float     # 平均资源利用率
    data_transfer_time: float           # 数据传输时间
    computation_time: float            # 计算时间
    data_locality_score: float         # 数据局部性分数

@dataclass
class EpisodeStats:
    makespan: float
    rolling_mean_makespan: float | None = None
    rolling_std_makespan: float | None = None

WEIGHTS = {
    'cpp': 0.20,      # 关键路径进度权重
    'lb': 0.15,       # 负载均衡权重
    'pu': 0.10,       # 并行利用率权重
    'qd': 0.10,       # 队列延迟惩罚权重
    'makespan': 0.25, # 总体makespan预测权重
    'resource': 0.10, # 资源利用率权重
    'data_locality': 0.10, # 数据局部性权重
}

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0, None) else default

def compute_step_reward(
    ctx: StepContext, 
    temperature: float = 0.75, 
    debug_writer: Optional[TextIO] = None,
    predicted_makespan: float = None,
    baseline_makespan: float = None
) -> Tuple[float, Dict[str, float]]:
    """
    计算每步的奖励，改进版：平衡多种因素
    
    Args:
        ctx: 步骤上下文
        temperature: 温度参数，用于奖励缩放
        debug_writer: 调试信息写入器
        predicted_makespan: 预测的总体makespan
        baseline_makespan: 基准makespan（如HEFT算法的结果）
    
    Returns:
        奖励值和指标字典
    """
    # 1. Critical Path Progress (关键路径进度)
    cpp_ratio = _safe_div(ctx.completed_critical_path_tasks, ctx.total_critical_path_tasks, 0.0)
    cpp = max(0.0, min(1.0, cpp_ratio))

    # 2. Load Balance (负载均衡)
    loads = list(ctx.node_busy_times.values()) or [0.0]
    mean_load = sum(loads) / len(loads)
    if mean_load <= 0:
        lb = 0.0
    else:
        abs_diff_sum = 0.0
        for i in loads:
            for j in loads:
                abs_diff_sum += abs(i - j)
        gini = abs_diff_sum / (2 * len(loads) * len(loads) * mean_load)
        lb = 1.0 - max(0.0, min(1.0, gini))

    # 3. Parallelism Utilization (并行利用率)
    pu_raw = _safe_div(ctx.ready_task_count, ctx.total_nodes, 0.0)
    pu = max(0.0, min(1.0, pu_raw))

    # 4. Queue Delay Penalty (队列延迟惩罚)
    if ctx.queue_wait_baseline and ctx.queue_wait_baseline > 0:
        qd_norm = ctx.avg_queue_wait / ctx.queue_wait_baseline
    else:
        qd_norm = 0.0
    qd = -max(0.0, min(1.0, qd_norm))
    
    # 5. Makespan相关奖励
    makespan_reward = 0.0
    if predicted_makespan is not None and baseline_makespan is not None and baseline_makespan > 0:
        # 改进的makespan预测奖励计算
        # 考虑当前进度与预期进度的差异
        expected_progress = 1.0 - (predicted_makespan / baseline_makespan)
        actual_progress = cpp_ratio
        
        # 计算进度差异奖励
        progress_diff = actual_progress - expected_progress
        
        # 计算makespan改进奖励
        makespan_improvement = (baseline_makespan - predicted_makespan) / baseline_makespan
        
        # 结合进度差异和makespan改进
        makespan_reward = 0.6 * math.tanh(makespan_improvement * 3.0) + 0.4 * math.tanh(progress_diff * 2.0)
    
    # 6. Resource Utilization (资源利用率)
    resource_reward = 0.0
    if hasattr(ctx, 'avg_resource_utilization'):
        # 理想资源利用率在0.7-0.9之间，过高可能导致拥塞，过低则浪费资源
        if ctx.avg_resource_utilization < 0.7:
            # 利用率过低，线性增加奖励
            resource_reward = ctx.avg_resource_utilization / 0.7
        elif ctx.avg_resource_utilization <= 0.9:
            # 理想范围，给予最高奖励
            resource_reward = 1.0
        else:
            # 利用率过高，可能导致拥塞，奖励逐渐降低
            resource_reward = max(0.0, 1.0 - (ctx.avg_resource_utilization - 0.9) * 5.0)
    
    # 7. Data Locality (数据局部性)
    data_locality_reward = 0.0
    if hasattr(ctx, 'data_locality_score') and hasattr(ctx, 'data_transfer_time') and hasattr(ctx, 'computation_time'):
        # 计算数据传输时间占总时间的比例
        total_time = ctx.data_transfer_time + ctx.computation_time
        if total_time > 0:
            transfer_ratio = ctx.data_transfer_time / total_time
            
            # 数据局部性分数越高，传输比例越低，奖励越高
            # 使用指数函数使奖励对局部性更敏感
            data_locality_reward = ctx.data_locality_score * math.exp(-transfer_ratio * 2.0)
    
    # Weighted sum (加权求和)
    raw = (
        WEIGHTS['cpp'] * cpp + 
        WEIGHTS['lb'] * lb + 
        WEIGHTS['pu'] * pu + 
        WEIGHTS['qd'] * qd + 
        WEIGHTS['makespan'] * makespan_reward +
        WEIGHTS['resource'] * resource_reward +
        WEIGHTS['data_locality'] * data_locality_reward
    )
    
    # 使用sigmoid函数代替tanh，使奖励分布更平滑
    shaped = 2.0 / (1.0 + math.exp(-raw / max(temperature, 1e-6))) - 1.0  # 映射到[-1, 1]

    metrics = {
        'cpp': cpp,
        'lb': lb,
        'pu': pu,
        'qd': qd,
        'makespan_reward': makespan_reward,
        'resource_reward': resource_reward,
        'data_locality_reward': data_locality_reward,
        'raw_sum': raw,
        'shaped': shaped,
    }
    
    if debug_writer:
        try:
            debug_writer.write(
                f"cpp={cpp:.4f}\tlb={lb:.4f}\tpu={pu:.4f}\tqd={qd:.4f}\t"
                f"makespan={makespan_reward:.4f}\tresource={resource_reward:.4f}\t"
                f"data_locality={data_locality_reward:.4f}\traw={raw:.5f}\tshaped={shaped:.5f}\n"
            )
        except Exception:
            pass
    return shaped, metrics

def compute_final_reward(
    makespan: float, 
    stats: EpisodeStats, 
    temperature: float = 0.75,
    debug_writer: Optional[TextIO] = None,
    baseline_makespan: float = None
) -> Tuple[float, Dict[str, float]]:
    """
    计算最终奖励，改进版：更稳定的makespan归一化
    
    Args:
        makespan: 完成工作流的总时间
        stats: 回合统计信息
        temperature: 温度参数，用于奖励缩放
        debug_writer: 调试信息写入器
        baseline_makespan: 基准makespan（如HEFT算法的结果）
    
    Returns:
        奖励值和指标字典
    """
    # 确保makespan是合理的值，防止异常大的值
    if makespan <= 0:
        return 0.0, {'makespan': makespan, 'reward': 0.0}
    
    # 使用滚动统计信息进行归一化
    if stats.rolling_mean_makespan is not None and stats.rolling_std_makespan and stats.rolling_std_makespan > 0:
        # 使用对数变换处理大范围值，但添加偏移量避免负值
        log_makespan = math.log(makespan + 1.0)
        log_mean = math.log(stats.rolling_mean_makespan + 1.0)
        log_std = max(stats.rolling_std_makespan, 1e-6)  # 避免除以零
        
        # 计算Z分数
        z_score = (log_mean - log_makespan) / log_std
        
        # 使用tanh函数将Z分数映射到[-1, 1]范围
        # 限制Z分数的影响范围，避免极端值
        z_score_clipped = max(-3.0, min(3.0, z_score))
        reward = math.tanh(z_score_clipped)
    else:
        # 如果没有历史统计信息，使用基于基准的归一化
        if baseline_makespan is not None and baseline_makespan > 0:
            # 计算相对于基准的改进程度
            improvement = (baseline_makespan - makespan) / baseline_makespan
            # 使用tanh函数进行归一化，限制在[-1, 1]范围内
            reward = math.tanh(improvement * 2.0)
        else:
            # 如果没有基准，使用对数变换并限制范围
            log_makespan = math.log(max(makespan, 1.0))
            # 假设合理范围在1到1000之间，映射到[-1, 1]
            reward = 1.0 - 2.0 * min(1.0, max(0.0, (log_makespan - 0.0) / (math.log(1000.0) - 0.0)))
    
    # 应用温度缩放
    if temperature > 0:
        reward = reward / temperature
    
    # 确保奖励在合理范围内
    reward = max(-1.0, min(1.0, reward))
    
    metrics = {
        'makespan': makespan,
        'log_makespan': math.log(makespan + 1.0) if makespan > 0 else 0.0,
        'reward': reward,
        'baseline_makespan': baseline_makespan,
    }
    
    if stats.rolling_mean_makespan is not None:
        metrics['makespan_mean'] = stats.rolling_mean_makespan
        metrics['makespan_std'] = stats.rolling_std_makespan
        metrics['z_score'] = z_score if 'z_score' in locals() else 0.0
    
    if debug_writer:
        try:
            debug_writer.write(
                f"makespan={makespan:.2f}\tlog_makespan={metrics['log_makespan']:.4f}\t"
                f"reward={reward:.5f}\tbaseline={baseline_makespan if baseline_makespan else 'N/A'}\n"
            )
        except Exception:
            pass
    return reward, metrics

__all__ = [
    'StepContext', 'EpisodeStats', 'compute_step_reward', 'compute_final_reward'
]
