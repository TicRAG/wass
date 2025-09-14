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

@dataclass
class EpisodeStats:
    makespan: float
    rolling_mean_makespan: float | None = None
    rolling_std_makespan: float | None = None

WEIGHTS = {
    'cpp': 0.35,
    'lb': 0.20,
    'pu': 0.25,
    'qd': 0.20,
}

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0, None) else default

def compute_step_reward(ctx: StepContext, temperature: float = 0.75, debug_writer: Optional[TextIO] = None) -> Tuple[float, Dict[str, float]]:
    # Critical Path Progress
    cpp_ratio = _safe_div(ctx.completed_critical_path_tasks, ctx.total_critical_path_tasks, 0.0)
    cpp = max(0.0, min(1.0, cpp_ratio))

    # Load Balance: 1 - Gini coefficient
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

    # Parallelism Utilization
    pu_raw = _safe_div(ctx.ready_task_count, ctx.total_nodes, 0.0)
    pu = max(0.0, min(1.0, pu_raw))

    # Queue Delay Penalty (negative)
    if ctx.queue_wait_baseline and ctx.queue_wait_baseline > 0:
        qd_norm = ctx.avg_queue_wait / ctx.queue_wait_baseline
    else:
        qd_norm = 0.0
    qd = -max(0.0, min(1.0, qd_norm))

    # Weighted sum
    raw = (WEIGHTS['cpp'] * cpp + WEIGHTS['lb'] * lb + WEIGHTS['pu'] * pu + WEIGHTS['qd'] * qd)
    shaped = math.tanh(raw / max(temperature, 1e-6))

    metrics = {
        'cpp': cpp,
        'lb': lb,
        'pu': pu,
        'qd': qd,
        'raw_sum': raw,
        'shaped': shaped,
    }
    if debug_writer:
        try:
            debug_writer.write(
                f"cpp={cpp:.4f}\tlb={lb:.4f}\tpu={pu:.4f}\tqd={qd:.4f}\traw={raw:.5f}\tshaped={shaped:.5f}\n"
            )
        except Exception:
            pass
    return shaped, metrics

def compute_final_reward(stats: EpisodeStats) -> float:
    if stats.rolling_mean_makespan is not None and stats.rolling_std_makespan and stats.rolling_std_makespan > 0:
        norm = (stats.makespan - stats.rolling_mean_makespan) / stats.rolling_std_makespan
    else:
        norm = stats.makespan
    return -norm

__all__ = [
    'StepContext', 'EpisodeStats', 'compute_step_reward', 'compute_final_reward'
]
