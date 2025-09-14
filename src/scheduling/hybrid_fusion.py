from __future__ import annotations
"""Hybrid decision fusion (R-2.3 skeleton).

Combines DRL Q-values, RAG similarity/confidence scores, and load-based penalty.
Actual data providers (DRL agent, RAG retriever, environment) will inject the
arrays required here.
"""
from typing import Dict, List, Any
import math

class FusionError(Exception):
    pass

def fuse_decision(q_values: List[float], rag_scores: List[float], load_values: List[float], progress: float,
                  rag_confidence_threshold: float = 0.05, makespan_predictions: List[float] = None,
                  baseline_makespan: float = None) -> Dict[str, Any]:
    """
    融合DRL、RAG和负载均衡决策，加入makespan预测权重
    
    Args:
        q_values: DRL Q值列表
        rag_scores: RAG相似度分数列表
        load_values: 负载值列表
        progress: 训练进度
        rag_confidence_threshold: RAG置信度阈值
        makespan_predictions: 各节点的makespan预测列表
        baseline_makespan: 基准makespan（如HEFT算法的结果）
        
    Returns:
        融合决策结果
    """
    if not (len(q_values) == len(rag_scores) == len(load_values)):
        raise FusionError("Input length mismatch")
    n = len(q_values)
    if n == 0:
        raise FusionError("Empty inputs")

    # Normalize Q
    q_mean = sum(q_values) / n
    q_var = sum((x - q_mean) ** 2 for x in q_values) / max(n-1, 1)
    q_std = math.sqrt(q_var) if q_var > 0 else 1.0
    q_norm = [(x - q_mean) / q_std for x in q_values]

    # Normalize RAG scores
    max_rag = max(rag_scores) if rag_scores else 1.0
    rag_norm = [s / max_rag if max_rag > 0 else 0.0 for s in rag_scores]

    # Load transformed to positive preference (lower load -> higher score)
    # 增强负载均衡权重：使用更强的指数变换，使高负载节点得分更低
    max_load = max(load_values) if load_values else 1.0
    load_norm_raw = [lv / max_load if max_load > 0 else 0.0 for lv in load_values]
    
    # 使用较缓和的指数变换，保留负载影响但不过分主导
    load_pref = [math.exp(-4.0 * lv) for lv in load_norm_raw]  # 增强系数从16.0缓和到4.0

    # 计算makespan预测得分（如果提供）
    makespan_scores = [0.0] * n
    if makespan_predictions is not None and baseline_makespan is not None and baseline_makespan > 0:
        for i, pred in enumerate(makespan_predictions):
            # makespan改进程度（越小越好）
            improvement = (baseline_makespan - pred) / baseline_makespan
            # 使用sigmoid函数将改进程度转换为得分
            makespan_scores[i] = 1.0 / (1.0 + math.exp(-improvement * 5.0))  # 乘以5.0使函数更陡峭

    progress = max(0.0, min(1.0, progress))
    
    # 重新平衡权重，给予RAG和DRL更多话语权
    alpha = 0.25 - 0.1 * progress  # DRL权重
    beta = 0.35 + 0.1 * progress   # RAG权重，随训练进度增加
    gamma = 0.20  # 负载均衡权重
    delta = 0.20  # makespan预测权重
    
    # 确保权重总和为1
    total_weight = alpha + beta + gamma + delta
    if total_weight > 0:
        alpha /= total_weight
        beta /= total_weight
        gamma /= total_weight
        delta /= total_weight

    # Confidence gating
    if max(rag_norm) < rag_confidence_threshold:
        beta = 0.0
        # 重新分配权重
        total_weight = alpha + gamma + delta
        if total_weight > 0:
            alpha /= total_weight
            gamma /= total_weight
            delta /= total_weight

    fused = []
    for i in range(n):
        fused.append(alpha * q_norm[i] + beta * rag_norm[i] + gamma * load_pref[i] + delta * makespan_scores[i])

    # Select best
    best_idx = max(range(n), key=lambda i: fused[i])

    return {
        'index': best_idx,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'delta': delta,
        'q_norm': q_norm,
        'rag_norm': rag_norm,
        'load_pref': load_pref,
        'makespan_scores': makespan_scores,
        'fused': fused
    }

__all__ = ['fuse_decision', 'FusionError']
