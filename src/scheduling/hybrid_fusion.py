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
                  rag_confidence_threshold: float = 0.05) -> Dict[str, Any]:
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
    max_load = max(load_values) if load_values else 1.0
    load_norm_raw = [lv / max_load if max_load > 0 else 0.0 for lv in load_values]
    load_pref = [1.0 - lv for lv in load_norm_raw]

    progress = max(0.0, min(1.0, progress))
    alpha = 0.4 + 0.2 * progress
    beta = 0.4 - 0.15 * progress
    gamma = max(0.0, 1.0 - alpha - beta)

    # Confidence gating
    if max(rag_norm) < rag_confidence_threshold:
        beta = 0.0
        gamma = max(0.0, 1.0 - alpha)

    fused = []
    for i in range(n):
        fused.append(alpha * q_norm[i] + beta * rag_norm[i] + gamma * load_pref[i])

    # Select best
    best_idx = max(range(n), key=lambda i: fused[i])

    return {
        'index': best_idx,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'q_norm': q_norm,
        'rag_norm': rag_norm,
        'load_pref': load_pref,
        'fused': fused
    }

__all__ = ['fuse_decision', 'FusionError']
