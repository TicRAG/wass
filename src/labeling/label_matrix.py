"""构建标签矩阵 (占位)."""
from __future__ import annotations
from typing import List
import numpy as np

from .lf_base import LFWrapper, ABSTAIN

class SimpleLabelMatrixBuilder:
    def build(self, data: List[dict], lfs: List[LFWrapper]):
        n = len(data)
        m = len(lfs)
        L = np.full((n, m), ABSTAIN, dtype=int)
        for i, sample in enumerate(data):
            for j, lf in enumerate(lfs):
                L[i, j] = lf(sample)
        return L

    @staticmethod
    def stats(L, abstain_val=ABSTAIN):
        """计算标签矩阵统计信息."""
        from ..utils import calculate_conflict_rate
        
        cover = (L != abstain_val).mean()
        abstain_rate = 1 - cover
        conflict_rate = calculate_conflict_rate(L)
        
        # 计算每个LF的覆盖率
        lf_coverage = []
        for j in range(L.shape[1]):
            lf_cover = (L[:, j] != abstain_val).mean()
            lf_coverage.append(lf_cover)
        
        return {
            "coverage": cover, 
            "abstain_rate": abstain_rate,
            "conflict_rate": conflict_rate,
            "lf_coverage": lf_coverage,
            "n_samples": L.shape[0],
            "n_lfs": L.shape[1]
        }
