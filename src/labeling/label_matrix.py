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
        cover = (L != abstain_val).mean()
        abstain_rate = 1 - cover
        return {"coverage": cover, "abstain_rate": abstain_rate}
