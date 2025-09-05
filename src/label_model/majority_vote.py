"""Majority Vote Label Model (占位)."""
from __future__ import annotations
import numpy as np
from typing import Any
from ..interfaces import LabelModel
from ..labeling.lf_base import ABSTAIN

class MajorityVote(LabelModel):
    def __init__(self, abstain_val: int = ABSTAIN):
        self.abstain_val = abstain_val

    def fit(self, L, **kwargs) -> None:
        # No training needed for majority vote
        pass

    def predict_proba(self, L) -> Any:
        # Assume binary 0/1 labels ignoring abstain
        n, m = L.shape
        probs = np.zeros((n, 2), dtype=float)
        for i in range(n):
            row = L[i]
            valid = row[row != self.abstain_val]
            if len(valid) == 0:
                probs[i] = [0.5, 0.5]
            else:
                p1 = (valid == 1).mean()
                probs[i] = [1 - p1, p1]
        return probs
