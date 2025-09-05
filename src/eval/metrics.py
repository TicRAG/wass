"""评估指标占位."""
from __future__ import annotations
from typing import List

def accuracy(preds: List[int], gold: List[int]) -> float:
    if not preds:
        return 0.0
    correct = sum(p==g for p,g in zip(preds, gold))
    return correct / len(preds)

def f1_binary(preds: List[int], gold: List[int]) -> float:
    tp = sum((p==1 and g==1) for p,g in zip(preds, gold))
    fp = sum((p==1 and g==0) for p,g in zip(preds, gold))
    fn = sum((p==0 and g==1) for p,g in zip(preds, gold))
    if tp==0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2*precision*recall/(precision+recall)
