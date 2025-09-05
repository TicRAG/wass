"""简单 BM25 风格检索占位 (未实现真实BM25)."""
from __future__ import annotations
from typing import List, Dict, Any
from collections import Counter
import math

class SimpleBM25Retriever:
    def __init__(self, field: str = 'text'):
        self.field = field
        self.docs: List[Dict[str, Any]] = []
        self.df = Counter()
        self.avg_len = 0.0

    def index(self, data: List[Dict[str, Any]]):
        self.docs = data
        total_len = 0
        for d in data:
            tokens = set(d.get(self.field, '').split())
            total_len += len(tokens)
            for t in tokens:
                self.df[t] += 1
        self.avg_len = total_len / max(1, len(data))

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = query.split()
        N = len(self.docs)
        scores = []
        for idx, d in enumerate(self.docs):
            text_tokens = d.get(self.field, '').split()
            tf = Counter(text_tokens)
            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                df = self.df.get(qt, 0) + 1
                idf = math.log((N - df + 0.5)/(df + 0.5) + 1)
                score += (tf[qt]) * idf
            scores.append((score, idx))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:top_k]]
