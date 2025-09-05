"""Graph 构建占位 (基于共现)."""
from __future__ import annotations
from typing import List, Dict, Any
from collections import defaultdict

class CooccurrenceGraphBuilder:
    def __init__(self, window_size: int = 5, field: str = 'text'):
        self.window_size = window_size
        self.field = field

    def build(self, data: List[Dict[str, Any]], labels):
        # 非真实图结构: 返回简单字典 {node: {neighbor: weight}}
        graph = defaultdict(lambda: defaultdict(int))
        for sample in data:
            text = sample.get(self.field, '')
            tokens = text.split()
            for i, tok in enumerate(tokens):
                for j in range(i+1, min(i + 1 + self.window_size, len(tokens))):
                    other = tokens[j]
                    if other == tok:
                        continue
                    graph[tok][other] += 1
                    graph[other][tok] += 1
        return graph
