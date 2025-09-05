"""RAG 融合占位."""
from __future__ import annotations
from typing import Dict, Any, List

class ConcatFusion:
    def fuse(self, sample: Dict[str, Any], retrieved: List[Dict[str, Any]]):
        ctxs = [r.get('text', '') for r in retrieved]
        sample = dict(sample)
        sample['retrieved_context'] = '\n'.join(ctxs)
        return sample
