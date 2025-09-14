from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

@dataclass
class KnowledgeCase:
    workflow_id: str
    task_id: str
    scheduler_type: str
    chosen_node: str
    task_execution_time: float
    workflow_makespan: float
    task_features: List[float] | None = None
    workflow_embedding: List[float] | None = None
    quality_score: float | None = None  # optional quality metric (e.g., inverse makespan)

class JSONKnowledgeBase:
    def __init__(self):
        self.cases: List[KnowledgeCase] = []
        self.case_index: Dict[str, List[int]] = {}
        self.embedding_dim: int | None = None

    def add_case(self, case: KnowledgeCase):
        idx = len(self.cases)
        self.cases.append(case)
        key = f"{case.workflow_id}:{case.task_id}"
        self.case_index.setdefault(key, []).append(idx)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "total_cases": len(self.cases),
            "embedding_dim": self.embedding_dim,
            "cases": [asdict(c) for c in self.cases]
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "JSONKnowledgeBase":
        kb = cls()
        kb.embedding_dim = data.get("embedding_dim")
        for raw in data.get("cases", []):
            case = KnowledgeCase(**raw)
            kb.add_case(case)
        return kb

    def save_json(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "JSONKnowledgeBase":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json_dict(data)

    def quality_filter(self, top_p: float = 0.5):
        scored = [c for c in self.cases if c.quality_score is not None]
        if not scored:
            return
        scored.sort(key=lambda c: c.quality_score, reverse=True)
        cutoff = int(len(scored) * top_p)
        keep_set = set(scored[:cutoff])
        # rebuild
        new_cases = []
        for c in self.cases:
            if (c in keep_set) or (c.quality_score is None):
                new_cases.append(c)
        self.cases = []
        self.case_index.clear()
        for c in new_cases:
            self.add_case(c)

    def compute_quality_scores(self, baseline_makespan: float | None = None):
        """Assign a quality_score to each case.

        Heuristic: higher score for lower workflow_makespan relative to baseline.
        If baseline not provided, use 90th percentile of observed makespans.
        """
        makespans = [c.workflow_makespan for c in self.cases if c.workflow_makespan]
        if not makespans:
            return
        if baseline_makespan is None:
            sorted_m = sorted(makespans)
            idx = int(0.9 * (len(sorted_m)-1))
            baseline_makespan = sorted_m[idx]
        for c in self.cases:
            if c.workflow_makespan and c.workflow_makespan > 0:
                ratio = baseline_makespan / c.workflow_makespan
                c.quality_score = float(ratio)
            else:
                c.quality_score = None

    def top_k(self, k: int) -> list[KnowledgeCase]:
        if k <= 0:
            return []
        scored = [c for c in self.cases if c.quality_score is not None]
        if not scored:
            return self.cases[:k]
        scored.sort(key=lambda c: c.quality_score, reverse=True)
        return scored[:k]

__all__ = ["KnowledgeCase", "JSONKnowledgeBase"]
