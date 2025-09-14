from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class WrenchKnowledgeCaseMinimal:
    workflow_id: str
    task_id: str
    scheduler_type: str
    chosen_node: str
    task_flops: float
    task_execution_time: float
    workflow_makespan: float | None = None

    @classmethod
    def from_legacy(cls, obj: Any, idx: int = 0) -> 'WrenchKnowledgeCaseMinimal':
        if hasattr(obj, '__dict__') and not isinstance(obj, dict):
            d = obj.__dict__
        elif isinstance(obj, dict):
            d = obj
        else:
            d = {}
        return cls(
            workflow_id=str(d.get('workflow_id', f'wf_{idx}')),
            task_id=str(d.get('task_id', f'task_{idx}')),
            scheduler_type=str(d.get('scheduler_type', 'unknown')),
            chosen_node=str(d.get('chosen_node', 'ComputeHost1')),
            task_flops=float(d.get('task_flops', 1e9)),
            task_execution_time=float(d.get('task_execution_time', 0.0)),
            workflow_makespan=float(d.get('workflow_makespan', d.get('makespan', 0.0)))
        )

__all__ = ["WrenchKnowledgeCaseMinimal"]
