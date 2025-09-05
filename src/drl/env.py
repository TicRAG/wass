"""DRL 环境占位 (Active Learning)."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import random

class ActiveLearningEnv:
    def __init__(self, unlabeled_pool: List[Dict[str, Any]]):
        self.pool = unlabeled_pool
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {"pool_size": len(self.pool)}

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        # action: index to sample (占位)
        self.step_count += 1
        done = self.step_count >= 5
        reward = random.random()
        next_state = {"pool_size": len(self.pool), "last_action": action}
        return next_state, reward, done, {}
