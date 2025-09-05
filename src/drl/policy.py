"""DRL 策略占位."""
from __future__ import annotations
import random

class RandomPolicy:
    def act(self, state):
        return random.randint(0, 10)
    def update(self, *batch):
        pass
