"""Label Function 基础与注册机制 (占位)."""
from __future__ import annotations
from typing import Callable, Dict, List, Any

REGISTRY: Dict[str, Callable] = {}
ABSTAIN = -1

class LFWrapper:
    def __init__(self, name: str, func: Callable, abstain: int = ABSTAIN):
        self.name = name
        self.func = func
        self.abstain = abstain
    def __call__(self, x: Any) -> int:
        try:
            return self.func(x)
        except Exception:
            return self.abstain

def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco

@register("keyword")
def lf_keyword_builder(keywords: List[str], label: int, field: str = "text", lower: bool = True, abstain: int = ABSTAIN):
    kws = [k.lower() if lower else k for k in keywords]
    def lf(sample):
        text = sample.get(field, '')
        t = text.lower() if lower else text
        for k in kws:
            if k in t:
                return label
        return abstain
    return lf

def build_lfs(config_list: List[dict]) -> List[LFWrapper]:
    lfs: List[LFWrapper] = []
    for cfg in config_list:
        t = cfg.get('type')
        name = cfg.get('name', t)
        builder = REGISTRY.get(t)
        if not builder:
            raise ValueError(f"Unknown LF type: {t}")
        params = {k: v for k, v in cfg.items() if k not in {'type', 'name'}}
        func = builder(**params)
        lfs.append(LFWrapper(name=name, func=func, abstain=cfg.get('abstain', ABSTAIN)))
    return lfs
