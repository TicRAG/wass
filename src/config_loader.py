"""多文件配置加载与合并工具.
- 支持 experiment.yaml 中 include 列表
- 合并策略: 浅层覆盖 (后加载键覆盖先前), 除非是 list 则拼接 (去重可选)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Union
import yaml

MERGE_LIST_DEDUP = True


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge(out[k], v)
        elif k in out and isinstance(out[k], list) and isinstance(v, list):
            if MERGE_LIST_DEDUP:
                existing = set(out[k])
                out[k] = out[k] + [x for x in v if x not in existing]
            else:
                out[k] = out[k] + v
        else:
            out[k] = v
    return out


def load_experiment_config(base_dir: str, filename: str = 'experiment.yaml') -> Dict[str, Any]:
    base_path = Path(base_dir)
    main_cfg = load_yaml(base_path / filename)
    includes: List[str] = main_cfg.get('include', [])
    merged: Dict[str, Any] = {}
    for inc in includes:
        part = load_yaml(base_path / inc)
        merged = merge(merged, part)
    merged = merge(merged, main_cfg)
    return merged

if __name__ == '__main__':
    cfg = load_experiment_config('configs')
    print(cfg)

# Deprecated config_loader removed 2025-09-11.
