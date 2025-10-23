"""Utility helpers for loading project configuration files."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"

@lru_cache(maxsize=1)
def load_training_config() -> Dict[str, Any]:
    """Return the parsed training configuration as a dictionary."""
    if not TRAINING_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Training config not found: {TRAINING_CONFIG_PATH}")
    with TRAINING_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    return data
