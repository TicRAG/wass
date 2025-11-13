"""Utilities for normalising workflow family identifiers.

The knowledge base and inference pipeline often operate on filenames that
include augmentation suffixes (e.g. `_aug3_aug7`).  To make it easier to group
related samples we derive a family label from the workflow metadata or the file
name.  The helper intentionally stays lightweight so it can be used from both
scripts and runtime code without introducing heavy dependencies.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

_KNOWN_WORKFLOW_TOKENS: dict[str, str] = {
    "montage": "montage",
    "epigenomics": "epigenomics",
    "genome": "epigenomics",
    "seismology": "seismology",
    "cybershake": "cybershake",
    "sipht": "sipht",
    "inspiral": "inspiral",
    "synthetic": "synthetic",
    "pipeline": "pipeline",
}

_SUFFIX_PATTERN = re.compile(r"_aug\d+", re.IGNORECASE)
_SPLIT_PATTERN = re.compile(r"[^a-z0-9]+")


def _tokenise(raw: str) -> Iterable[str]:
    for token in _SPLIT_PATTERN.split(raw.lower()):
        token = token.strip()
        if not token:
            continue
        if token.startswith("aug") and token[3:].isdigit():
            continue
        yield token


def _load_name_field(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        name = data.get("name")
        if isinstance(name, str) and name.strip():
            return name
    except Exception:
        pass
    return ""


@lru_cache(maxsize=None)
def infer_workflow_family(path_like: str | Path) -> str:
    """Return a best-effort workflow family string for the given file.

    The function prefers explicit metadata (`workflow.name`).  If that is
    missing it falls back to the filename.  Known WFCommons applications are
    mapped to stable canonical labels, while other cases simply return the
    first meaningful token.
    """
    path = Path(path_like)
    candidates = []
    name_field = _load_name_field(path)
    if name_field:
        candidates.extend(_tokenise(name_field))
    stem = _SUFFIX_PATTERN.sub("", path.stem)
    candidates.extend(_tokenise(stem))
    for token in candidates:
        canonical = _KNOWN_WORKFLOW_TOKENS.get(token)
        if canonical:
            return canonical
    return candidates[0] if candidates else "unknown"


__all__ = ["infer_workflow_family"]
