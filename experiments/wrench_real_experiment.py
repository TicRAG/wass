#!/usr/bin/env python3
"""DEPRECATED: wrench_real_experiment

All functionality relocated.
Current entry points:
  - scripts/train_wass_paper_aligned.py
  - scripts/evaluate_paper_methods.py

Historical version: legacy_archived/experiments/wrench_real_experiment.py

This stub exports nothing and raises if executed or imported directly
to prevent accidental reliance on removed legacy logic.
"""

__all__: list[str] = []

def _deprecated() -> None:  # pragma: no cover
    raise RuntimeError(
        "Deprecated module. Use new training/eval scripts or consult legacy_archived."
    )

if __name__ == "__main__":  # pragma: no cover
    _deprecated()
