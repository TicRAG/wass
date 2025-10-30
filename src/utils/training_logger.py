from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _slugify(value: str) -> str:
    cleaned = []
    for ch in value.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {" ", "-", ".", ":", "/"}:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return slug or "run"


class TrainingLogger:
    """Utility to persist per-episode training metrics for later analysis."""

    FIELDNAMES = [
        "timestamp",
        "episode",
        "strategy",
        "strategy_label",
        "run_label",
        "seed",
        "status",
        "workflow",
        "task_count",
        "makespan",
        "episode_reward",
        "avg_rag_reward",
        "min_rag_reward",
        "max_rag_reward",
        "std_rag_reward",
        "clamped_pct",
        "sim_time",
        "episode_wallclock",
        "reward_mode",
        "rag_enabled",
        "replay_steps",
        "extra_json",
    ]

    def __init__(
        self,
        strategy_label: str,
        *,
        output_dir: str | Path = "results/training_runs",
        seed: Optional[int] = None,
        run_label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.strategy_label = strategy_label
        self.strategy_slug = _slugify(strategy_label)
        self.seed = seed
        self.run_label = run_label or strategy_label
        self.run_slug = _slugify(self.run_label)
        self.metadata = metadata or {}

        self.log_dir = Path(output_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        seed_part = f"seed{self.seed}" if self.seed is not None else "seedNA"
        filename = f"{self.run_slug}_{seed_part}_{timestamp}.csv"
        self.file_path = self.log_dir / filename
        self.file = self.file_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()
        self.file.flush()

        meta_payload = {
            "strategy_label": self.strategy_label,
            "strategy_slug": self.strategy_slug,
            "run_label": self.run_label,
            "run_slug": self.run_slug,
            "seed": self.seed,
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
            "log_file": self.file_path.name,
            "metadata": self.metadata,
        }
        meta_path = self.file_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def log_episode(self, episode: int, workflow: Optional[str], metrics: Optional[Dict[str, Any]] = None) -> None:
        extras: Dict[str, Any] = {}
        row: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "episode": episode,
            "strategy": self.strategy_slug,
            "strategy_label": self.strategy_label,
            "run_label": self.run_label,
            "seed": self.seed,
            "workflow": Path(workflow).name if workflow else None,
        }
        if metrics:
            for key, value in metrics.items():
                if key in self.FIELDNAMES:
                    row[key] = value
                else:
                    extras[key] = value
        if extras:
            row["extra_json"] = json.dumps(extras, separators=(",", ":"), ensure_ascii=True)
        for key in self.FIELDNAMES:
            if key not in row:
                row[key] = ""
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover
        return f"TrainingLogger(file_path={self.file_path!s})"
