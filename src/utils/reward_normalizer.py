from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .workflow_family import infer_workflow_family


@dataclass(frozen=True)
class _FamilyStats:
    baseline_mean: float
    baseline_std: float
    ratio_low: float
    ratio_high: float
    ratio_std: float


class FamilyRewardNormalizer:
    """Compute workflow-family specific reward normalization factors.

    The normalizer derives baseline makespans from the knowledge-base metadata
    (heuristic traces) and uses them to:
      * convert raw makespans into baseline ratios (agent / heuristic);
      * clip those ratios with family-specific percentile ranges to contain
        outliers;
      * rescale dense per-step rewards into a similar magnitude across
        families; and
      * expose makespan-based curriculum bands for balanced sampling.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        clip_percentiles: Tuple[float, float] = (5.0, 95.0),
        min_ratio_low: float = 0.65,
        min_ratio_high: float = 1.15,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.clip_percentiles = clip_percentiles
        self.min_ratio_low = min_ratio_low
        self.min_ratio_high = min_ratio_high
        self.family_stats: Dict[str, _FamilyStats] = {}
        self.workflow_baselines: Dict[str, float] = {}
        self.workflow_families: Dict[str, str] = {}
        self.family_dense_scale: Dict[str, float] = {}
        self._load_reference_stats()

    def _load_reference_stats(self) -> None:
        if not self.metadata_path.exists():
            print(
                f"[RewardNorm] metadata file not found at {self.metadata_path}; "
                "reward normalization will fall back to naive ratios."
            )
            return
        try:
            df = pd.read_csv(
                self.metadata_path,
                usecols=["workflow_file", "scheduler_used", "makespan", "workflow_family"],
            )
        except Exception as exc:
            print(f"[RewardNorm] Failed to load metadata ({exc}); disabling normalization heuristics.")
            return
        df = df.dropna(subset=["workflow_file", "scheduler_used", "makespan"]).copy()
        if df.empty:
            return
        df["workflow_file"] = df["workflow_file"].astype(str)
        grouped = (
            df.groupby(["workflow_file", "scheduler_used"])
            .agg(makespan=("makespan", "mean"), workflow_family=("workflow_family", "first"))
            .reset_index()
        )
        baseline_candidates = grouped[grouped["scheduler_used"].isin(["HEFT", "MINMIN"])].copy()
        if baseline_candidates.empty:
            print("[RewardNorm] No HEFT/MINMIN entries in metadata; baseline ratios unavailable.")
            return
        baselines = (
            baseline_candidates.groupby("workflow_file")
            .agg(baseline=("makespan", "min"), workflow_family=("workflow_family", "first"))
            .reset_index()
        )
        random_runs = grouped[grouped["scheduler_used"] == "Random"]["workflow_file"].to_list()
        random_lookup = (
            grouped[grouped["scheduler_used"] == "Random"]
            .set_index("workflow_file")["makespan"].to_dict()
        )
        family_to_baselines: Dict[str, List[float]] = {}
        family_to_ratios: Dict[str, List[float]] = {}
        for _, row in baselines.iterrows():
            workflow_file = row["workflow_file"]
            baseline = float(row["baseline"])
            family = row.get("workflow_family")
            if not isinstance(family, str) or not family:
                family = infer_workflow_family(workflow_file)
            self.workflow_baselines[workflow_file] = baseline
            self.workflow_families[workflow_file] = family
            family_to_baselines.setdefault(family, []).append(baseline)
            random_makespan = random_lookup.get(workflow_file)
            if random_makespan and random_makespan > 0 and baseline > 0:
                ratio = float(random_makespan / baseline)
                family_to_ratios.setdefault(family, []).append(ratio)
        clip_low, clip_high = self.clip_percentiles
        for family, baseline_values in family_to_baselines.items():
            baseline_array = np.asarray(baseline_values, dtype=np.float64)
            ratio_samples = family_to_ratios.get(family, [])
            if not ratio_samples:
                # Inject neutral ratios so clipping still works when random data missing.
                ratio_samples = [1.0, 1.1]
            ratio_array = np.asarray(ratio_samples, dtype=np.float64)
            baseline_mean = float(np.mean(baseline_array))
            baseline_std = float(np.std(baseline_array))
            if baseline_std <= 0:
                baseline_std = max(baseline_mean * 0.1, 1.0)
            ratio_low = float(np.percentile(ratio_array, clip_low))
            ratio_high = float(np.percentile(ratio_array, clip_high))
            ratio_low = min(ratio_low, 0.98)
            ratio_low = max(ratio_low, self.min_ratio_low)
            ratio_high = max(ratio_high, self.min_ratio_high)
            ratio_std = float(np.std(ratio_array))
            if ratio_std <= 1e-6:
                ratio_std = max(0.2, (ratio_high - ratio_low) / 4.0)
            self.family_stats[family] = _FamilyStats(
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                ratio_low=ratio_low,
                ratio_high=ratio_high,
                ratio_std=ratio_std,
            )

    def get_family(self, workflow_file: str) -> str:
        workflow_file = Path(workflow_file).name
        return self.workflow_families.get(workflow_file) or infer_workflow_family(workflow_file)

    def _get_stats(self, workflow_file: str) -> Tuple[str, _FamilyStats | None]:
        family = self.get_family(workflow_file)
        return family, self.family_stats.get(family)

    def compute_final_reward(self, workflow_file: str, makespan: float) -> float:
        family, stats = self._get_stats(workflow_file)
        baseline = self.workflow_baselines.get(Path(workflow_file).name)
        if baseline is None or baseline <= 0:
            baseline = stats.baseline_mean if stats else makespan
        if baseline is None or baseline <= 0:
            return 0.0
        ratio = makespan / baseline
        if stats:
            ratio = max(stats.ratio_low, min(ratio, stats.ratio_high))
            centered = 1.0 - ratio
            scaled = centered / stats.ratio_std
            return float(scaled)
        return float(1.0 - ratio)

    def normalize_dense_rewards(
        self,
        workflow_file: str,
        raw_rewards: Iterable[float],
        multiplier: float = 1.0,
    ) -> List[float]:
        rewards = np.asarray(list(raw_rewards), dtype=np.float64)
        if rewards.size == 0:
            return []
        family, _stats = self._get_stats(workflow_file)
        abs_values = np.abs(rewards)
        episode_scale = float(np.percentile(abs_values, 90)) if abs_values.size else 0.0
        family_scale = self.family_dense_scale.get(family)
        if family_scale is None:
            family_scale = max(episode_scale, 1e-3)
        else:
            family_scale = max(0.9 * family_scale + 0.1 * max(episode_scale, 1e-3), 1e-3)
        self.family_dense_scale[family] = family_scale
        clipped = np.clip(rewards, -3.0 * family_scale, 3.0 * family_scale)
        normalized = clipped / family_scale
        normalized *= multiplier
        return normalized.astype(np.float32).tolist()

    def assign_bands(
        self,
        workflows: Iterable[str],
        quantiles: Tuple[float, float] = (0.33, 0.66),
    ) -> Dict[str, str]:
        workflows = list(workflows)
        if not workflows:
            return {}
        baseline_values: List[Tuple[str, float]] = []
        for wf in workflows:
            key = Path(wf).name
            baseline = self.workflow_baselines.get(key)
            family, stats = self._get_stats(wf)
            if baseline is None and stats is not None:
                baseline = stats.baseline_mean
            if baseline is None:
                # Fall back to makespan surrogate = 1.0 to keep workflow in rotation.
                baseline = 1.0
            baseline_values.append((wf, float(baseline)))
        values = np.array([b for _, b in baseline_values], dtype=np.float64)
        if values.size == 0:
            return {}
        q_low = float(np.quantile(values, quantiles[0]))
        q_high = float(np.quantile(values, quantiles[1]))
        band_map: Dict[str, str] = {}
        for wf, baseline in baseline_values:
            if baseline <= q_low:
                band = "short"
            elif baseline >= q_high:
                band = "long"
            else:
                band = "medium"
            band_map[wf] = band
        return band_map

    def get_ratio_clip(self, workflow_file: str) -> Tuple[float, float]:
        _, stats = self._get_stats(workflow_file)
        if stats:
            return stats.ratio_low, stats.ratio_high
        return self.min_ratio_low, max(self.min_ratio_high, 1.2)

    def summarize_rewards(self, workflow_file: str, rewards: Iterable[float]) -> Dict[str, float]:
        rewards = np.asarray(list(rewards), dtype=np.float64)
        if rewards.size == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "positive_frac": 0.0,
                "negative_frac": 0.0,
                "count": 0.0,
            }
        positive = float(np.sum(rewards > 0))
        negative = float(np.sum(rewards < 0))
        count = float(rewards.size)
        mean = float(np.mean(rewards))
        std = float(np.std(rewards))
        return {
            "mean": mean,
            "std": std,
            "positive_frac": positive / count,
            "negative_frac": negative / count,
            "count": count,
        }
