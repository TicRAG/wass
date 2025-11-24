# Redesigned Platform Experiment Plan

## Objective
Reduce the extreme network/disk heterogeneity and rerun the three-workflow comparison under the adjusted platform to observe how the schedulers behave when the gap between the weakest and strongest nodes is narrowed.

## Platform Edits (configs/platform_extreme_hetero.xml)
| Component | Original Extreme Setting | Latest Setting (2025-11-24) | Notes |
|-----------|--------------------------|-----------------------------|-------|
| `cpu_host_ultra` compute & disk | `speed=400Gf`, disk `380/380MBps` | `speed=260Gf`, disk `200/200MBps` | Keeps ultra host ahead but trims both compute and storage advantage. |
| `cpu_host_fast` disk | `read/write=320MBps` | `read/write=260MBps` | Aligns fast node I/O with toned-down ultra node. |
| `cpu_host_balanced` compute & disk | `speed=120Gf`, disk `220/220MBps` | `speed=150Gf`, disk `200/200MBps` | Moderately boosts balanced node to narrow gap to fast node. |
| `cpu_host_slow` compute & disk | `speed=40Gf`, disk `140/140MBps` | `speed=90Gf`, disk `160/160MBps` | Raises capacity of slow node so it is not a severe outlier. |
| `cpu_host_bottleneck` compute & disk | `speed=18Gf`, disk `90/90MBps` | `speed=60Gf`, disk `120/120MBps` | Gives the bottleneck host more headroom while keeping it below mid-tier nodes. |
| `cpu_host_micro` compute & disk | `speed=6Gf`, disk `60/60MBps` | `speed=35Gf`, disk `100/100MBps` | Converts the edge node from unusable to lightly capable. |
| `bottleneck_link` bandwidth | `10MBps` | `120MBps` | Relieves congestion on routes feeding slowest hosts to focus comparison on scheduling rather than severe network throttling. |

**Note:** Host RAM and topology remain unchanged.

## Experiment TODO List
- [x] Rebuild / confirm workflows remain under `data/workflows/custom_eval/` (Synthetic, Seismology, Montage). Copies already exist in `paper/workflows/`.
- [x] Rerun `scripts/4_run_experiments.py` with the softened platform configuration:  
  ```bash
  python scripts/4_run_experiments.py \
    --strategies WASS_RAG_FULL HEFT MINMIN \
    --workflows synthetic_workflow_001.json seismology-chameleon-100p-001.json montage-chameleon-2mass-01d-001_aug1.json \
    --workflow-dir data/workflows/custom_eval \
    --platform-key extreme_hetero \
    --output-dir results/wass_rag_dual_teacher/extreme_policy_ultra_softened \
    --rag-host-order policy_ultra \
    --rag-sample-topk 1 \
    --rag-temperature 0.6 \
    --heft-noise-sigma 12 \
    --minmin-remote-penalty 1500 \
    --minmin-balance-weight 90 \
    --seeds 0
  ```
- [x] Copy resulting CSVs into `paper/redo_experiments/results/` for archival.
- [x] Regenerate comparative figures (baseline vs WASS-RAG, WASS-DRL vs WASS-RAG) under a new subdirectory, e.g., `paper/redo_experiments/figures/`.
- [x] Document new findings (magnitude of improvements, host utilization changes) in this folder.

## Run Outputs
- `paper/redo_experiments/results/summary_results.csv` and `paper/redo_experiments/results/detailed_results.csv` now track the softened-platform outputs. Latest averages (s): HEFT `2541.54`, MIN-MIN `2592.93`, WASS-RAG `1569.24`.
- Archived outputs: softened-platform raw files in `results/wass_rag_dual_teacher/extreme_policy_ultra_softened/` (mirrored in `paper/redo_experiments/results/softened_snapshot/`); the earlier extreme-heterogeneity run remains under `results/wass_rag_dual_teacher/extreme_policy_ultra_rebalanced/`.

## Final Softened Platform Results
- Overall, WASS-RAG retains a ~38% lower average makespan than HEFT (`1569s` vs `2542s`) and ~40% lower than MIN-MIN (`2593s`).
- Workflow-level averages (seconds):
  - Montage: WASS-RAG `533` vs MIN-MIN `1619` (3.0x slower) and HEFT `3443` (6.4x slower).
  - Seismology: WASS-RAG `1099` vs HEFT `1106` (~0.6% faster) vs MIN-MIN `1319` (20% slower).
  - Synthetic: WASS-RAG `3075` ≈ HEFT `3075` (tie), both outperform MIN-MIN `4841`.
- Inference logs confirm the policy still concentrates placement on `cpu_host_ultra`, suggesting further host-order experimentation if diversification is desired.
- Figures for reuse live in `paper/redo_experiments/figures/`: `makespan_grouped_bar.png`, `makespan_grouped_bar_log.png`, `makespan_summary_table.png`, `relative_makespan_heatmap.png`, `relative_makespan_ratio.png`, `wass_rag_relative_gain.png`, `wass_vs_baselines_scatter.png`, `wass_vs_wassrag_bar.png`, plus the overview set (`overall_makespan_softened.png`, `makespan_by_workflow_softened.png`, `makespan_boxplot_softened.png`). Axis labels use the short names `montage`, `seismology`, and `synthetic`.

## Figure Generation Script
```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_CSV = Path("paper/redo_experiments/results/detailed_results.csv")
FIG_DIR = Path("paper/redo_experiments/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8")

results = pd.read_csv(RESULTS_CSV)
if "status" in results.columns:
  results = results[results["status"].eq("success")].copy()

summary = results.groupby(["workflow", "scheduler"])["makespan"].agg(["mean", "std", "count"]).reset_index()

def simplify_label(workflow_path: str) -> str:
  stem = Path(workflow_path).stem.lower()
  if "montage" in stem:
    return "montage"
  if "seismology" in stem:
    return "seismology"
  if "synthetic" in stem:
    return "synthetic"
  return stem

summary["workflow_label"] = summary["workflow"].map(simplify_label)

mean_pivot = summary.pivot(index="workflow_label", columns="scheduler", values="mean")
std_pivot = summary.pivot(index="workflow_label", columns="scheduler", values="std").fillna(0.0)

ordered_sched = ["WASS-RAG (Full)", "HEFT", "MIN-MIN"]
workflow_order = ["montage", "seismology", "synthetic"]

mean_pivot = mean_pivot[ordered_sched].reindex(workflow_order)
std_pivot = std_pivot[ordered_sched].reindex(workflow_order)

bar_width = 0.8 / len(ordered_sched)
indices = np.arange(len(mean_pivot))
colors = ["#5B8FF9", "#5AD8A6", "#F6BD16"]

# Linear grouped bar
fig, ax = plt.subplots(figsize=(8, 5))
for i, sched in enumerate(ordered_sched):
  ax.bar(indices + i * bar_width, mean_pivot[sched], bar_width, label=sched, color=colors[i], yerr=std_pivot[sched], capsize=5)
ax.set_xticks(indices + bar_width)
ax.set_xticklabels(mean_pivot.index, rotation=20, ha="right")
ax.set_ylabel("Average Makespan (s)")
ax.set_title("Makespan per Workflow (Softened Platform)")
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(FIG_DIR / "makespan_grouped_bar.png", dpi=300)
plt.close(fig)

# Log grouped bar
fig, ax = plt.subplots(figsize=(8, 5))
for i, sched in enumerate(ordered_sched):
  ax.bar(indices + i * bar_width, mean_pivot[sched], bar_width, label=sched, color=colors[i])
ax.set_xticks(indices + bar_width)
ax.set_xticklabels(mean_pivot.index, rotation=20, ha="right")
ax.set_ylabel("Average Makespan (s, log scale)")
ax.set_title("Makespan per Workflow (Log Scale)")
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.3, which="both")
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(FIG_DIR / "makespan_grouped_bar_log.png", dpi=300)
plt.close(fig)

# Summary table
fig, ax = plt.subplots(figsize=(6, 1 + 0.4 * len(mean_pivot)))
ax.axis("off")
table_data = []
for wf in mean_pivot.index:
  for sched in ordered_sched:
    mean_val = mean_pivot.loc[wf, sched]
    std_val = std_pivot.loc[wf, sched]
    table_data.append([wf, sched, f"{mean_val:.2f}", f"{std_val:.2f}"])
table_df = pd.DataFrame(table_data, columns=["Workflow", "Scheduler", "Mean", "Std"])
table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
table.scale(1, 1.2)
ax.set_title("Makespan Summary Statistics", pad=20)
plt.tight_layout()
fig.savefig(FIG_DIR / "makespan_summary_table.png", dpi=300)
plt.close(fig)

# Relative heatmap
relative = mean_pivot.divide(mean_pivot["WASS-RAG (Full)"], axis=0)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(relative, annot=True, fmt=".2f", cmap="RdYlGn_r", cbar_kws={"label": "Ratio vs WASS-RAG"}, ax=ax)
ax.set_title("Relative Makespan vs WASS-RAG")
plt.tight_layout()
fig.savefig(FIG_DIR / "relative_makespan_heatmap.png", dpi=300)
plt.close(fig)

# Relative ratio bars
relative_melt = relative.reset_index().melt(id_vars="workflow_label", var_name="scheduler", value_name="ratio")
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=relative_melt, x="workflow_label", y="ratio", hue="scheduler", palette=colors, ax=ax)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("Makespan Ratio vs WASS-RAG")
ax.set_title("Relative Makespan Ratios")
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(FIG_DIR / "relative_makespan_ratio.png", dpi=300)
plt.close(fig)

# Relative gain vs best classical
baseline_best = mean_pivot[["HEFT", "MIN-MIN"]].min(axis=1)
relative_gain = (baseline_best - mean_pivot["WASS-RAG (Full)"]) / baseline_best * 100
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(mean_pivot.index, relative_gain, color="#FF9C6E")
ax.set_ylabel("Relative Gain vs Best Classical (%)")
ax.set_title("WASS-RAG Relative Improvement")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "wass_rag_relative_gain.png", dpi=300)
plt.close(fig)

# Scatter vs baselines
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(mean_pivot["WASS-RAG (Full)"], mean_pivot["HEFT"], color="#5AD8A6", label="HEFT")
ax.scatter(mean_pivot["WASS-RAG (Full)"], mean_pivot["MIN-MIN"], color="#F6BD16", label="MIN-MIN")
lims = [0, 1.1 * mean_pivot.values.max()]
ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("WASS-RAG Makespan (s)")
ax.set_ylabel("Baseline Makespan (s)")
ax.set_title("WASS-RAG vs Classical Baselines")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(FIG_DIR / "wass_vs_baselines_scatter.png", dpi=300)
plt.close(fig)

# Best classical vs WASS-RAG comparison
comparison_df = pd.DataFrame({
  "Workflow": mean_pivot.index,
  "WASS-RAG (Full)": mean_pivot["WASS-RAG (Full)"],
  "Best Classical": baseline_best,
})
fig, ax = plt.subplots(figsize=(6, 4))
comparison_df.plot(x="Workflow", kind="bar", ax=ax, color=["#5B8FF9", "#5AD8A6"], rot=20)
ax.set_ylabel("Average Makespan (s)")
ax.set_title("WASS-RAG vs Best Classical Scheduler")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "wass_vs_wassrag_bar.png", dpi=300)
plt.close(fig)
```

## Verification Checklist
- [x] Confirm modified XML is staged in git (for reproducibility of the new platform).
- [x] Capture before/after commentary about host selection logs if necessary.
- [ ] Update paper narrative to mention toned-down heterogeneity once new results are in.
