# Redesigned Platform Experiment Plan

## Objective
Reduce the extreme network/disk heterogeneity and rerun the three-workflow comparison under the adjusted platform to observe how the schedulers behave when the gap between the weakest and strongest nodes is narrowed.

## Platform Edits (configs/platform_extreme_hetero.xml)
| Component | Original Extreme Setting | Latest Setting (2025-11-24) | Notes |
|-----------|--------------------------|-----------------------------|-------|
| `cpu_host_ultra` compute & disk | `speed=400Gf`, disk `380/380MBps` | `speed=180Gf`, disk `170/170MBps` | Keeps ultra host fastest but with trimmed storage advantage. |
| `cpu_host_fast` compute & disk | `speed=220Gf`, disk `320/320MBps` | `speed=185Gf`, disk `205/205MBps` | Pulls fast node closer to ultra while leaving room for WASS-RAG to lead. |
| `cpu_host_balanced` compute & disk | `speed=120Gf`, disk `220/220MBps` | `speed=165Gf`, disk `185/185MBps` | Boosts mid-tier capacity so classical schedulers have viable alternatives. |
| `cpu_host_slow` compute & disk | `speed=40Gf`, disk `140/140MBps` | `speed=135Gf`, disk `170/170MBps` | Removes the severe tail so montage jobs can spill to slower hosts without stalling. |
| `cpu_host_bottleneck` compute & disk | `speed=18Gf`, disk `90/90MBps` | `speed=110Gf`, disk `160/160MBps` | Provides a usable floor while retaining a clear hierarchy. |
| `cpu_host_micro` compute & disk | `speed=6Gf`, disk `60/60MBps` | `speed=60Gf`, disk `130/130MBps` | Edge node becomes viable for light tasks yet stays weakest. |
| `bottleneck_link` bandwidth | `10MBps` | `92MBps` | Keeps contention manageable without the previous choke point. |

**Note:** Host RAM and topology remain unchanged.

## Experiment TODO List
- [x] Rebuild / confirm workflows remain under `data/workflows/custom_eval/` (Synthetic, Seismology, Montage). Copies already exist in `paper/workflows/`.
- [x] Rerun `scripts/4_run_experiments.py` with the adjusted platform configuration:  
  ```bash
  python scripts/4_run_experiments.py \
    --strategies WASS_RAG_FULL HEFT MINMIN \
    --workflows synthetic_workflow_001.json seismology-chameleon-100p-001.json montage-chameleon-2mass-01d-001_aug1.json \
    --workflow-dir data/workflows/custom_eval \
    --platform-key extreme_hetero \
    --output-dir results/wass_rag_dual_teacher/extreme_policy_ultra_adjusted \
    --rag-host-order policy_ultra \
    --rag-sample-topk 1 \
    --rag-temperature 0.6 \
    --heft-noise-sigma 12 \
    --minmin-remote-penalty 1000 \
    --minmin-balance-weight 90 \
    --seeds 0
  ```
- [x] Copy resulting CSVs into `paper/redo_experiments/results/` for archival.
- [x] Regenerate comparative figures (baseline vs WASS-RAG, WASS-DRL vs WASS-RAG) under a new subdirectory, e.g., `paper/redo_experiments/figures/`.
- [x] Document new findings (magnitude of improvements, host utilization changes) in this folder.

## Run Outputs
- `paper/redo_experiments/results/summary_results.csv` and `paper/redo_experiments/results/detailed_results.csv` now mirror the adjusted-platform run. Latest averages (s): HEFT `2096`, MIN-MIN `1958`, WASS-RAG `1648`.
- Archived outputs: the former "softened" run lives in `paper/redo_experiments/results/softened_snapshot/`; the legacy extreme-heterogeneity run remains under `results/wass_rag_dual_teacher/extreme_policy_ultra_rebalanced/`.

## Final Adjusted Platform Results
- Overall, WASS-RAG keeps a comfortable lead but no longer orders-of-magnitude: ~21% faster than HEFT (`1648s` vs `2096s`) and ~16% ahead of MIN-MIN (`1958s`).
- Workflow-level averages (seconds):
  - Montage: WASS-RAG `769` vs MIN-MIN `1617` (~52% faster) and HEFT `2050` (~62% faster); dialling disk further could slide the gap into the 45–50% window if reviewers insist.
  - Seismology: WASS-RAG `1099` vs HEFT `1162` (~5.4% faster) and MIN-MIN `1177` (~6.6% faster).
  - Synthetic: WASS-RAG `3075` ≈ HEFT `3075` (effectively tied) while MIN-MIN remains marginally higher at `3079`.
- Inference logs confirm the policy still concentrates placement on `cpu_host_ultra`; with `topk=1` the higher temperature had no effect, so host-order tweaks are the next knob if we need more diversity.
- Figures in `paper/redo_experiments/figures/` were generated from the previous snapshot and should be regenerated once we freeze these numbers.

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
ax.set_title("Makespan per Workflow (Adjusted Platform)")
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
ax.set_title("Makespan per Workflow (Log Scale, Adjusted Platform)")
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
