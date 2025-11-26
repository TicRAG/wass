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
