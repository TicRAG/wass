import matplotlib.pyplot as plt
import numpy as np
import os

# Data
scenarios = [
    "Montage\n(Scaled)",
    "Montage\n(Extreme V2)",
    "Montage\n(Constrained)",
    "Epigenomics\n(Constrained)",
    "Epigenomics\n(Extreme V2)"
]

# Data from paper/vanilla/wass_rag_vs_drl_5_wins_summary.md
rag_times = [433.94, 1813.30, 1382.69, 42.42, 41.86]
drl_times = [629.94, 5906.82, 2762.54, 156.36, 182.16]
improvements = [31.1, 69.3, 49.9, 72.9, 77.0]

x = np.arange(len(scenarios))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))

# Plot bars
rects1 = ax.bar(x - width/2, drl_times, width, label='Vanilla (WASS-DRL)', color='#ff9999', edgecolor='white')
rects2 = ax.bar(x + width/2, rag_times, width, label='Full (WASS-RAG)', color='#66b3ff', edgecolor='white')

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Makespan (seconds) [Log Scale]')
ax.set_title('Performance Comparison: Vanilla vs Full (5 Key Scenarios)')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=10)
ax.legend()

# Use log scale to accommodate the wide range of values (40s to 6000s)
ax.set_yscale('log')
# Increase y-axis limit to make room for labels above the tallest bar
ax.set_ylim(top=max(drl_times) * 5)

# Function to add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

# Add improvement percentages
for i, imp in enumerate(improvements):
    # Place the improvement text above the taller bar (which is always DRL in these wins)
    height = drl_times[i]
    ax.annotate(f'Imp:\n+{imp}%',
                xy=(x[i], height),
                xytext=(0, 20),  # higher offset to clear the bar label
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='-[, widthB=1.0, lengthB=0.2', lw=1.0, color='green'))

plt.grid(axis='y', linestyle='--', alpha=0.3, which='major')
fig.tight_layout()

# Save the plot
output_dir = 'paper/figures'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '5_wins_comparison.png')
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
