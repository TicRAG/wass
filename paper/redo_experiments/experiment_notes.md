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
- Figures for reuse live in `paper/redo_experiments/figures/`: `makespan_grouped_bar.png`, `makespan_grouped_bar_log.png`, `makespan_summary_table.png`, `relative_makespan_heatmap.png`, `relative_makespan_ratio.png`, `wass_rag_relative_gain.png`, `wass_vs_baselines_scatter.png`, `wass_vs_wassrag_bar.png`, plus the overview set (`overall_makespan_softened.png`, `makespan_by_workflow_softened.png`, `makespan_boxplot_softened.png`).

## Verification Checklist
- [x] Confirm modified XML is staged in git (for reproducibility of the new platform).
- [x] Capture before/after commentary about host selection logs if necessary.
- [ ] Update paper narrative to mention toned-down heterogeneity once new results are in.
