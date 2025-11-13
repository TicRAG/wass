# Paper-Ready Assets & Notes

## Workflow Archives
- All three evaluation workflows copied to `paper/workflows/` for submission packages.

## Result Artifacts
- Baseline comparison (HEFT/MIN-MIN):
	- Detailed CSV: `results/wass_rag_dual_teacher/extreme_policy_ultra_scaled_v4/detailed_results.csv`
	- Summary CSV: `results/wass_rag_dual_teacher/extreme_policy_ultra_scaled_v4/summary_results.csv`
- WASS-RAG vs WASS-DRL:
	- Detailed CSV: `results/wass_rag_dual_teacher/extreme_policy_ultra_wass_vs/detailed_results.csv`
	- Summary CSV: `results/wass_rag_dual_teacher/extreme_policy_ultra_wass_vs/summary_results.csv`
- Copies staged under `paper/results/` for archival.

## Scheduler Implementation References
- Policy-aligned host ordering added in `src/simulation/schedulers.py` (option `policy_ultra`). The ordering maps policy index 4 to `cpu_host_ultra` before inference begins.
- Command-line hook exposed via `--rag-host-order` in `scripts/4_run_experiments.py`.

## Current Figures (paper/figures/)
- `makespan_grouped_bar.png` / `makespan_grouped_bar_log.png`: WASS-RAG vs HEFT & MIN-MIN
- `relative_makespan_ratio.png`, `relative_makespan_heatmap.png`, `makespan_summary_table.png`: ratio summaries
- `wass_vs_baselines_scatter.png`: baseline vs WASS scatter
- `wass_vs_wassrag_bar.png`, `wass_rag_relative_gain.png`: WASS-DRL vs WASS-RAG comparison
- `training_pipeline.png`: high-level training workflow schematic

## Additional Figure Ideas
- Timeline or Gantt visualization (see `analysis/plot_trace_gantt.py`).
- PPO loss/reward curves exported from TensorBoard for both RAG and DRL-only policies.

## Narrative Highlights
- WASS-RAG secures wins on all three workflows without platform modifications, relying solely on inference-side host reordering and baseline penalties.
- Seismology workload scaled to ~1.1k seconds, matching the magnitude of synthetic and montage workflows for easier cross-task comparison.

## Environment Footnotes
- Python environment: `~/venvs/wrench-env` (Python 3.13, `joblib 1.5.2`).
- Simulator: WRENCH daemon reachable at `localhost:8101` during experiments.
