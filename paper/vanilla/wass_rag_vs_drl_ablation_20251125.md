# WASS-RAG vs WASS-DRL Ablation (2025-11-25)

## Experiment setup
- Environment: `~/venvs/wrench-env` (Python, WRENCH simulator available)
- Command: `python scripts/4_run_experiments.py --strategies WASS_RAG_FULL WASS_DRL_VANILLA --output-dir results/ablation_rag_vs_drl_20251125 --seeds 0 1 2 3 4`
- Workflow set: `data/workflows/experiment` (epigenomics, montage, seismology, five synthetic variants)
- Seeds: 0, 1, 2, 3, 4 (single repetition)
- Checkpoints: default `models/saved_models/drl_agent.pth` (RAG) and `drl_agent_no_rag.pth` (DRL)

Generated artifacts are stored under `results/ablation_rag_vs_drl_20251125/` (`detailed_results.csv`, `summary_results.csv`).

## Key findings
- WASS-RAG matches or beats WASS-DRL on every workflow; gains are concentrated on the real WFCommons benchmarks.
- Synthetic workflows show identical makespan for both agents, indicating they follow the same deterministic schedule on those instances.

### Per-workflow makespan comparison

| Workflow | WASS-RAG (avg makespan) | WASS-DRL (avg makespan) | DRL - RAG (↓ better) | Relative improvement |
| --- | --- | --- | --- | --- |
| epigenomics-chameleon-hep-1seq-100k-001.json | 16.39 | 17.64 | 1.25 | 7.08% |
| montage-chameleon-2mass-01d-001.json | 14.35 | 20.56 | 6.21 | 30.20% |
| seismology-chameleon-100p-001.json | 0.37 | 0.58 | 0.21 | 36.14% |
| synthetic_workflow_000.json | 2924.94 | 2924.94 | 0.00 | 0.00% |
| synthetic_workflow_001.json | 4343.11 | 4343.11 | 0.00 | 0.00% |
| synthetic_workflow_002.json | 3892.99 | 3892.99 | 0.00 | 0.00% |
| synthetic_workflow_003.json | 4628.07 | 4628.07 | 0.00 | 0.00% |
| synthetic_workflow_004.json | 3398.62 | 3398.62 | 0.00 | 0.00% |

*Values are averages across all seeds (five runs per workflow). Detailed per-run data is available in `results/ablation_rag_vs_drl_20251125/detailed_results.csv`.*

## Conclusions
- RAG-enhanced policy yields consistent reductions in makespan (7%–36%) on real workloads relative to the DRL-only agent, confirming the benefit of retrieval augmentation.
- Parity on synthetic cases suggests those workloads may be too simple or not aligned with the training distribution; future ablations should focus on richer or perturbed synthetic scenarios.
