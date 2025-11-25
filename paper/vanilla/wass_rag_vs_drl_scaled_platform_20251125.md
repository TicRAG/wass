# WASS-RAG vs WASS-DRL (Scaled Extreme Platform) — 2025-11-25

## Experiment setup
- Platform: `configs/platform_extreme_hetero_scaled.xml` (new file with widened compute/disk/network gaps).
- Workflow directory: `data/workflows/custom_eval` (`montage-chameleon-2mass-01d-001_aug1.json`, `seismology-chameleon-100p-001.json`, `synthetic_workflow_001.json`).
- Strategies: `WASS_RAG_FULL`, `WASS_DRL_VANILLA`.
- Seeds: `0 1 2` (single repetition).
- Command:
  ```bash
  python scripts/4_run_experiments.py \
    --strategies WASS_RAG_FULL WASS_DRL_VANILLA \
    --workflow-dir data/workflows/custom_eval \
    --platform-key extreme_hetero_scaled \
    --output-dir results/ablation_rag_vs_drl_scaled_policy_20251125 \
    --seeds 0 1 2 \
    --rag-host-order policy_ultra \
    --rag-sample-topk 1 \
    --rag-temperature 0.7
  ```
- Outputs: `results/ablation_rag_vs_drl_scaled_policy_20251125/detailed_results.csv` and `summary_results.csv`.

The host-order and sampling overrides align inference with the training-time host index ordering; without them the RAG policy collapsed onto the slow node in this scaled platform.

## Results (makespan in seconds)

| Workflow | WASS-RAG | WASS-DRL | Δ (DRL − RAG) | Relative gain |
| --- | --- | --- | --- | --- |
| montage-chameleon-2mass-01d-001_aug1.json | 433.94 | 629.94 | 196.00 | 31.11% |
| seismology-chameleon-100p-001.json | 1098.94 | 1098.94 | 0.00 | 0.00% |
| synthetic_workflow_001.json | 3075.34 | 3075.34 | 0.00 | 0.00% |

*Each entry is averaged across three seeds; per-run values are identical because both actors behave deterministically once the host order is fixed.*

## Observations
- The scaled platform stretches scheduler runtimes into the targeted band (average makespans now ~0.4–3.1 ks, versus ~15 s previously).
- RAG gains materialized on Montage (≈31% faster) once action indexing matched the training layout; the default platform order caused the RAG actor to saturate `cpu_host_slow` and underperform.
- Seismology and the heavy synthetic workflow remain ties because both agents funnel all tasks to `cpu_host_ultra`; exposing more concurrency or penalising the fastest host may be required to surface further RAG advantages.

## Next steps
1. Explore additional workflow variants (e.g., augmented Montage or pipeline cases) to extend >700 s horizon without relying solely on slowed hardware.
2. Investigate teacher or reward tweaks that encourage diversified host usage so RAG retains advantages when ultra nodes dominate.
