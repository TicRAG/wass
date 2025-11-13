# Experiment Parameters (Extreme Policy Ultra Suite)

## Workflow Assets
- `synthetic_workflow_001.json` (unchanged) — copied to `paper/workflows/`
- `seismology-chameleon-100p-001.json` — per-task `flops`, `runtime`, `runtimeInSeconds`, and all file `sizeInBytes` fields multiplied by 60 000 (initial ×3 000 followed by ×20); archived in `paper/workflows/`
- `montage-chameleon-2mass-01d-001_aug1.json` (unchanged) — copied to `paper/workflows/`

## Platform / Simulator
- Platform key: `extreme_hetero` → resolves to `configs/platform_extreme_hetero.xml`
- WRENCH daemon: running locally on `localhost:8101`
- Simulation seeds: `[0]`
- Repetitions per seed: `1`

## WASS-RAG Inference (policy)
- Model checkpoint: `models/saved_models/drl_agent.pth`
- Feature scaler: `models/saved_models/feature_scaler.joblib`
- Variant: `rag`
- `host_order_mode`: `policy_ultra` (host ordering forced to `cpu_host_micro`, `cpu_host_bottleneck`, `cpu_host_slow`, `cpu_host_balanced`, `cpu_host_ultra`, `cpu_host_fast`)
- `sample_top_k`: `1`
- `temperature`: `0.6`
- `epsilon`: `0.0`
- `greedy_threshold`: `1.1`
- `stochastic_tie_break`: `False`
- Teacher tracing: disabled (`--trace-log-dir` omitted)

## Baseline Schedulers
### HEFT
- Gaussian duration noise σ: `12`
- Bandwidth default: `100 000` bytes/s (scheduler default)

### MIN-MIN
- Communication scale: `1.0`
- Remote-edge penalty: `1500`
- Balance weight: `90`
- Availability weight: `0`

### FIFO
- Not executed in this suite.

## Runner Configuration
- Strategies evaluated: `WASS-RAG (Full)`, `HEFT`, `MIN-MIN`
- Workflow directory: `data/workflows/custom_eval`
- Output directory: `results/wass_rag_dual_teacher/extreme_policy_ultra_scaled_v4`
- Random seeds argument: `--seeds 0`
- `min_host_speed`: default `0.0`
- Trace logging arguments: omitted (no teacher traces)

## Result Highlights (Seed 0)
| Workflow | WASS-RAG Makespan | HEFT Makespan | MIN-MIN Makespan |
|----------|------------------:|--------------:|-----------------:|
| synthetic_workflow_001.json | 3075.34 | 3075.34 | 4993.05 |
| seismology-chameleon-100p-001.json | **1098.94** | 2592.27 | 5411.45 |
| montage-chameleon-2mass-01d-001_aug1.json | **347.70** | 6453.93 | 3041.60 |

All CSV artifacts are stored under `results/wass_rag_dual_teacher/extreme_policy_ultra_scaled_v4/`.

## WASS-RAG vs WASS-DRL (Vanilla) Comparison
- Output directory: `results/wass_rag_dual_teacher/extreme_policy_ultra_wass_vs`
- Strategies: `WASS-RAG (Full)` (settings above) and `WASS-DRL (Vanilla)` using checkpoint `models/saved_models/drl_agent_no_rag.pth`
- Shared inference parameters: `host_order_mode=policy_ultra`, `temperature=0.6`, `sample_top_k=1`, `epsilon=0.0`, `greedy_threshold=1.1`
- Seed configuration: identical (`--seeds 0`)

| Workflow | WASS-RAG Makespan | WASS-DRL Makespan |
|----------|------------------:|------------------:|
| Synthetic | 3075.34 | 3075.34 |
| Seismology | **1098.94** | 2592.27 |
| Montage | **347.70** | 629.94 |

CSV snapshots mirrored in `paper/results/extreme_policy_ultra_wass_vs_*`.
