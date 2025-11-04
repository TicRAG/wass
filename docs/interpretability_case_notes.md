# Trace Case Study Notes

## Aggregate Trace Insights (extreme_top3_noise01)
- WASS-RAG inference yielded identical shaped-reward traces across all five seeds per workflow, confirming deterministic replay when teacher logging is enabled.
- Montage (`montage-chameleon-2mass-01d-001.json`) exhibits the widest potential swings: shaped rewards span `[-0.85, +0.78]`, linked to `mDiffFit` tasks. These episodes reveal both strongly penalised and rewarded assignments, making them prime positive/negative contrast cases.
- Seismology (`seismology-chameleon-100p-001.json`) stays strictly negative with mean shaped reward `-0.235`, and recurrent high-magnitude penalties (`-0.59`) on `sG1IterDecon` tasks; this highlights where the teacher signal consistently discourages the agent.
- Epigenomics (`epigenomics-chameleon-hep-1seq-100k-001.json`) produces compact signals (mean `-0.035`, min `-0.11`), implying the retrieval neighbourhood sits near the decision frontier with modest guidance.
- Synthetic workloads flatten around `-0.0087` with zero variance, suggesting the stored KB entries already mirror the deterministic policies—useful as a sanity baseline but low interpretive value.

## Recommended Visualization Set
1. **Montage seed0** (`charts/trace_extreme_top3/montage_seed0.png`): displays both extreme positive and negative shaped rewards; annotate contrasting `mDiffFit` job placements.
2. **Seismology seed0** (`charts/trace_extreme_top3/seismology_seed0.png`): captures persistent negative potentials; focus discussion on repeated `sG1IterDecon` penalties and absence of positive deltas.
3. **Epigenomics seed0** (`charts/trace_extreme_top3/epigenomics_seed0.png`): use as “steady-state” example showing narrow reward band and mostly homogeneous host choices.
4. **Optional sanity**: take one synthetic workflow plot if appendix needs demonstration of converged/no-variation behaviour; otherwise omit from main narrative.

## Narrative Draft (WIP)
### Montage seed0 — contrastive spikes
`results/traces/extreme_top3_noise01_summary/montage-chameleon-2mass-01d-001/montage-chameleon-2mass-01d-001_seed0_rep0_20251104T060238/trace_summary.json` records the widest dynamic range in the batch (shaped reward span `[-0.851, +0.779]`). The Gantt view in `charts/trace_extreme_top3/montage_seed0.png` highlights alternating bursts inside the `mDiffFit` fan-out: positive peaks (e.g. `mDiffFit_ID0000084`, +0.779) follow long compute stretches where the teacher predicts a steep potential jump, yet the top-10 neighbours in `trace_neighbors.csv` all map back to `HEFT` replays with `q≈0`. Conversely, the heavy penalties (`mDiffFit_ID0000085`, `mDiffFit_ID0000011` at -0.851) appear after similar fan-in states but where the KB again offers near-zero `q` mass. The juxtaposition suggests the agent amplifies minute score differences from duplicated HEFT exemplars; tie-breaking noise or richer neighbour diversity should soften these cliffs.

### Seismology seed0 — sustained negatives
`results/traces/extreme_top3_noise01_summary/seismology-chameleon-100p-001/seismology-chameleon-100p-001_seed0_rep0_20251104T060251/trace_summary.json` shows a narrow, strictly negative band (mean `-0.235`, min `-0.593`). In `charts/trace_extreme_top3/seismology_seed0.png` the entire `sG1IterDecon` cascade sits below zero with no compensating upticks. The neighbour table collapses every decision onto two identical `HEFT` runs with `q=1.0`, so the teacher keeps signalling “far from completion” regardless of placement. The lack of gradient in retrieved examples implies the KB cannot distinguish subtle host ordering effects for this workload; we should call out how the agent follows deterministic host reuse (`cpu_host_micro`) despite repetitive penalties.

### Epigenomics seed0 — narrow frontier
For `results/traces/extreme_top3_noise01_summary/epigenomics-chameleon-hep-1seq-100k-001/epigenomics-chameleon-hep-1seq-100k-001_seed0_rep0_20251104T060234/trace_summary.json`, rewards cluster between `-0.110` and `-0.009` with mean `-0.035`. The chart (`charts/trace_extreme_top3/epigenomics_seed0.png`) shows a gentle braid where short `fast2bfq` and `map` decisions oscillate around zero. Unlike the other cases, `trace_neighbors.csv` mixes Random and HEFT augmentations with dense `q≈1.0` support, indicating the KB already encodes near-optimal completions; shaped rewards only tweak the schedule when compute/communication jitter accumulates. This makes the workflow an ideal “calibration” paragraph demonstrating behaviour when the agent and teacher agree.

## Next Actions
- Fold the paragraph drafts and stochastic findings into the案例解读草稿, then circulate for review.
- Quantify per-seed variance deltas (deterministic vs. stochastic) and decide whether to swap the montage panel.

## Stochastic Tie-Break Probe Plan
- Add a `--stochastic-tie-break` switch to `scripts/4_run_experiments.py` that flips `WASS_DRL_Scheduler_Inference` into sampling mode (`deterministic=False`).
- Re-run WASS-RAG (Full) on `montage-chameleon-2mass-01d-001.json` for seeds `0-4` with trace logging into `results/traces/extreme_top3_noise01_stochastic/`.
- Reuse `analysis/interpretability_case_study.py` to emit `trace_summary.json` per seed and compare reward spans against the deterministic baseline.
- If spikes persist, inspect host assignment sequences for injected randomness; otherwise, document reduced variance in the narrative.

## Stochastic Tie-Break Probe – Findings
- Command: `python scripts/4_run_experiments.py --strategies WASS_RAG_FULL --workflows montage-chameleon-2mass-01d-001.json --seeds 0 1 2 3 4 --trace-log-dir results/traces/extreme_top3_noise01_stochastic --output-dir results/stochastic_montage --stochastic-tie-break`.
- Outputs: `results/traces/extreme_top3_noise01_stochastic_summary/montage-chameleon-2mass-01d-001/*/trace_summary.json` and `results/stochastic_montage/{detailed,summary}_results.csv`.
- Reward span per seed (mean / min / max):
	- seed0: `0.066 / -0.787 / +0.747`
	- seed1: `-0.063 / -0.885 / +0.747`
	- seed2: `0.056 / -0.811 / +0.803`
	- seed3: `-0.012 / -0.787 / +0.743`
	- seed4: `0.036 / -0.885 / +0.743`
- Compared to the deterministic run (single-host `cpu_host_micro`), stochastic sampling spreads decisions across all six hosts (see per-host reward means in the new summaries) and drags the minimum rewards lower while keeping similar positive peaks. Variance now stems from divergent host choices rather than identical HEFT neighbours.
