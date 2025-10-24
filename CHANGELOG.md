# CHANGELOG

## 2025-10-24 Critical Logic Fixes & Consolidation

### Fixed & Consolidated
- Seeding Logic: Upgraded seeding to use the improved reseed implementation. The old `reseed_wfcommons_kb.py` has been merged into `scripts/1_seed_knowledge_base.py` and the former file removed. Nested loop bug fixed; augmented workflows (`training_aug/`) now included.
- DRL Training Scaler Consistency (`scripts/3_train_drl_agent.py`): Added feature scaler loading and passing to scheduler to match KB seeding distribution.
- Inference Scheduler Scaling (`src/simulation/schedulers.py`): `WASS_DRL_Scheduler_Inference` now loads feature scaler (or accepts injected) and applies scaling when building PyG data.
- Unified Reward Logic (`scripts/2_train_rag_agent.py` & pipeline): Dense mode distributes final penalty across all steps; pipeline enforces unified `reward_mode` via `REWARD_MODE` env var for fair comparisons.
- Pipeline Simplification (`run_pipeline.sh`): Removed `USE_RESEED` flag; improved seeding script is now the default.

### Added
- CHANGELOG documenting critical fixes and script consolidation prior to paper experiments.

### Notes
- Install dependencies from `requirements.txt` before running (`torch`, `torch_geometric`, `numpy`, `scikit-learn`, `joblib`, `wrench`).
- Knowledge Base now contains both HEFT and Random scheduler decisions across original and augmented workflows (with base key duplication for augmented suffix matching).

### Suggested Next Steps
- Add smoke test to assert KB contains >0 entries per scheduler and RAG reward non-zero on a sample workflow.
- Add unit tests for reward shaping equivalence (dense aggregated == final) within numerical tolerance.
