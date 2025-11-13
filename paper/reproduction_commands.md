# Reproduction Commands

1. Activate the project environment and start required services:
   ```bash
   source ~/venvs/wrench-env/bin/activate
   # ensure the WRENCH daemon is running on localhost:8101 before launching experiments
   ```

2. Run the three-workflow comparison on the shared `platform_extreme_hetero` configuration:
   ```bash
   python scripts/4_run_experiments.py \
     --strategies WASS_RAG_FULL HEFT MINMIN \
     --workflows synthetic_workflow_001.json seismology-chameleon-100p-001.json montage-chameleon-2mass-01d-001_aug1.json \
     --workflow-dir data/workflows/custom_eval \
     --platform-key extreme_hetero \
     --output-dir results/wass_rag_dual_teacher/extreme_policy_ultra_scaled_v4 \
     --rag-host-order policy_ultra \
     --rag-sample-topk 1 \
     --rag-temperature 0.6 \
     --heft-noise-sigma 12 \
     --minmin-remote-penalty 1500 \
     --minmin-balance-weight 90 \
     --seeds 0
   ```

All three workflow JSON files referenced above are mirrored under `paper/workflows/` for archival and submission use.

## Optional: Training the Policies
1. Seed the knowledge base embeddings (required once):
    ```bash
    python scripts/1_seed_knowledge_base.py \
       --workflow-metadata data/knowledge_base/workflow_metadata.csv \
       --output-index data/knowledge_base/workflow_embeddings.index
    ```
2. Train the RAG-enhanced scheduler (produces `models/saved_models/drl_agent.pth`):
    ```bash
    python scripts/2_train_rag_agent.py \
       --config configs/training_config.yaml \
       --logdir runs/rag_policy
    ```
3. Train the DRL-only baseline (produces `models/saved_models/drl_agent_no_rag.pth`):
    ```bash
    python scripts/3_train_drl_agent.py \
       --config configs/training_config.yaml \
       --logdir runs/drl_baseline
    ```
