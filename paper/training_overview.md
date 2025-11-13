# Training Overview

## Pipeline Stages
1. **Knowledge Base Seeding**
   - Script: `scripts/1_seed_knowledge_base.py`
   - Inputs: WFCommons metadata (`data/knowledge_base/workflow_metadata.csv`) and workflow embeddings.
   - Output: Populated FAISS index (`data/knowledge_base/workflow_embeddings.index`) consumed by the RAG teacher.
2. **RAG Policy Pretraining**
   - Script: `scripts/2_train_rag_agent.py`
   - Key settings (from `configs/training_config.yaml`):
     - Optimizer: Adam, learning rate `3e-4`
     - Batch size: `64` tasks per update
     - Discount factor γ: `0.99`
     - GNN encoder: `DecoupledGNNEncoder` with `(in=4, hidden=64, out=32)`
   - Checkpoint artifact: `models/saved_models/drl_agent.pth`
3. **Pure DRL Fine-tuning (Optional)**
   - Script: `scripts/3_train_drl_agent.py`
   - Produces RAG-free baseline policy `models/saved_models/drl_agent_no_rag.pth`
4. **Experiment Execution**
   - Script: `scripts/4_run_experiments.py`
   - Leverages trained checkpoints plus baseline schedulers for evaluation on held-out workflows.

## Notable Hyperparameters
- PPO horizon: `1024` steps
- PPO epochs: `4`
- Generalized advantage λ: `0.95`
- Entropy coefficient: `0.01`
- Value loss coefficient: `0.5`
- Gradient clipping: `0.5`

## Training Data Splits
- **Training workflows**: `data/workflows/training/` and `data/workflows/training_aug/`
- **Evaluation workflows**: `data/workflows/custom_eval/`

## Logging & Monitoring
- TensorBoard logs: `runs/` (created during training)
- Checkpoints stored under `models/saved_models/`

## Suggested Figures
- Loss & reward curves from TensorBoard scalars (export to CSV for publication)
- Training pipeline schematic (see `paper/figures/training_pipeline.png` once generated)
