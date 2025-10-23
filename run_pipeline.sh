#!/bin/bash
set -euo pipefail

###############################################################################
# WASS-RAG End-to-End Pipeline
# 1. Activate environment (if present)
# 2. Convert WFCommons workflows -> data/workflows/*.json (with flops/memory/runtime)
# 3. Validate converted workflows
# 4. Seed Knowledge Base (uses converted workflows)
# 5. Train RAG-enabled agent
# 6. Train DRL-only agent
# 7. Run final experiments
#
# Optional env vars:
#   SKIP_CONVERT=1         Skip conversion step
#   SKIP_TRAIN_RAG=1       Skip RAG training
#   SKIP_TRAIN_DRL=1       Skip DRL-only training
#   SKIP_EXPERIMENTS=1     Skip final experiments
#   CLEAN=1                Clean previous results/models (keeps converted workflows)
#
###############################################################################

echo "🚀 [WASS-RAG Pipeline] Starting..."

# Activate Python environment if available
if [[ -d "${HOME}/venvs/wrench-env" ]]; then
	# shellcheck disable=SC1090
	source "${HOME}/venvs/wrench-env/bin/activate"
	echo "✅ Activated Python env: wrench-env"
else
	echo "⚠️  Python env ~/venvs/wrench-env not found; proceeding with current interpreter." 
fi

if [[ "${CLEAN:-0}" == "1" ]]; then
	echo "🧹 [Clean] Removing old results, knowledge base, and models (keeping workflows)..."
	rm -rf results/* || true
	rm -rf data/knowledge_base/* || true
	rm -rf models/saved_models/* || true
fi

# Ensure required directories exist
mkdir -p data/workflows
mkdir -p data/knowledge_base
mkdir -p results/final_experiments
mkdir -p models/saved_models

echo "📁 Directory structure ready."

###############################################################################
# Step 1: Convert WFCommons workflows (unless skipped)
###############################################################################
if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
	echo "🔄 [Step 1] Converting WFCommons workflows..."
	python scripts/0_convert_wfcommons.py --input_dir configs/wfcommons --output_dir data/workflows
else
	echo "⏭  Skipping conversion step (SKIP_CONVERT=1)."
fi

###############################################################################
# Step 2: Validate converted workflows
###############################################################################
echo "🩺 [Step 2] Validating converted workflows..."
python scripts/validate_workflows.py --dir data/workflows

###############################################################################
# Step 3: Seed Knowledge Base
###############################################################################
echo "🧠 [Step 3] Seeding Knowledge Base..."
python scripts/1_seed_knowledge_base.py

###############################################################################
# Step 4: Train RAG-enabled agent (optional skip)
###############################################################################
if [[ "${SKIP_TRAIN_RAG:-0}" != "1" ]]; then
	echo "🎓 [Step 4] Training RAG-enabled agent..."
	python scripts/2_train_rag_agent.py
else
	echo "⏭  Skipping RAG training (SKIP_TRAIN_RAG=1)."
fi

###############################################################################
# Step 5: Train DRL-only agent (optional skip)
###############################################################################
if [[ "${SKIP_TRAIN_DRL:-0}" != "1" ]]; then
	echo "🤖 [Step 5] Training DRL-only (no-RAG) agent..."
	python scripts/3_train_drl_agent.py
else
	echo "⏭  Skipping DRL-only training (SKIP_TRAIN_DRL=1)."
fi

###############################################################################
# Step 6: Final experiments (optional skip)
###############################################################################
if [[ "${SKIP_EXPERIMENTS:-0}" != "1" ]]; then
	echo "📊 [Step 6] Running final experiments..."
	python scripts/4_run_experiments.py
else
	echo "⏭  Skipping experiments (SKIP_EXPERIMENTS=1)."
fi

echo "🎉 [WASS-RAG Pipeline] All requested steps completed!"