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
#   RAG_EPISODES=50        Override total episodes for RAG training script
#   DRL_EPISODES=50        Override total episodes for DRL-only training script
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

# Ensure required directories exist (including new split subdirs)
mkdir -p data/workflows
mkdir -p data/workflows/training
mkdir -p data/workflows/experiment
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

echo "ℹ️  手动划分模式: 本脚本不再自动复制/拆分 workflows。"
echo "   你需要自行将训练集放入 data/workflows/training/ ，实验集放入 data/workflows/experiment/。"
echo "   转换输出仍写入 data/workflows/ 根目录 (不会被移动)。"

###############################################################################
# Step 2: Validate converted workflows
###############################################################################
echo "🩺 [Step 2] Validating workflows (root, training/, experiment/)..."
if compgen -G "data/workflows/*.json" > /dev/null; then
	python scripts/validate_workflows.py --dir data/workflows || true
else
	echo "  ⚠️  Skip root validation (no *.json)."
fi
if compgen -G "data/workflows/training/*.json" > /dev/null; then
	python scripts/validate_workflows.py --dir data/workflows/training || true
else
	echo "  ❌ training/ 为空：请手动挑选并复制若干 *.json 到 data/workflows/training/ 后再运行后续步骤。"
fi
if compgen -G "data/workflows/experiment/*.json" > /dev/null; then
	python scripts/validate_workflows.py --dir data/workflows/experiment || true
else
	echo "  ⚠️ experiment/ 为空：最终实验将被跳过或无数据 (脚本4会直接退出)。"
fi

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
	if [[ -n "${RAG_EPISODES:-}" ]]; then
		echo "   ➤ Using RAG_EPISODES=${RAG_EPISODES} (override)"
		python scripts/2_train_rag_agent.py --max_episodes "${RAG_EPISODES}" || { echo "❌ RAG training failed"; exit 1; }
	else
		python scripts/2_train_rag_agent.py || { echo "❌ RAG training failed"; exit 1; }
	fi
else
	echo "⏭  Skipping RAG training (SKIP_TRAIN_RAG=1)."
fi

###############################################################################
# Step 5: Train DRL-only agent (optional skip)
###############################################################################
if [[ "${SKIP_TRAIN_DRL:-0}" != "1" ]]; then
	echo "🤖 [Step 5] Training DRL-only (no-RAG) agent..."
	if [[ -n "${DRL_EPISODES:-}" ]]; then
		echo "   ➤ Using DRL_EPISODES=${DRL_EPISODES} (override)"
		python scripts/3_train_drl_agent.py --max_episodes "${DRL_EPISODES}" || { echo "❌ DRL-only training failed"; exit 1; }
	else
		python scripts/3_train_drl_agent.py || { echo "❌ DRL-only training failed"; exit 1; }
	fi
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