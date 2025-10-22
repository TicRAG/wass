#!/bin/bash
set -e

echo "🚀 [WASS-RAG Pipeline] Starting..."

# 清理旧数据
echo "🧹 [Step 0] Cleaning old data and results..."
rm -rf results/*
rm -rf data/*
rm -rf models/saved_models/*

# 确保目录存在 (避免脚本失败)
mkdir -p data
mkdir -p results
mkdir -p models/saved_models
mkdir -p data/knowledge_base # KnowledgeBase 可能会需要

# 注意：所有脚本路径都已更新为 scripts/ 目录
echo "🧠 [Step 1] Seeding Knowledge Base..."
python scripts/1_seed_knowledge_base.py

echo "🎓 [Step 2] Training RAG-enabled agent..."
python scripts/2_train_rag_agent.py

echo "🤖 [Step 3] Training DRL-only (no-RAG) agent..."
python scripts/3_train_drl_agent.py

echo "📊 [Step 4] Running final experiments..."
python scripts/4_run_experiments.py

echo "🎉 [WASS-RAG Pipeline] All steps completed successfully!"