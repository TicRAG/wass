#!/bin/bash
# WASS项目虚拟环境激活脚本 (Linux/macOS)

echo "🚀 激活WASS虚拟环境..."
source ./wass_env/bin/activate

echo "✅ 虚拟环境已激活！"
echo ""
echo "💡 可用的命令:"
echo "  - python demo.py                                    # 运行完整演示"
echo "  - python scripts/run_lf_experiments.py            # Label Function实验"
echo "  - python scripts/run_model_comparison.py          # 标签模型对比"
echo "  - python scripts/analyze_results.py results/ --report  # 分析结果"
echo "  - python -m src.pipeline_enhanced configs_example.yaml  # 单个实验"
echo ""
echo "📖 查看 EXPERIMENT_GUIDE.md 了解详细实验指南"
echo ""

# 启动新的shell会话以保持环境激活
exec $SHELL
