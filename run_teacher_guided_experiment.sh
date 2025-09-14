#!/bin/bash

# 教师引导的WASS-DRL和RAG运行脚本
# 该脚本将运行完整的教师引导实验流程

set -e

echo "🚀 开始教师引导的WASS-DRL和RAG实验流程"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请确保已安装Python3"
    exit 1
fi

# 检查必要的依赖
echo "📋 检查依赖..."
python3 -c "import torch, numpy, yaml, wrench" 2>/dev/null || {
    echo "❌ 缺少必要的Python依赖，请安装torch, numpy, yaml, wrench"
    exit 1
}

# 创建必要的目录
mkdir -p results/teacher_guided_experiments
mkdir -p models/checkpoints
mkdir -p src/knowledge_base

# 设置配置文件路径
CONFIG_FILE="configs/experiment.yaml"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "📁 使用配置文件: $CONFIG_FILE"

# 步骤1: 生成教师引导的知识库
echo ""
echo "📚 步骤1: 生成教师引导的知识库..."
python3 scripts/teacher_guided_kb_generator.py --config $CONFIG_FILE --num-cases 500

# 步骤2: 训练教师引导的DRL智能体
echo ""
echo "🧠 步骤2: 训练教师引导的DRL智能体..."
python3 scripts/teacher_guided_drl_trainer.py --config $CONFIG_FILE --episodes 200

# 步骤3: 运行综合实验
echo ""
echo "🧪 步骤3: 运行综合实验..."
python3 scripts/teacher_guided_experiment.py --config $CONFIG_FILE --runs 5 --workflow-sizes "5,10,15"

# 步骤4: 显示结果摘要
echo ""
echo "📊 步骤4: 显示结果摘要..."
if [ -f "results/teacher_guided_experiments/analysis_results.json" ]; then
    echo "📈 实验结果摘要:"
    python3 -c "
import json
with open('results/teacher_guided_experiments/analysis_results.json', 'r') as f:
    results = json.load(f)

print('\\n== 全局调度器性能 ==')
print(f'{'调度器':<15} {'平均Makespan':<15} {'标准差':<10} {'最佳':<10}')
print('-' * 60)

best_scheduler = None
best_makespan = float('inf')

for scheduler, stats in results.items():
    avg_makespan = stats['avg_makespan']
    std_makespan = stats['std_makespan']
    min_makespan = stats['min_makespan']
    
    print(f'{scheduler:<15} {avg_makespan:<15.2f} {std_makespan:<10.2f} {min_makespan:<10.2f}')
    
    if avg_makespan < best_makespan:
        best_makespan = avg_makespan
        best_scheduler = scheduler

print(f'\\n🏆 最佳调度器: {best_scheduler} (平均Makespan: {best_makespan:.2f}s)')
"
else
    echo "❌ 实验结果文件不存在"
fi

echo ""
echo "✅ 教师引导实验流程完成!"
echo "📁 结果保存在: results/teacher_guided_experiments/"
echo "📁 模型保存在: models/checkpoints/"
echo "📁 知识库保存在: src/knowledge_base/"