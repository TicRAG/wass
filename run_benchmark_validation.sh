#!/bin/bash

# WASS-RAG 基准验证实验脚本
# 在"公平赛道"模式下验证HEFT vs FIFO的性能对比

set -e

echo "🚀 启动WASS-RAG基准验证实验..."
echo "================================================"

# 设置实验参数
EXPERIMENT_NAME="benchmark_validation"
WORKFLOW_CCR=10.0  # 高CCR值，确保HEFT优势
REPETITIONS=5      # 重复实验次数
TASK_COUNTS="50,100,200"  # 测试的工作流规模
SCHEDULERS="HEFT,FIFO"    # 仅测试HEFT和FIFO

# 创建实验目录
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_DIR}"
mkdir -p "${EXPERIMENT_DIR}/workflows"
mkdir -p "${EXPERIMENT_DIR}/platforms"
mkdir -p "${EXPERIMENT_DIR}/results"

echo "📁 实验目录: ${EXPERIMENT_DIR}"
echo "🎯 工作流CCR: ${WORKFLOW_CCR}"
echo "🔁 重复次数: ${REPETITIONS}"
echo "📊 测试规模: ${TASK_COUNTS}"
echo "⚖️  调度器: ${SCHEDULERS}"

# 步骤1: 预生成所有测试用例（公平赛道模式）
echo ""
echo "📋 步骤1: 预生成固定测试用例..."
python3 << 'EOF'
import sys
import os
sys.path.append('scripts')

from workflow_generator import WorkflowGenerator
from platform_generator import PlatformGenerator
import json

# 实验参数
experiment_name = "benchmark_validation"
workflow_ccr = 10.0
repetitions = 5
task_counts = [50, 100, 200]
scales = ['small', 'medium', 'large']

# 创建测试用例
experiment_dir = f"experiments/{experiment_name}"
workflow_dir = f"{experiment_dir}/workflows"
platform_dir = f"{experiment_dir}/platforms"

# 确保目录存在
os.makedirs(workflow_dir, exist_ok=True)
os.makedirs(platform_dir, exist_ok=True)

# 生成所有测试用例
test_cases = []

print("🔧 生成工作流和平台配置...")

workflow_gen = WorkflowGenerator()
platform_gen = PlatformGenerator(seed=42)  # 固定种子确保可重现

for task_count in task_counts:
    for rep in range(repetitions):
        for scale in scales:
            # 生成工作流
            workflow_file = f"{workflow_dir}/workflow_montage_{task_count}_rep{rep}.json"
            workflow_path = workflow_gen.generate_single_workflow(
                pattern='montage',
                task_count=task_count,
                random_seed=42 + rep,  # 每个重复使用不同但固定的种子
                filename=f"workflow_montage_{task_count}_rep{rep}.json"
            )
            
            # 生成平台
            platform_file = platform_gen.generate_single_platform(
                scale=scale,
                repetition_index=rep,
                seed=42
            )
            
            test_case = {
                'workflow_file': workflow_path,
                'platform_file': platform_file,
                'task_count': task_count,
                'scale': scale,
                'repetition': rep,
                'ccr': workflow_ccr
            }
            test_cases.append(test_case)

# 保存测试用例列表
with open(f"{experiment_dir}/test_cases.json", 'w') as f:
    json.dump(test_cases, f, indent=2)

print(f"✅ 生成了 {len(test_cases)} 个测试用例")
print("📊 测试用例已保存到 test_cases.json")
EOF

# 步骤2: 运行公平实验
echo ""
echo "⚖️  步骤2: 在公平赛道上运行实验..."
python3 scripts/fair_experiment_controller.py \
    --mode "custom" \
    --patterns montage \
    --sizes 50 100 200 \
    --scales small medium large \
    --schedulers FIFO HEFT \
    --repeats 5

# 步骤3: 生成验证报告
echo ""
echo "📊 步骤3: 生成验证报告..."
python3 << 'EOF'
import pandas as pd
import json
import os
import numpy as np
import glob

# 查找最新的实验结果文件
results_dir = "results/fair_experiments"

# 查找最新的CSV结果文件
csv_files = glob.glob(f"{results_dir}/fair_experiment_results_*.csv")
if not csv_files:
    print("❌ 未找到实验结果文件")
    exit(1)

# 加载实验结果
latest_csv = max(csv_files, key=os.path.getctime)
print(f"📊 使用结果文件: {latest_csv}")

# 加载实验结果
df = pd.read_csv(latest_csv)

# 计算HEFT vs FIFO的对比
summary = []
for workflow_size in df['workflow_size'].unique():
    for scale in df['platform_scale'].unique():
        subset = df[(df['workflow_size'] == workflow_size) & (df['platform_scale'] == scale)]
        
        heft_makespan = subset[subset['scheduler'] == 'HEFT']['makespan'].mean()
        fifo_makespan = subset[subset['scheduler'] == 'FIFO']['makespan'].mean()
        
        improvement = ((fifo_makespan - heft_makespan) / fifo_makespan) * 100
        
        summary.append({
            'workflow_size': workflow_size,
            'platform_scale': scale,
            'heft_makespan': round(heft_makespan, 2),
            'fifo_makespan': round(fifo_makespan, 2),
            'improvement_percent': round(improvement, 2),
            'heft_wins': len(subset[subset['scheduler'] == 'HEFT'])
        })

summary_df = pd.DataFrame(summary)

# 保存验证报告到实验目录
experiment_dir = "experiments/benchmark_validation"
os.makedirs(f"{experiment_dir}/results", exist_ok=True)
report_path = f"{experiment_dir}/results/validation_report.csv"
summary_df.to_csv(report_path, index=False)

# 打印结果
print("\n🎯 验证结果摘要:")
print("=" * 60)
print(summary_df.to_string(index=False))
print("=" * 60)

# 检查HEFT是否在所有情况下都优于FIFO
all_heft_wins = (summary_df['improvement_percent'] > 0).all()
if all_heft_wins:
    print("✅ 验证成功！HEFT在所有测试场景中都优于FIFO")
    print(f"📈 平均性能提升: {summary_df['improvement_percent'].mean():.2f}%")
else:
    print("❌ 验证失败！存在HEFT不如FIFO的场景")
    print("请检查实验配置或工作流参数")

# 保存验证状态
validation_status = {
    'heft_consistently_better': bool(all_heft_wins),
    'average_improvement': float(summary_df['improvement_percent'].mean()),
    'total_scenarios': len(summary_df),
    'successful_scenarios': len(summary_df[summary_df['improvement_percent'] > 0])
}

status_path = f"{experiment_dir}/results/validation_status.json"
with open(status_path, 'w') as f:
    json.dump(validation_status, f, indent=2)

print(f"\n📋 详细报告已保存: {report_path}")
print(f"📊 验证状态已保存: {status_path}")
EOF

# 完成提示
echo ""
echo "🎉 基准验证实验完成！"
echo "📁 结果目录: experiments/benchmark_validation/results/"
echo "📊 验证报告: experiments/benchmark_validation/results/validation_report.csv"
echo "🔍 检查验证状态: experiments/benchmark_validation/results/validation_status.json"

# 如果验证成功，提示下一步操作
if [ -f "experiments/benchmark_validation/results/validation_status.json" ]; then
    if grep -q '"heft_consistently_better": true' "experiments/benchmark_validation/results/validation_status.json"; then
        echo ""
        echo "🚀 验证成功！现在可以安全地继续第三步："
        echo "   1. 净化知识库（仅保留HEFT和WassHeuristicScheduler）"
        echo "   2. 在src/ai_schedulers.py中实现R_RAG动态奖励机制"
    else
        echo ""
        echo "⚠️  验证未通过！请检查实验配置或工作流参数"
        echo "   建议：调整CCR值或工作流规模后重新运行"
    fi
else
    echo ""
    echo "❌ 验证状态文件未生成，请检查实验过程"
fi