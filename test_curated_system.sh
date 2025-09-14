#!/bin/bash
# 净化系统测试脚本

echo "🧪 开始净化系统测试..."

# 1. 检查文件存在
files=(
    "data/curated_kb_training_dataset.json"
    "data/curated_kb_metadata.json"
    "src/ai_schedulers.py"
)

for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file 存在"
    else
        echo "❌ $file 缺失"
        exit 1
    fi
done

# 2. 检查知识库内容
echo ""
echo "📊 知识库内容检查:"
python3 -c "
import json
with open('data/curated_kb_training_dataset.json') as f:
    data = json.load(f)
schedulers = {}
for sample in data:
    sched = sample.get('scheduler')
    schedulers[sched] = schedulers.get(sched, 0) + 1
print('调度器分布:', schedulers)
print('总样本数:', len(data))
print('知识库净化状态:', '成功' if set(schedulers.keys()).issubset({'HEFT', 'WassHeuristic'}) else '失败')
"

# 3. 检查R_RAG实现
echo ""
echo "🎯 R_RAG实现检查:"
if grep -q "teacher_makespan - student_makespan" src/ai_schedulers.py; then
    echo "✅ R_RAG差分奖励机制已实现"
else
    echo "❌ R_RAG差分奖励机制缺失"
fi

echo ""
echo "🎉 净化系统测试完成！"
echo ""
echo "🚀 下一步操作:"
echo "   python scripts/train_predictor_from_kb.py configs/experiment.yaml"
echo "   python experiments/wrench_real_experiment.py"
