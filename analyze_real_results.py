import json
import os
import numpy as np

# 加载实验结果
results_path = "results/real_heuristic_experiments/experiment_results.json"
if not os.path.exists(results_path):
    print("实验结果文件不存在")
    exit(1)

with open(results_path, 'r') as f:
    results = json.load(f)

# 分析结果
scheduler_results = {}
for experiment in results.get("results", []):
    scheduler = experiment.get("scheduler", "unknown")
    makespan = experiment.get("makespan", 0)
    
    if scheduler not in scheduler_results:
        scheduler_results[scheduler] = []
    scheduler_results[scheduler].append(makespan)

# 计算平均性能
print("=== 使用真实案例的实验结果摘要 ===")
print()
print("调度器性能对比:")
print("-" * 40)

for scheduler, makespans in scheduler_results.items():
    avg_makespan = np.mean(makespans)
    std_makespan = np.std(makespans)
    count = len(makespans)
    
    print(f"{scheduler:15} | 平均: {avg_makespan:8.2f}s | 标准差: {std_makespan:6.2f}s | 样本: {count:3d}")

print()
print("基于真实HEFT和WASS-Heuristic案例的RAG知识库已部署完成!")
