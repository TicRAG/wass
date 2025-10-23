# 执行流程 (最新工作流管线)

推荐使用统一的转换 + 校验 + 训练 + 实验流水线。工作流来源于 `configs/wfcommons/*.json`，需首先转换后才能用于训练与评估。

## 完整步骤（手动）
1. 转换 WFCommons 工作流：
	```bash
	python scripts/0_convert_wfcommons.py --input_dir configs/wfcommons --output_dir data/workflows
	
	手动将转换后的 JSON 文件划分到:
	- data/workflows/training
	- data/workflows/experiment

	当前管线不再自动拆分，确保实验集合与训练集合互不重叠。

	```
2. 校验转换结果：
	```bash
	python scripts/validate_workflows.py --dir data/workflows
	```
3. 播种知识库（生成嵌入与经验）：
	```bash
	python scripts/1_seed_knowledge_base.py
	```
4. 训练含 RAG 的调度智能体：
	```bash
	python scripts/2_train_rag_agent.py
	```
5. 训练纯 DRL（无 RAG）智能体：
	```bash
	python scripts/3_train_drl_agent.py
	```
6. 运行最终对比实验（FIFO / HEFT / WASS_DRL / WASS_RAG）：
	（若需要对比两种智能体，确保脚本中取消注释 WASS_DRL 和 WASS_RAG 调度器）
	```bash
	python scripts/4_run_experiments.py
	```

## 一键执行
使用更新后的管线脚本：
```bash
bash run_pipeline.sh SKIP_CONVERT=1
```
支持的环境变量：
```bash
SKIP_CONVERT=1       # 跳过工作流转换
SKIP_TRAIN_RAG=1     # 跳过 RAG 训练
SKIP_TRAIN_DRL=1     # 跳过 DRL-only 训练
SKIP_EXPERIMENTS=1   # 跳过最终实验
CLEAN=1              # 清理旧的 results / models（保留已转换工作流）
```
示例：只重新跑最终实验：
```bash
SKIP_CONVERT=1 SKIP_TRAIN_RAG=1 SKIP_TRAIN_DRL=1 bash run_pipeline.sh
```

## 结果产物
训练模型：`models/saved_models/` 目录下保存 DRL / RAG 版本的权重（RAG 使用双 GNN：冻结检索编码器 + 可训练策略编码器）。
最终实验结果：`results/final_experiments/detailed_results.csv` 与 `summary_results.csv`。
已转换工作流：`data/workflows/*.json`（包含 flops / memory / runtime 字段）。

## 目标
验证 WASS_RAG 与 WASS_DRL 在 Makespan 上优于传统 FIFO（以及与 HEFT 进行对比）。

## 最新架构要点
* 双 GNN 编码器：避免训练导致的检索嵌入漂移。
* 教师奖励计算接收预计算状态嵌入，减少重复前向。
* 终止奖励模式统一：`reward_mode=final` 由 PPOTrainer 自动对单一终止奖励进行折扣展开。
* 合成工作流生成器已弃用（`src/workflows/generator.py` / `manager.py` 保留仅作历史参考）。

> 如果只看到 FIFO 与 HEFT，请编辑 `scripts/4_run_experiments.py` 取消注释 WASS_DRL / WASS_RAG 两行，以重新启用智能体调度器对比。
