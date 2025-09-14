# WASS-RAG 项目整改进度追踪

更新时间: 2025-09-14 (第二次更新)

## 1. 背景摘要
参见用户提供的整改报告问题概述：训练/测试环境脱节、知识库加载失败、奖励函数与目标不一致、代码结构混乱。

## 2. 阶段一（基础架构重构）进度

| 任务ID | 描述 | 当前状态 | 说明 |
|--------|------|----------|------|
| R-1.1 | 代码结构模块化 | 完成 | 模块划分 `src/{knowledge_base,drl,scheduling,environment}`；去除脚本内联类。 |
| R-1.2 | 数据交换格式标准化(JSON) | 完成 | `convert_kb_pickle_to_json.py` 转换 600 条案例；调度器 JSON→PKL→默认 回退链。 |
| R-1.3 | 统一仿真环境 | 进行中 | 仍使用 mock；`wrench_adapter.py` 占位，待接入真实 WRENCH。 |

## 3. 阶段二（学习与奖励一致性）进度

| 任务ID | 描述 | 当前状态 | 说明 |
|--------|------|----------|------|
| R-2.1 | 奖励函数重构 | 部分完成 | `docs/reward_design.md` + `src/drl/reward.py`；已接入训练（makespan 最终奖励 + shaping）。 |
| R-2.2 | 知识库质量过滤 | 完成 | 质量评分、top / filter API；导出 *_filtered.json。 |
| R-2.3 | DRL-RAG 决策融合 | 完成 | `hybrid_fusion.py` + 真实Q值融合 + 日志 `fusion_debug.log`。 |
| R-2.4 | 训练配置扩展 | 完成 | `drl.yaml` episodes=1000 + checkpoint/logging 字段；训练脚本适配。 |

## 4. 阶段三（再训练与验证）计划概览
- R-3.1 大规模再训练（≥1000 episodes，滚动窗口统计稳定）
- R-3.2 指标验证：makespan 分布、改进率、融合贡献对比（生成 charts 扩展）
- R-3.3 消融实验：仅DRL / 仅RAG / Fusion / 无质量过滤

## 5. 已完成的具体改动（汇总）
- JSON 知识库：`json_kb.py` + 转换脚本；统一加载回退链（JSON→PKL→默认）。
- 质量控制：评分、过滤、top 选择；生成过滤版数据文件。
- 奖励框架：`reward_design.md` + `reward.py`（step shaping + makespan final）；已集成 `improved_drl_trainer.py`。
- 决策融合：`hybrid_fusion.py` 与真实 Q 值获取；加权自适应融合 + 回退；调试日志。
- 训练扩展：`drl.yaml` 扩展 episodes/intervals/checkpoint/logging；训练器新增 checkpoint、best model、流式 JSONL 指标写入。
- 文档更新：本文件结构化阶段进度，新增阶段二/三部分。

## 6. 下一步计划（短期）
1. 填充 `wrench_adapter.py`：真实仿真 -> 状态提取 -> StepContext 精确化。
2. 增加 reward debug: 将 step shaping 分量逐条写入 `reward_debug.log`（用于消融分析）。
3. 启动 1000-episode 再训练（监控 rolling 平稳 & epsilon 收敛节奏）。
4. 训练后生成 makespan 分布与改进率趋势图（扩展 `charts/generate_charts.py`）。
5. 消融脚本：快速禁用融合 / 质量过滤 / shaping，产出对比表。

## 7. 风险与注意事项
- Mock 环境仍与真实 WRENCH 行为存在差距；当前 StepContext 若指标分布偏离，shaping 可能放大噪声。
- 融合权重自适应逻辑需在长训练中观察是否偏置单一信号；必要时加入温和正则。
- 指标日志体积 (JSONL) 随 episodes 增长；需定期归档或压缩。
- Checkpoint 频率 100 在 1000 episodes 下可接受；若 future 扩展 >10k episodes 需调大间隔。

## 8. 待决策点
## 9. 近期新增进展 (2025-09-14)
- DRL 再训练 1000 episodes 完成：`models/improved_wass_drl.pth` 生成；流式指标与 reward 调试日志写入正常。
- Checkpoint / best model 机制运作（若存在 `models/checkpoints/best_model.pth`）。
- 已准备对比实验运行脚本：`experiments/wrench_real_experiment.py` 支持 FIFO / HEFT / WASS-Heuristic / WASS-DRL / WASS-RAG。

## 10. 调度方法对比运行指南 (概要)
1. 训练模型（已完成）：`python scripts/improved_drl_trainer.py --config configs/experiment.yaml --episodes 1000 --output models/improved_wass_drl.pth`
2. 确保 RAG 知识库存在：`data/wrench_rag_knowledge_base.json` 或回退 `*.pkl`；否则调度器将生成默认案例。
3. 运行统一实验：`python experiments/wrench_real_experiment.py`（自动：统一随机工作流，多调度器同一批次比较，结果写入 `results/wrench_experiments/detailed_results.json`）。
4. 查看汇总分析：脚本末尾打印平均 / 标准差 / 最佳 makespan；最佳调度器一般预期为 WASS-RAG（若 DRL 模型 & KB 正常加载且融合生效）。
5. 生成自定义图表（待实现/扩展）：读取 `detailed_results.json` 聚合按 workflow_size 的平均 makespan，输出折线或箱型图。

bash scripts/run_wass_scheduler_benchmark.sh > experiment.log

## 11. 后续评估增强 (规划)
- 添加脚本：`scripts/analyze_scheduler_results.py`（聚合表 + 改进率： (baseline - wass_rag)/baseline ）。
- 可视化：在 `charts/generate_charts.py` 中增加 `plot_scheduler_comparison()`，输入 `detailed_results.json`。
- 消融：扩展 `WASSRAGScheduler` 参数（禁用融合 / 禁用 RAG / 仅 DRL）生成额外标签列。
- 是否重构 StepContext 以支持 GPU 利用率 / I/O 等扩展指标。
- 是否引入分段线性或 percentile 基线归一化替换静态 scaling。
- 知识库是否需要分层缓存（热区/冷区）加速相似度检索。
- 训练期间是否启用早停（基于 rolling makespan 改善阈值）。

---
(请在后续提交中继续更新本文件。)
