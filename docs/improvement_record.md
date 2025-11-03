# WASS-RAG 改进执行记录

## 文档说明
- 目的：跟踪 `WASS` 升级为 `WASS-RAG` 过程中各阶段任务的推进情况，确保实施细节与《改进计划》保持一致。
- 关联文档：`docs/improvement_plan.md`
- 更新节奏：默认每周更新一次；若关键里程碑完成需即时更新。

## 更新历史
| 版本 | 日期 | 作者 | 说明 |
| --- | --- | --- | --- |
| v0.14 | 2025-10-31 | TODO | 建立合成工作流生成器，规划差异化基准重跑实验 |
| v0.13 | 2025-10-30 | TODO | 生成 P1 实验图表与消融汇总，标记任务进度 |
| v0.12 | 2025-10-30 | TODO | README 增补 P1 实验 CLI 使用说明，状态回填 |
| v0.11 | 2025-10-30 | TODO | 启动 P1-1：实验运行器支持五策略 CLI，并完成单例 smoke test |
| v0.10 | 2025-10-30 | TODO | 修复 PBRS 状态编码、补充教师单元测试，任务 3 验收 |
| v0.9 | 2025-10-30 | TODO | 50-episode PBRS 长程训练验证奖励分布稳定，并新增 `Φ_t` 日志 |
| v0.8 | 2025-10-30 | TODO | RAG/PBRS 与禁用回退 smoke test 完成，记录奖励区间 |
| v0.7 | 2025-10-30 | TODO | 重新播种知识库并验证 `(s_key, q)` 统计与教师覆盖 |
| v0.6 | 2025-10-30 | TODO | 完成双编码器单元测试覆盖并标记任务 1 已完成 |
| v0.5 | 2025-10-30 | TODO | 训练脚本新增 `--disable_rag` 开关，支持回退至纯 DRL 流程 |
| v0.4 | 2025-10-30 | TODO | 知识库改造：按决策采集嵌入与剩余工期，新增势函数实现 |
| v0.4 | 2025-10-30 | TODO | 在训练调度器中接入 PBRS 势函数，更新奖励管线 |
| v0.3 | 2025-10-30 | TODO | 扩展播种脚本支持 MIN-MIN，并记录剩余完工时间 |
| v0.2 | 2025-10-30 | TODO | 完成双编码器架构初始实现，更新执行记录 |
| v0.1 | 2025-10-30 | TODO | 根据改进计划生成初始执行记录框架 |

---

## 阶段 P0：核心功能对齐（最高优先级）

### 任务 1：实现双 GNN 编码器架构
- 状态：`已完成`
- 负责人：待指派
- 涉及文件：`src/drl/gnn_encoder.py`
- 关键要求：
  - 建立 `policy_encoder`（可训练）与 `retrieval_encoder`（冻结）两个实例。
  - 确保 Actor/Critic 使用 `policy_encoder` 嵌入，检索相关逻辑统一使用冻结的 `retrieval_encoder`。
  - 训练启动时同步两者权重，此后冻结检索编码器参数。
- 验收标准：
  1. 编码器解耦逻辑通过单元测试或集成测试验证。
  2. 训练日志显示仅 `policy_encoder` 参与梯度更新。
  3. 检索模块可成功从冻结编码器生成键向量。
- 备注：2025-10-30 引入 `DecoupledGNNEncoder` 并在训练脚本中应用（`scripts/2_train_rag_agent.py`）；新增单元测试 `tests/test_decoupled_gnn_encoder.py` 覆盖同步/冻结/梯度分离逻辑并通过运行 `pytest tests/test_decoupled_gnn_encoder.py` 验证。
- 依赖：无。

### 任务 2：改造知识库（Scheduling Knowledge Base）
- 状态：`已完成`
- 负责人：待指派
- 涉及文件：`scripts/1_seed_knowledge_base.py`, `src/rag/teacher.py`, `data/knowledge_base/`
- 关键要求：
  - 定义键值结构：`s_key_i` 为冻结编码器输出，`q_i` 为归一化剩余完工时间。
  - 扩展脚本支持 HEFT、MIN-MIN、Random 三种教师策略批量填充知识库。
  - 每个状态均记录对应剩余完工时间并落盘。
- 验收标准：
  1. 知识库存储格式稳定（通过单元测试或样例验证）。
  2. 三种教师策略生成的数据量与质量满足后续检索需求。
  3. `workflow_metadata` 更新包含教师来源标签。
- 备注：2025-10-30 播种脚本按调度决策采集检索嵌入与剩余完工时间，知识库转为 `(s_key, q)` 键值结构（`scripts/1_seed_knowledge_base.py`, `src/simulation/schedulers.py`, `src/rag/teacher.py`）；同日运行 `python scripts/1_seed_knowledge_base.py` 生成 7350 条记录（3 种教师各 2450 条），验证 `workflow_metadata.csv` 中 `q_value∈[0.035,1]`。
- 依赖：任务 1。

### 任务 3：实现基于 RAG 的势函数奖励塑造
- 状态：`已完成`
- 负责人：待指派
- 涉及文件：`src/rag/teacher.py`, `src/drl/agent.py`
- 关键要求：
  - 在教师模块实现 `calculate_potential(state)`，完成检索、相似度计算与插值。
  - DRL 奖励变换：`r'_t = r_t + λ (γ Φ_{t+1} - Φ_t)`。
  - 支持配置 `k`、温度 `τ`、塑造强度 `λ` 等超参数。
- 验收标准：
  1. 单元测试覆盖关键数学分支（Softmax 权重、冻结编码器调用）。
  2. 训练脚本可切换是否启用 PBRS 并保持稳定运行。
  3. 记录并验证新奖励的数值范围与预期一致。
- 备注：2025-10-30 初版 `calculate_potential` 完成并由训练调度器消费，任务奖励改为 PBRS 形式（`src/rag/teacher.py`, `src/simulation/schedulers.py`），新增训练 CLI 开关回退至纯 DRL（`scripts/2_train_rag_agent.py`）；短程 smoke test（3 episodes）对比：启用 PBRS 时 `AvgRAG∈[-0.01,0.0156]`、`clamped≤12.6%`，禁用时奖励保持 0；长程 50-episode 运行 `python scripts/2_train_rag_agent.py --max_episodes 50 --profile` 观察摄动范围 `AvgRAG∈[-0.01,0.0202]`、`clamped≤15%`，梯度范数变化 Δ≤0.0108，PBRS 未出现爆炸或塌缩；2025-10-30 修复检索图保留任务状态（`src/simulation/schedulers.py`），运行 3-episode 复测观察 `Φ_t∈[0.059,1.0]`、`ΔΦ_min≈-0.366`；新增单元测试 `tests/test_teacher_potential.py` 覆盖相似度加权、调度器筛选与空邻域分支，任务验收完成。
- 依赖：任务 1、任务 2。

- **阶段里程碑**：完成 P0 后需产出可运行的 `WASS-RAG (Full)` 版本并通过最小规模实验自测。

---

## 验证与测试计划（持续更新）
- **知识库再生成**：2025-10-30 运行 `python scripts/1_seed_knowledge_base.py`，产出 7350 条 `(s_key, q)` 记录，`workflow_metadata.csv` 验证三种教师均衡覆盖并保留 `q_value` 合理区间；索引保存成功。
- **编码器梯度检查**：单元测试 `tests/test_decoupled_gnn_encoder.py` 已验证冻结检索分支无梯度；后续在完整训练日志中记录梯度范数以长期监控。
- **PBRS 数值监控**：2025-10-30 运行 `python scripts/2_train_rag_agent.py --max_episodes 3` 与 `--max_episodes 50 --profile`，观察 `AvgRAG` 在 `[-0.01,0.0202]` 区间、`clamped≤15%`，梯度范数变化 Δ≤0.0108；训练日志现已输出 `Φ_t`/`ΔΦ` 范围与样本，后续实验可直接对照监控势函数漂移。
- **RAG 关闭回退**：2025-10-30 运行 `python scripts/2_train_rag_agent.py --max_episodes 3 --disable_rag`，奖励回退为 0，确认 PBRS 完全停用。
- **集成回归**：在关键脚本（播种、训练）结束时添加 smoke test，确保调度器、教师、知识库之间的接口契合。

## 阶段 P1：实验与消融研究（高优先级）

### 任务 1：扩展实验运行器与基线
- 状态：`已完成`
- 负责人：待指派
- 涉及文件：`src/simulation/experiment_runner.py`, `scripts/4_run_experiments.py`
- 关键要求：
  - 支持五种策略：`WASS-RAG (Full)`, `WASS-DRL (Vanilla)`, `WASS-RAG (HEFT-only)`, `HEFT`, `MIN-MIN`。
  - 命令行参数覆盖策略、工作流、随机种子等配置。
  - 输出结构化结果（CSV/JSON）写入 `results/` 对应目录。
- 验收标准：
  1. 每种策略独立运行成功并生成有效结果。
  2. CLI 使用说明更新至 `README.md` 或脚本帮助信息。
  3. 结果文件结构与 `analysis` 模块约定一致。
- 依赖：阶段 P0 完成。
- 备注：2025-10-30 更新 `scripts/4_run_experiments.py` 引入 argparse CLI，支持策略筛选、工作流筛选、随机种子/重复次数与增广数据开关；将 `src/simulation/experiment_runner.py` 统一传递 `workflow_file`，确保 DRL 推理调度器可复原图状态。运行 `python scripts/4_run_experiments.py --strategies HEFT --seeds 0 --workflows epigenomics-chameleon-hep-1seq-100k-001` 等命令验证五种策略均可加载模型/调度器并生成 `results/final_experiments/*.csv`；随后执行 `source $HOME/venvs/wrench-env/bin/activate && python scripts/4_run_experiments.py --seeds 0 1 2 3 4` 完成三工作流 × 五策略 × 五种子主实验，产出 `summary_results.csv`（HEFT/MIN-MIN 平均 makespan≈10.86，WASS-RAG Full≈11.24，Vanilla≈11.38）；同日补充 README“最终实验 CLI”章节，记录环境激活与参数示例。下一步移动至绘图脚本与消融分析。

### 任务 2：执行主实验并绘图
- 状态：`进行中`
- 负责人：待指派
- 涉及文件：`scripts/4_run_experiments.py`, `analysis/plot_results.py`
- 关键要求：
  - 为每个（策略，工作流）组合至少运行 5 个随机种子。
  - 生成训练曲线、性能对比图、消融研究表。
  - 支持批量处理实验结果并输出 PDF/PNG。
- 验收标准：
  1. 图表与 `docs/keyimage.md` 示意一致。
  2. 原始数据与图表版本可追踪（存储命名规范统一）。
  3. 结果脚本可复现并附带运行说明。
- 依赖：P1 任务 1。
- 备注：2025-10-30 调整 `analysis/plot_results.py` 读取 `detailed_results.csv`，输出整体/按工作流的误差条图与箱线图，并生成 `results/final_experiments/ablation_summary.csv`；命令 `source $HOME/venvs/wrench-env/bin/activate && python analysis/plot_results.py` 成功产出 `charts/overall_makespan_bar.png` 等三张图。训练曲线仍待根据日志格式设计，后续需完善。2025-10-31 新增 `scripts/generate_synthetic_workflows.py` 可生成高并行度基准，后续步骤：1) 生成不少于 3 条并行度更高的工作流；2) 更新 `configs/workflow_config.yaml`/实验目录引用合成基准；3) 重跑 HEFT/MIN-MIN/DRL/RAG 比较确认 makespan 可区分；4) 用新结果刷新图表与消融表。

- **阶段里程碑**：提交完整对比实验数据及图表草稿，供论文撰写使用。

---

## 阶段 P2：深度分析与验证（中等优先级）

### 任务 1：可解释性案例分析
- 状态：`未开始`
- 负责人：待指派
- 涉及文件：`src/rag/teacher.py`, `analysis/interpretability_case_study.py`
- 关键要求：
  - 在 PBRS 计算中添加检索日志，包括近邻 `q_i` 与教师来源。
  - 新建脚本生成案例可视化（包含甘特图与注释）。
  - 支持从调度日志挑选关键决策点进行解读。
- 验收标准：
  1. 日志格式化输出便于后续分析。
  2. 至少完成一份可解释性案例图。
  3. 结果附带说明文档与生成脚本。
- 依赖：阶段 P0、P1 完成。

### 任务 2：验证双编码器有效性
- 状态：`未开始`
- 负责人：待指派
- 涉及文件：`analysis/embedding_drift_analysis.py`, 训练脚本
- 关键要求：
  - 运行“冻结/不冻结”两组对照实验。
  - 使用 t-SNE/UMAP 可视化嵌入漂移，输出图像与量化指标。
  - 记录实验配置，确保可复现。
- 验收标准：
  1. 图表展示漂移趋势差异明显。
  2. 文本分析总结关键发现并与论文论点对齐。
  3. 数据与图表归档到 `results/` 对应目录。
- 依赖：阶段 P0、P1 完成。

- **阶段里程碑**：形成深度分析章节初稿与支撑数据。

---

## 阶段 P3：清理与归档（低优先级）

### 任务 1：代码重构与文档
- 状态：`未开始`
- 负责人：待指派
- 涉及文件：核心模块源码、`README.md`
- 关键要求：
  - 清理实验脚本与工具函数，确保命名、日志、异常处理一致。
  - 为关键函数（如 `calculate_potential`）补充 Docstring 与类型注解。
  - 更新项目 README，加入复现实验指南。
- 验收标准：
  1. 静态检查通过，代码风格统一。
  2. 文档覆盖核心使用场景与注意事项。
  3. 团队评审通过。
- 依赖：阶段 P0-P2。

### 任务 2：最终结果归档
- 状态：`未开始`
- 负责人：待指派
- 涉及文件：`results/final_paper_experiments/`
- 关键要求：
  - 整理最终实验数据、图表、日志并统一归档。
  - 核查文件命名、目录结构与版本信息。
  - README 添加复现步骤与数据说明。
- 验收标准：
  1. 归档目录结构清晰，可直接用于论文附录或补充材料。
  2. README 复现流程实测可行。
  3. 资料通过团队审查。
- 依赖：阶段 P0-P2。

- **阶段里程碑**：提交完整归档包，完成项目收尾。

---

## 后续维护建议
- 每次任务状态更新后，同步修改“状态”“负责人”“备注”字段，并在“更新历史”追加记录。
- 若新增任务或调整优先级，应同步更新《改进计划》与本执行记录。
- 建议结合 Issue/看板工具实现细粒度追踪，本文件用于跨团队及管理层对齐。
