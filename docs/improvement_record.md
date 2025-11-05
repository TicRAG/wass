# WASS-RAG 改进执行记录

## 文档说明
- 目的：跟踪 `WASS` 升级为 `WASS-RAG` 过程中各阶段任务的推进情况，确保实施细节与《改进计划》保持一致。
- 关联文档：`docs/improvement_plan.md`
- 更新节奏：默认每周更新一次；若关键里程碑完成需即时更新。

## 更新历史
| 版本 | 日期 | 作者 | 说明 |
| --- | --- | --- | --- |
| v0.29 | 2025-11-05 | GitHub Copilot | 确认 WASS-RAG 慢主机偏好来源于知识库标签缺乏主机质量信号，制定“重构 q_value + 重新播种”即时任务 |
| v0.28 | 2025-11-04 | GitHub Copilot | 扩展 WASS-RAG 推断温度/贪婪/Top-K 控制并运行首批性能扫描 |
| v0.27 | 2025-11-04 | GitHub Copilot | 制定 WASS-RAG 性能优化与解释性串联执行计划 |
| v0.26 | 2025-11-04 | GitHub Copilot | 为 WASS-RAG 推断新增随机化 tie-breaking 选项并完成 montage 随机探针 |
| v0.25 | 2025-11-04 | GitHub Copilot | 批量解析 extreme_top3_noise01 trace，汇总摘要并生成甘特图 |
| v0.24 | 2025-11-04 | GitHub Copilot | 新增 trace 甘特图脚本并跑通首个案例可视化 |
| v0.23 | 2025-11-04 | GitHub Copilot | Trace 分析脚本支持 top-k 事件摘要并导出 CSV/JSON 汇总 |
| v0.22 | 2025-11-04 | GitHub Copilot | 新增解释性日志分析脚本并产出 10-episode 样例 trace |
| v0.21 | 2025-11-04 | GitHub Copilot | RAG 训练脚本开放 trace 日志开关，管线脚本传递 run_label 与 trace 路径 |
| v0.20 | 2025-11-04 | GitHub Copilot | 宣布阶段 P2 全面启动，准备可解释性与嵌入漂移分析数据基线 |
| v0.19 | 2025-11-03 | GitHub Copilot | 刷新 WASS DRL/RAG 模型至 6 主机动作空间，跑通完整管线并回填结果 |
| v0.18 | 2025-11-03 | GitHub Copilot | 更新管线默认主机筛选逻辑，移除 min-host-speed 依赖 |
| v0.17 | 2025-11-03 | GitHub Copilot | 启动阶段 P2：可解释性与双编码器漂移验证进入执行 |
| v0.16 | 2025-11-03 | GitHub Copilot | 完成 P1 主实验绘图与一键管线脚本，实现参数快照 |
| v0.15 | 2025-11-03 | GitHub Copilot | 添加一键实验自动化任务与参数保存计划 |
| v0.14 | 2025-10-31 | GitHub Copilot | 建立合成工作流生成器，规划差异化基准重跑实验 |
| v0.13 | 2025-10-30 | GitHub Copilot | 生成 P1 实验图表与消融汇总，标记任务进度 |
| v0.12 | 2025-10-30 | GitHub Copilot | README 增补 P1 实验 CLI 使用说明，状态回填 |
| v0.11 | 2025-10-30 | GitHub Copilot | 启动 P1-1：实验运行器支持五策略 CLI，并完成单例 smoke test |
| v0.10 | 2025-10-30 | GitHub Copilot | 修复 PBRS 状态编码、补充教师单元测试，任务 3 验收 |
| v0.9 | 2025-10-30 | GitHub Copilot | 50-episode PBRS 长程训练验证奖励分布稳定，并新增 `Φ_t` 日志 |
| v0.8 | 2025-10-30 | GitHub Copilot | RAG/PBRS 与禁用回退 smoke test 完成，记录奖励区间 |
| v0.7 | 2025-10-30 | GitHub Copilot | 重新播种知识库并验证 `(s_key, q)` 统计与教师覆盖 |
| v0.6 | 2025-10-30 | GitHub Copilot | 完成双编码器单元测试覆盖并标记任务 1 已完成 |
| v0.5 | 2025-10-30 | GitHub Copilot | 训练脚本新增 `--disable_rag` 开关，支持回退至纯 DRL 流程 |
| v0.4 | 2025-10-30 | GitHub Copilot | 知识库改造：按决策采集嵌入与剩余工期，新增势函数实现 |
| v0.4 | 2025-10-30 | GitHub Copilot | 在训练调度器中接入 PBRS 势函数，更新奖励管线 |
| v0.3 | 2025-10-30 | GitHub Copilot | 扩展播种脚本支持 MIN-MIN，并记录剩余完工时间 |
| v0.2 | 2025-10-30 | GitHub Copilot | 完成双编码器架构初始实现，更新执行记录 |
| v0.1 | 2025-10-30 | GitHub Copilot | 根据改进计划生成初始执行记录框架 |

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
- 状态：`已完成`
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
- 备注：2025-10-30 调整 `analysis/plot_results.py` 读取 `detailed_results.csv`，输出整体/按工作流的误差条图与箱线图，并生成 `results/final_experiments/ablation_summary.csv`；同日命令 `source $HOME/venvs/wrench-env/bin/activate && python analysis/plot_results.py` 成功产出 `charts/overall_makespan_bar.png` 等三张图。2025-10-31 新增 `scripts/generate_synthetic_workflows.py` 可生成高并行度基准。2025-11-03 使用 `scripts/run_full_pipeline.py --output-dir results/extreme_top3_noise01 --min-host-speed 100 --heft-noise-sigma 0.1 --seeds 0 1 2 3 4` 触发主实验，生成 `results/extreme_top3_noise01/{detailed_results.csv,summary_results.csv}` 并同步到 `results/final_experiments/`，成功刷新 `charts/*.png` 与 `results/final_experiments/ablation_summary.csv`，任务验收。

### 任务 3：一键式实验自动化脚本
- 状态：`已完成`
- 负责人：待指派
- 涉及文件：`scripts/run_full_pipeline.py`（新增）, `scripts/generate_synthetic_workflows.py`, `scripts/1_seed_knowledge_base.py`, `scripts/2_train_rag_agent.py`, `scripts/3_train_drl_agent.py`, `scripts/4_run_experiments.py`, `analysis/plot_results.py`
- 关键要求：
  - 实现单条命令触发“生成（或更新）工作流 → 播种知识库 → 训练 DRL/RAG → 执行对比实验 → 汇总表格与图表”完整流程。
  - 支持将关键参数（平台筛选、HEFT 噪声、MIN-MIN 权重、随机种子等）写入统一的配置文件或 `results/` 目录中的元数据记录，便于复现。
  - 运行结束时输出结果路径摘要，并校验各阶段日志/CSV/图表均已生成。
- 参考命令：
  ```
  /home/zhaotao/venvs/wrench-env/bin/python scripts/4_run_experiments.py \
      --strategies FIFO HEFT MINMIN WASS_RAG_FULL WASS_DRL_VANILLA \
      --rag-model models/saved_models/drl_agent.pth \
      --drl-model models/saved_models/drl_agent_no_rag.pth \
      --workflow-dir data/workflows/experiment \
      --output-dir results/extreme_top3_noise01 \
      --min-host-speed 100 \
      --heft-noise-sigma 0.1 \
      --seeds 0 1 2 3 4
  ```
- 验收标准：
  1. 在激活后的虚拟环境中运行脚本可顺利完成端到端流程并返回 0。
  2. `results/` 目录自动生成最新的详细/摘要 CSV、图表文件以及配置快照（JSON 或 YAML）。
  3. README 或文档补充脚本使用示例与参数说明。
- 依赖：P1 任务 1、任务 2。
- 备注：2025-11-03 新增 `scripts/run_full_pipeline.py` 串联五个阶段，默认执行“知识库播种→多策略实验→结果同步→图表输出”，支持 `--include-training`、`--skip-*` 等选项控制阶段开关。脚本自动在目标目录写入 `pipeline_config.json` 记录参数快照，并输出最终图表/摘要路径，任务验收完成。同日将 `--min-host-speed` 改为可选参数，默认不筛除慢速节点，以避免与预训练模型动作空间不匹配，若需实验过滤可手动指定阈值。

- **阶段里程碑**：提交完整对比实验数据及图表草稿，供论文撰写使用。

---

## 阶段 P2：深度分析与验证（中等优先级）

2025-11-04：阶段 P2 正式启动。已将最新 `results/extreme_top3_noise01` 作为分析基线，后续任务将围绕该批数据产出可解释性日志与嵌入对照实验。

### 任务 1：可解释性案例分析
- 状态：`进行中`
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
- 备注：2025-11-03 启动阶段 P2，计划在 `calculate_potential` 调用栈中插入检索明细（top-k 相似度、教师标签、q 值）并定义标准化日志格式；2025-11-04 将 `results/extreme_top3_noise01` 运行记录登记为案例分析输入，完成 `TeacherTraceLogger` 接线（训练脚本新增 `--trace_log_dir`，`run_full_pipeline.py` 自动传递 `run_label` 与日志目录），落地日志解析脚本 `analysis/interpretability_case_study.py`（含 top-k 摘要/导出）、可视化脚本 `analysis/plot_trace_gantt.py`，并分别运行 `python scripts/2_train_rag_agent.py --max_episodes 3 --trace_log_dir results/traces --run_label trace_smoke`、`python scripts/2_train_rag_agent.py --max_episodes 10 --trace_log_dir results/traces --run_label trace_long`、`python scripts/2_train_rag_agent.py --max_episodes 1 --trace_log_dir results/traces --run_label trace_regression` 产出样例（`results/traces/trace_smoke_trace_20251104T094635.jsonl`、`results/traces/trace_long_trace_20251104T101208.jsonl`、`results/traces/trace_regression_trace_20251104T103147.jsonl`；可视化示例：`charts/trace_regression_episode1.png`、`charts/trace_long_episode1_montage.png`、`charts/trace_long_episode1_montage_refined.png`）。
- 备注：2025-11-03 启动阶段 P2，计划在 `calculate_potential` 调用栈中插入检索明细（top-k 相似度、教师标签、q 值）并定义标准化日志格式；2025-11-04 将 `results/extreme_top3_noise01` 运行记录登记为案例分析输入，完成 `TeacherTraceLogger` 接线（训练脚本新增 `--trace_log_dir`，`run_full_pipeline.py` 自动传递 `run_label` 与日志目录），落地日志解析脚本 `analysis/interpretability_case_study.py`（含 top-k 摘要/导出）、可视化脚本 `analysis/plot_trace_gantt.py`，并分别运行 `python scripts/2_train_rag_agent.py --max_episodes 3 --trace_log_dir results/traces --run_label trace_smoke`、`python scripts/2_train_rag_agent.py --max_episodes 10 --trace_log_dir results/traces --run_label trace_long`、`python scripts/2_train_rag_agent.py --max_episodes 1 --trace_log_dir results/traces --run_label trace_regression` 产出样例（`results/traces/trace_smoke_trace_20251104T094635.jsonl`、`results/traces/trace_long_trace_20251104T101208.jsonl`、`results/traces/trace_regression_trace_20251104T103147.jsonl`；可视化示例：`charts/trace_regression_episode1.png`、`charts/trace_long_episode1_montage.png`、`charts/trace_long_episode1_montage_refined.png`）。2025-11-04 批量运行 `scripts/4_run_experiments.py --strategies WASS_RAG_FULL --trace-log-dir results/traces/extreme_top3_noise01` 收集 8 workflow × 5 seeds JSONL，统一落地至 `results/traces/extreme_top3_noise01_summary/`（含 `aggregate_metrics.csv`、`workflow_summary.csv`），并生成首批基线甘特图 `charts/trace_extreme_top3/{epigenomics_seed0,montage_seed0,seismology_seed0}.png`。下一步提炼跨 seed 指标差异，撰写案例解读草稿并挑选对比图收录至文档。
 - 备注：2025-11-03 启动阶段 P2，计划在 `calculate_potential` 调用栈中插入检索明细（top-k 相似度、教师标签、q 值）并定义标准化日志格式；2025-11-04 将 `results/extreme_top3_noise01` 运行记录登记为案例分析输入，完成 `TeacherTraceLogger` 接线（训练脚本新增 `--trace_log_dir`，`run_full_pipeline.py` 自动传递 `run_label` 与日志目录），落地日志解析脚本 `analysis/interpretability_case_study.py`（含 top-k 摘要/导出）、可视化脚本 `analysis/plot_trace_gantt.py`，并分别运行 `python scripts/2_train_rag_agent.py --max_episodes 3 --trace_log_dir results/traces --run_label trace_smoke`、`python scripts/2_train_rag_agent.py --max_episodes 10 --trace_log_dir results/traces --run_label trace_long`、`python scripts/2_train_rag_agent.py --max_episodes 1 --trace_log_dir results/traces --run_label trace_regression` 产出样例（`results/traces/trace_smoke_trace_20251104T094635.jsonl`、`results/traces/trace_long_trace_20251104T101208.jsonl`、`results/traces/trace_regression_trace_20251104T103147.jsonl`；可视化示例：`charts/trace_regression_episode1.png`、`charts/trace_long_episode1_montage.png`、`charts/trace_long_episode1_montage_refined.png`）。2025-11-04 批量运行 `scripts/4_run_experiments.py --strategies WASS_RAG_FULL --trace-log-dir results/traces/extreme_top3_noise01` 收集 8 workflow × 5 seeds JSONL，统一落地至 `results/traces/extreme_top3_noise01_summary/`（含 `aggregate_metrics.csv`、`workflow_summary.csv`），并生成首批基线甘特图 `charts/trace_extreme_top3/{epigenomics_seed0,montage_seed0,seismology_seed0}.png`；同日整理《Trace Case Study Notes》（`docs/interpretability_case_notes.md`）概括跨 workflow 奖励分布，选定 montage/epigenomics/seismology 作为主案例，并建议保留 synthetic 作为附录基线。下一步撰写案例解读草稿并从邻域明细抽取注释。
 - 备注：2025-11-03 启动阶段 P2，计划在 `calculate_potential` 调用栈中插入检索明细（top-k 相似度、教师标签、q 值）并定义标准化日志格式；2025-11-04 将 `results/extreme_top3_noise01` 运行记录登记为案例分析输入，完成 `TeacherTraceLogger` 接线（训练脚本新增 `--trace_log_dir`，`run_full_pipeline.py` 自动传递 `run_label` 与日志目录），落地日志解析脚本 `analysis/interpretability_case_study.py`（含 top-k 摘要/导出）、可视化脚本 `analysis/plot_trace_gantt.py`，并分别运行 `python scripts/2_train_rag_agent.py --max_episodes 3 --trace_log_dir results/traces --run_label trace_smoke`、`python scripts/2_train_rag_agent.py --max_episodes 10 --trace_log_dir results/traces --run_label trace_long`、`python scripts/2_train_rag_agent.py --max_episodes 1 --trace_log_dir results/traces --run_label trace_regression` 产出样例（`results/traces/trace_smoke_trace_20251104T094635.jsonl`、`results/traces/trace_long_trace_20251104T101208.jsonl`、`results/traces/trace_regression_trace_20251104T103147.jsonl`；可视化示例：`charts/trace_regression_episode1.png`、`charts/trace_long_episode1_montage.png`、`charts/trace_long_episode1_montage_refined.png`）。2025-11-04 批量运行 `scripts/4_run_experiments.py --strategies WASS_RAG_FULL --trace-log-dir results/traces/extreme_top3_noise01` 收集 8 workflow × 5 seeds JSONL，统一落地至 `results/traces/extreme_top3_noise01_summary/`（含 `aggregate_metrics.csv`、`workflow_summary.csv`），并生成首批基线甘特图 `charts/trace_extreme_top3/{epigenomics_seed0,montage_seed0,seismology_seed0}.png`；同日整理《Trace Case Study Notes》（`docs/interpretability_case_notes.md`）概括跨 workflow 奖励分布，选定 montage/epigenomics/seismology 作为主案例，并建议保留 synthetic 作为附录基线。2025-11-04 新增 `--stochastic-tie-break` 开关并运行 montage 随机化探针（输出 `results/traces/extreme_top3_noise01_stochastic_summary/`、`results/stochastic_montage/`），对比展示随机打破 host 平局后的奖励跨度与主机分布。下一步撰写案例解读草稿并从邻域明细抽取注释；同时按性能优先级安排 WASS-RAG 参数调优（蒙太奇/地震工作流上扫描检索温度、动作贪婪阈值、批处理大小），锁定明显领先 HEFT 的配置后再整合解释性叙述与邻域注释。

### 任务 2：验证双编码器有效性
- 状态：`进行中`
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
- 备注：2025-11-03 启动阶段 P2，对照实验准备进入执行：计划先基于 `scripts/run_full_pipeline.py --include-training` 生成“冻结/解冻”两套模型快照，再撰写 `analysis/embedding_drift_analysis.py` 调用 UMAP+Matplotlib 对比嵌入；确认所需训练日志已在 `pipeline_config.json` 记录以便复现实验。
- 备注：2025-11-03 启动阶段 P2，对照实验准备进入执行：计划先基于 `scripts/run_full_pipeline.py --include-training` 生成“冻结/解冻”两套模型快照，再撰写 `analysis/embedding_drift_analysis.py` 调用 UMAP+Matplotlib 对比嵌入；确认所需训练日志已在 `pipeline_config.json` 记录以便复现实验。2025-11-04 已整理 6 主机新版模型权重作为对照实验基线。

### 任务 3：知识库奖励信号重构（新增）
- 状态：`待验收`
- 负责人：待指派
- 涉及文件：`scripts/1_seed_knowledge_base.py`, `src/simulation/schedulers.py`, `data/knowledge_base/*`
- 关键要求：
  - 将知识库 `q_value` 从“剩余工期比例”改为显式反映主机算力/执行效率的指标。
  - 播种阶段补充主机速度、任务 FLOPs 等元信息，保证教师在推断期能区分快慢节点。
  - 重新运行播种脚本生成新版 KB，并输出前后统计对比（host 采样次数、q 值统计）。
- 验收标准：
  1. `workflow_metadata.csv` 中各主机的 `q_value` 分布与速度正相关（快节点均值更高）。
  2. 教师检索日志包含新的主机质量信号（如 `host_speed`）。
  3. 播种脚本与调度器改动通过 lint/单元测试，运行 `python scripts/1_seed_knowledge_base.py` 成功落盘。
- 依赖：阶段 P0 完成。
- 备注：2025-11-05 基于推断/训练 trace 差异定位慢主机偏好，决定立即执行“重构 q_value + 重播种”两步动作，并在 `results/tmp/instrumentation_trace` 验证原始偏差。首要步骤：扩展 `KnowledgeRecordingMixin` 记录 `host_speed`, `task_flops`，修改播种脚本按主机速度归一化写入 `q_value`，随后重新播种生成新版知识库。2025-11-05 运行 `python scripts/1_seed_knowledge_base.py` 重新播种，生成 84,717 条记录并将 `data/knowledge_base/workflow_metadata.csv` 主机均值调整为 `ultra=0.9151 > fast=0.5167 > balanced=0.2976 > slow=0.1225 > bottleneck=0.0728 > micro=0.0442`；教师记录现包含 `host_speed/task_flops/compute_duration`。同日执行 `python scripts/2_train_rag_agent.py --max_episodes 60 --include_aug --reward_mode dense --run_label host_q_refresh` 重新训练策略，并用 `python scripts/4_run_experiments.py --strategies WASS_RAG_FULL --workflows montage-chameleon-2mass-01d-001.json --seeds 0 --trace-log-dir results/tmp/inference_traces --trace-run-label host_q_refresh --stochastic-tie-break` 采集推断日志。最新 trace (`results/tmp/inference_traces/montage-chameleon-2mass-01d-001_seed0_rep0_20251105T033520.jsonl`) 中主机选择频次 `ultra:22 / fast:19 / balanced:20 / micro:20 / slow:13 / bottleneck:9`，慢主机不再占主导；待更多工作流与种子验证通过后可正式验收。

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

## 近期行动计划（2025-11）
- **性能优化冲刺**：针对 montage、epigenomics、seismology 工作流批量扫描检索温度、贪婪阈值、批处理大小与随机种子，锁定显著优于 HEFT 的 WASS-RAG 组合，并在 `results/stochastic_montage/` 基础上追加 makespan/奖励对照回归。
- **解释性整合**：在确定领先配置后，回填对应 trace 至 `results/traces/extreme_top3_noise01`，从 `analysis/interpretability_case_study.py` 导出跨种子摘要，并在 `docs/interpretability_case_notes.md` 撰写案例解读草稿与邻域注释。
- **嵌入漂移对照**：依据优化结果同步刷新冻结/解冻模型快照，运行 `analysis/embedding_drift_analysis.py` 生成 UMAP/t-SNE 图与量化指标，验证双编码器收益。
- **文档与验收**：更新 `README.md` 与《改进计划》记录新参数、性能差异和复现步骤，确认 Task 1、Task 2 验收条件对齐最新实验。

### 数据扩充计划（2025-11-04）
- **负责人**：待指派
- **目标**：扩充训练工作流与知识库数据量，为重新训练 WASS-RAG 提供更丰富样本。
- **子任务**：
  1. 运行 `scripts/augment_workflows.py` 为每个 WFCommons 基础工作流生成 ≥10 个扰动变体，写入 `data/workflows/training_aug/`。
  2. 运行 `scripts/generate_synthetic_workflows.py` 追加 ≥20 个高并行度/高噪声合成工作流至 `data/workflows/training_aug/`。
  3. 重新执行 `scripts/1_seed_knowledge_base.py`，将扩充后的训练集写入知识库（校验 `workflow_metadata.csv` 行数 ≥ 2× 当前规模）。
  4. 使用 `scripts/2_train_rag_agent.py --include_aug` 启动新一轮训练，保存模型至 `models/saved_models/` 并记录日志至 `results/training_runs/`。
 - **执行记录（2025-11-04）**：
   - 运行 `python scripts/augment_workflows.py` 生成 150 个扰动变体，输出至 `data/workflows/training_aug/`（共 185 个训练工作流）。
   - 运行 `python scripts/generate_synthetic_workflows.py --count 20` 追加 20 个合成工作流。
   - 运行 `python scripts/1_seed_knowledge_base.py` 重新播种知识库，`workflow_metadata.csv` 行数增至 84,717（覆盖扩充语料）。
   - 运行 `python scripts/2_train_rag_agent.py --include_aug --max_episodes 400 --log_dir results/training_runs/retrain_20251104 --run_label retrain_v1` 完成本轮训练，模型已保存至 `models/saved_models/drl_agent.pth`。
  - **当前状态**：扩充与再训练已完成，但最新基准中 WASS-RAG 仍落后 HEFT（详见 `results/perf_sweep_extreme_top3/`），需继续执行性能调优与策略扫描以达到优势。
- **验收指标**：
  - `data/workflows/training_aug/` 含 ≥45 个新 JSON。
  - 知识库记录数 ≥15k，覆盖新增工作流。
  - 新训练日志与模型权重成功生成，makespan/奖励曲线稳定。

### 近期重点行动拆解
1. **优化试验矩阵执行（负责人待定，状态：`进行中`）**
    - 批量运行 `scripts/4_run_experiments.py`，系统化扫描检索温度、贪婪阈值、批处理大小与随机种子组合。
    - 将结果写入 `results/perf_sweep_extreme_top3/`，并与 HEFT 基线对比 makespan、奖励分布及主机利用率。
    - 选定领先配置后，回填摘要至 `results/final_experiments/summary_results.csv` 并更新 `charts/`。
  - 2025-11-04：已启动首轮组合（温度∈{0.5,0.7,0.9}×贪婪阈值∈{0.85,0.9}×批处理大小∈{64,96}×种子∈{0,1,2}），运行日志写入 `results/perf_sweep_extreme_top3/runlog_20251104T` 前缀。
  - 当前执行命令示例：`python scripts/4_run_experiments.py --strategies WASS_RAG_FULL HEFT --workflows data/workflows/experiment/extreme_top3_noise01.json --seeds 0 1 2 --rag-temperature 0.7 --rag-greedy-threshold 0.9 --rag-sample-topk 3 --rag-epsilon 0.05 --stochastic-tie-break --output-dir results/perf_sweep_extreme_top3/temp0p7_top3_seed012`
  - 2025-11-04：完成基线批次（WASS-RAG Full 随机平局 vs HEFT，工作流=epigenomics/montage/seismology，种子=0-2），输出至 `results/perf_sweep_extreme_top3/baseline_seed012/`，平均 makespan 显示 HEFT (22.64) 领先于 WASS-RAG (59.54)，需通过超参扫描继续缩小差距。
  - 2025-11-04：新增 CLI 参数与调度器逻辑（温度、贪婪阈值、Top-K、epsilon），分别运行 `temp0p7_top3_seed012`、`temp0p9_top2_seed012` 组合；目前 WASS-RAG makespan 分别为 75.19/84.25，仍显著落后 HEFT，后续需调整参数网格与训练快照。
2. **解释性案例跟进（负责人待定，状态：`待开始`）**
    - 以领先配置重新生成 trace（保持原始与随机平局两套），整理至 `results/traces/extreme_top3_noise01` 与 `_stochastic_summary/`。
    - 使用 `analysis/interpretability_case_study.py` 与 `analysis/plot_trace_gantt.py` 导出跨种子摘要及标注图，在 `docs/interpretability_case_notes.md` 完成案例解读初稿。
    - 收敛后将关键邻域注释与图表纳入《改进计划》与论文草稿占位。
3. **嵌入漂移对照实验（负责人待定，状态：`待开始`）**
    - 激活 `scripts/run_full_pipeline.py --include-training` 生成冻结/解冻模型快照，存入 `models/saved_models/embedding_drift/{frozen,unfrozen}/`。
    - 实现 `analysis/embedding_drift_analysis.py` 可复现流程，输出 UMAP/t-SNE 图与漂移指标表至 `results/embedding_drift/`。
    - 汇总发现并在 Task 2 备注中记录与性能改进的因果关联。

### 扩充计划（2025-11-04）
- **目标**：扩大训练语料与知识库覆盖面，重新训练 WASS-RAG 以改善 HEFT 对比表现。
- **步骤拆解**：
  1. 运行 `augment_workflows.py` 生成 10 个扰动版本，输出至 `data/workflows/training_aug/`。
  2. 运行 `generate_synthetic_workflows.py` 追加 20 个高并行结构，输出至同目录。
  3. 重新执行 `scripts/1_seed_knowledge_base.py`，用扩充后的 workflow 集重建 KB。
  4. 执行 `scripts/2_train_rag_agent.py --include_aug --max_episodes 400` 完成新一轮训练，记录日志至 `results/training_runs/retrain_20251104/`。
  5. 完成后刷新性能 sweep 与解释性任务输入。

---

## 后续维护建议
- 每次任务状态更新后，同步修改“状态”“负责人”“备注”字段，并在“更新历史”追加记录。
- 若新增任务或调整优先级，应同步更新《改进计划》与本执行记录。
- 建议结合 Issue/看板工具实现细粒度追踪，本文件用于跨团队及管理层对齐。
