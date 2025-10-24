好的，我已经阅读并分析了您提供的所有代码文件。这是一个关于使用深度强化学习（DRL）和检索增强生成（RAG）技术来优化工作流调度的研究项目。

项目核心是通过一个名为WASS（Workflow Aware Scheduling System）的智能调度器，来决定如何将一系列计算任务（工作流）最有效地分配到不同的计算节点上，以缩短总体完成时间（Makespan）。

下面是我为您生成的 **README.md** 文件：

-----

# WASS-RAG: 基于 WFCommons 的检索增强深度强化学习工作流调度

WASS-RAG 通过深度强化学习 (DRL) + 检索增强 (RAG) 的组合提升科学工作流调度性能。当前版本以真实 WFCommons 工作流基准为唯一数据来源，旧的“动态合成工作流生成”功能已弃用 (Deprecated)，仅保留历史代码供参考，不再在任何训练或实验脚本中调用。

核心目标：训练一个智能调度 Agent，相比传统调度算法（FIFO、HEFT）减少整体完成时间 (Makespan)。

运行基础：基于 [WRENCH](https://wrench-project.org/) 模拟框架，在可控虚拟平台上加载转换后的 WFCommons 工作流并进行调度对比。

## 当前核心特性 (Active)

  * **WFCommons 驱动数据**：仅使用真实工作流 (epigenomics, montage, seismology 等)；所有任务属性通过统一转换脚本生成。
  * **图神经编码 (GNN)**：采用 GATv2 编码任务依赖结构，生成调度状态嵌入。
  * **PPO 策略优化**：Actor-Critic 结构，支持 dense / final 两种奖励模式。
  * **RAG 奖励教师**：基于 FAISS 的知识库检索相似已调度工作流案例，生成差异/比例改进奖励（支持 median/ratio/bonus/clamp 等配置）。
  * **双编码器稳定性**：冻结检索 GNN + 可训练策略 GNN，避免训练期间嵌入漂移破坏检索一致性。
  * **实验脚本一键链路**：`run_pipeline.sh` 串联转换→校验→知识库播种→RAG 训练→DRL 训练→终测对比。

## 已弃用功能 (Deprecated)

以下模块不再在主流程中使用，仅保留以供参考：

  * `src/workflows/generator.py`: 合成工作流生成器（pipeline/montage/ligo 等模式）。
  * `src/workflows/manager.py`: 旧的生成调度入口；当前仅使用其 `get_platform_file()`。

请勿使用上述模块进行知识库构建或训练；所有新实验必须基于 WFCommons 转换结果。后续版本将移除这些文件。 

## 项目是如何工作的？

整个项目的执行流程被设计为三个主要阶段，通过 `bash.sh` 或 `liucheng.md` 中的脚本顺序执行：

### 阶段一：知识库构建 (`scripts/1_seed_knowledge_base.py`)

这是为RAG机制准备“养料”的阶段。

1.  **放置工作流**: 先运行转换脚本并手动将转换后的文件划分至 `data/workflows/training` 与 `data/workflows/experiment` 目录（当前不自动拆分）。
2.  **模拟与记录**: 使用传统调度算法（如 HEFT）在指定平台上执行这些工作流并记录调度决策。
3.  **编码与存储**: 在模拟过程中，记录下每个决策（哪个任务分配给哪个主机）以及对应的性能。然后，使用GNN编码器将每个工作流的状态图转换为向量嵌入，连同其性能数据和决策记录一起存入FAISS向量索引构建的知识库中。

### 阶段二：智能体训练 (`scripts/2_train_rag_agent.py` 与 `scripts/3_train_drl_agent.py`)

这个阶段的目标是训练出聪明的调度“大脑”。项目包含两个并行的训练脚本：

  * `2_train_rag_agent.py`: 训练带有 RAG 的智能体。采用双 GNN 编码架构：一个冻结的 `rag_gnn_encoder` 用于稳定的检索嵌入，和一个可训练的 `policy_gnn_encoder` 用于策略梯度更新，避免训练过程中的表示漂移导致检索失效。教师模块接收预计算的状态嵌入，消除重复 GNN 前向。
  * `3_train_drl_agent.py`: 训练纯 DRL 智能体，不访问知识库。统一的 PPOTrainer 处理两种奖励模式：`dense`（逐步给予等分奖励并进行折扣）与 `final`（仅终止奖励，经 gamma 回溯折扣）。

### 阶段三：实验与评估 (`scripts/4_run_experiments.py`)

训练完成后，就到了检验成果的时候。

1.  **生成测试工作流**: 系统会生成一组全新的、从未在训练中见过的工作流用于最终测试。
2.  **性能大比拼**: `run_experiments.py` 脚本会组织一场竞赛，让以下几位“选手”在相同的平台上运行相同的测试工作流：
      * **FIFO**: 先进先出，一个基础的调度器。
      * **HEFT**: 异构最早完成时间，一个经典且性能优秀的启发式算法。
      * **WASS-DRL**: 只使用DRL训练的智能体。
      * **WASS-RAG**: 使用了RAG增强训练的智能体。
3.  **结果分析与可视化**: 所有实验完成后，脚本会自动收集每个调度器的Makespan，并生成详细的CSV结果文件（如 `detailed_results.csv` 和 `summary_results.csv`）和对比图表，以直观地展示WASS-RAG相对于其他算法的性能优势。

## 如何开始？

1.  **环境准备**:
    根据 `requirements.txt` 安装所有必要的Python依赖库。请注意，`torch-geometric` 等库可能对Python版本有要求。

2.  **配置文件**:

      * `configs/workflow_config.yaml`: 在这里定义你想要生成的工作流类型、规模和数量。
      * `configs/test_platform.xml`: 在这里定义你的模拟计算环境，包括主机的数量、计算速度、核心数和网络带宽等。

3.  **执行完整流程**:
  运行 `run_pipeline.sh` 脚本可串联转换（可选）、知识库播种、RAG 训练、DRL 训练与最终实验。工作流的训练/实验划分需手动完成后再调用训练脚本。

    ```bash
    bash bash.sh
    ```

    可选环境变量覆盖训练轮数（无需修改配置文件）：

    ```bash
    RAG_EPISODES=50 DRL_EPISODES=30 bash run_pipeline.sh
    ```

    说明：
    - RAG_EPISODES：`scripts/2_train_rag_agent.py` 使用的最大 episode 数 (`--max_episodes`)
    - DRL_EPISODES：`scripts/3_train_drl_agent.py` 使用的最大 episode 数
    - 若未设置，脚本内部使用各自配置文件中的默认 `total_episodes`。

4.  **查看结果**:
    实验完成后，所有详细的性能数据和总结报告将保存在 `results/final_experiments/` 目录下。

## 最终目标

从 `liucheng.md` 和实验结果 可以看出，本项目的最终目标是证明：

> WASS-RAG 和 WASS-DRL 调度器的性能（以Makespan衡量）优于传统的 FIFO 调度器。

通过引入RAG，WASS-RAG旨在比纯DRL版本的调度器达到更高的性能和更快的收敛速度。

## 工作流来源与转换流程

本项目的训练与实验使用来自 [WFCommons](https://wfcommons.org/) 的真实科学工作流基准 (epigenomics, montage, seismology 等) 的 JSON 描述。为了让调度与图编码模块使用统一的字段 (runtime, flops, memory, dependencies)，我们提供了标准转换脚本，并弃用了早期的合成工作流生成代码：

1. 原始 WFCommons JSON 位于 `configs/wfcommons/*.json`。
2. 运行 `scripts/0_convert_wfcommons.py`：
  - 读取 execution.machines、execution.tasks、specification.files、specification.tasks。
  - 计算每个任务的 FLOPs: runtimeInSeconds * (avgCPU/100) * cpu_speed_MHz * 1e6。
  - 估算内存: sum(input file sizes) + sum(output file sizes) + 100MB 基础开销。
  - 写入 `task['runtime']` = execution.tasks.runtimeInSeconds，提供给 GNN 编码与 PPO 状态向量。
  - 保持原始结构并补充 `flops` / `memory` / `runtime` 字段，输出到 `data/workflows/*.json`。
3. 训练与推理脚本 (`scripts/2_train_rag_agent.py`, `scripts/3_train_drl_agent.py`, `scripts/4_run_experiments.py`) 直接从 `data/workflows/training` 与 `data/workflows/experiment` 读取已转换文件；不再自动生成或拆分内部合成工作流。
4. 图数据构建 (`src/drl/utils.py`) 会自动检测：
  - 如果存在 `workflow['specification']['tasks']`，映射 `parents` 为 `dependencies`。
  - 从 `workflow.execution.tasks` 中补全缺失的 `runtime`。
5. 快速校验脚本 `scripts/validate_workflows.py` 可确保所有任务都具备非负的 `flops` 与 `memory`。

### 常用命令示例

```bash
# 1. 执行转换
python scripts/0_convert_wfcommons.py --input_dir configs/wfcommons --output_dir data/workflows

# 2. 校验转换结果
python scripts/validate_workflows.py --dir data/workflows

# 3. 开始训练 (示例：RAG 版本)
python scripts/2_train_rag_agent.py

# 3.a 覆盖训练轮数 (RAG / DRL)
RAG_EPISODES=20 DRL_EPISODES=10 bash run_pipeline.sh

# 4. 运行最终实验
python scripts/4_run_experiments.py
```

### 字段说明 (转换后任务)
| 字段 | 含义 |
|------|------|
| id | 任务唯一标识 |
| parents | 任务依赖的前置任务 (原始 wfcommons) |
| children | 自动推导的后继任务列表 |
| flops | 估算的总浮点运算次数 |
| memory | 估算的内存需求 (字节) |
| runtime | 预计运行时间 (秒, 来自 execution.tasks.runtimeInSeconds) |
| inputFiles/outputFiles | 输入与输出文件 ID |

若需调整 FLOPs 或内存估算策略，可修改 `scripts/0_convert_wfcommons.py` 中 `compute_flops` / `compute_memory` 与 `BASE_OVERHEAD`。

## 架构最新要点
1. 双 GNN 编码：RAG 编码器冻结，策略编码器训练，避免检索嵌入语义漂移。
2. 单次状态前向：调度器预计算嵌入并传递给教师，减少重复计算。
3. 统一稀疏奖励折扣：`reward_mode='final'` 时 PPOTrainer 根据步数对单一终止奖励进行 gamma 回溯分配。
4. 合成生成器弃用：`src/workflows/generator.py` 与 `manager.py` 保留仅作历史参考，标记 Deprecated。
5. 手动工作流划分：用户负责将转换结果分类至训练与实验目录，保证实验公平性。
 6. 梯度路径修复：旧的临时“Plan B”将已 detach 的嵌入写入缓冲，导致策略 GNN 不再更新。现已改为存储克隆的 PyG `Data` 图对象；PPO 第一训练 epoch 使用带梯度的重新嵌入，后续 epoch 对同一嵌入使用 `detach()` 避免二次反向传播错误。
 7. 推理检查点过滤：推理调度器加载时自动忽略 `gnn_encoder.*` 键，仅加载 `actor.*` 与 `critic.*` 参数；如需在推理阶段使用图重新嵌入，可后续扩展为加载与冻结编码器。

### 梯度修复要点
* 重放缓冲保存图快照（非纯嵌入张量），PPO 更新前动态 GNN 前向一次。
* 多 epoch PPO：第 1 次 epoch 保留完整计算图；其后使用 `detach()` 防止 “backward through the graph a second time” 异常。
* 训练脚本增加 GNN 参数范数前后差异日志 (`pre_norm` / `post_norm`) 验证实际更新。

## 后续改进建议
* 引入优势估计 (GAE) 提升策略梯度稳定性。
* 使用指数移动平均 (EMA) 维护一个慢速更新的检索编码器替代完全冻结策略。
* 运行时奖励 shaping：结合 makespan 预测残差作为辅助 dense signal。
* 在线更新知识库：周期性对新策略的高性能轨迹进行再编码与插入。

