好的，我已经阅读并分析了您提供的所有代码文件。这是一个关于使用深度强化学习（DRL）和检索增强生成（RAG）技术来优化工作流调度的研究项目。

项目核心是通过一个名为WASS（Workflow Aware Scheduling System）的智能调度器，来决定如何将一系列计算任务（工作流）最有效地分配到不同的计算节点上，以缩短总体完成时间（Makespan）。

下面是我为您生成的 **README.md** 文件：

-----

# WASS-RAG: 基于检索增强的深度强化学习工作流调度系统

WASS-RAG 是一个旨在使用深度强化学习（DRL）和检索增强生成（RAG）技术优化科学工作流调度的研究项目。本项目的核心目标是开发一个智能调度代理（Agent），它能够学习并做出比传统调度算法（如 FIFO 和 HEFT）更优的决策，从而最小化工作流的总执行时间（Makespan）。

整个项目基于 [WRENCH](https://wrench-project.org/) 模拟框架构建，允许在可控的虚拟环境中生成、执行和评估复杂的工作流。

## 核心特性

  * **动态工作流生成**: 能够程序化地生成多种类型和规模的科学工作流，如 Montage（天文学）、LIGO（引力波探测）、CyberShake（地震模拟）等，以模拟真实世界的计算挑战。
  * **基于GNN的状态编码**: 使用图注意力网络（GATv2）将复杂的工作流依赖关系编码为向量表示，为强化学习智能体提供决策依据。
  * **PPO驱动的强化学习代理**: 采用近端策略优化（PPO）算法训练一个Actor-Critic模型，使其能够根据当前工作流的状态选择最佳的计算节点进行任务分配。
  * **RAG增强的决策奖励**: 创新性地引入了检索增强生成（RAG）机制。通过构建一个存储历史最优调度经验的“知识库”，智能体在训练时可以参考相似工作流的“专家决策”（如HEFT算法的决策），从而获得更精确和动态的奖励信号，加速学习过程并提升最终性能。
  * **全面的实验与评估**: 提供了一整套从生成数据、构建知识库、训练模型到最终进行多调度器性能对比的自动化脚本，并能生成详细的性能分析报告和图表。

## 项目是如何工作的？

整个项目的执行流程被设计为三个主要阶段，通过 `bash.sh` 或 `liucheng.md` 中的脚本顺序执行：

### 阶段一：知识库构建 (`scripts/seed_knowledge_base.py`)

这是为RAG机制准备“养料”的阶段。

1.  **生成工作流**: 首先，系统会根据 `configs/workflow_config.yaml` 中的定义，生成大量不同类型和规模的训练用工作流。
2.  **模拟与记录**: 接着，使用传统的优秀调度算法（如HEFT）和随机算法在 `configs/test_platform.xml` 所定义的模拟平台上执行这些工作流。
3.  **编码与存储**: 在模拟过程中，记录下每个决策（哪个任务分配给哪个主机）以及对应的性能。然后，使用GNN编码器将每个工作流的状态图转换为向量嵌入，连同其性能数据和决策记录一起存入FAISS向量索引构建的知识库中。

### 阶段二：智能体训练 (`train.py` 和 `train_no_rag.py`)

这个阶段的目标是训练出聪明的调度“大脑”。项目包含两个并行的训练脚本：

  * `train.py`: **训练带有RAG的WASS智能体**。在训练过程中，除了基于最终完成时间的全局奖励外，智能体在做出每个决策后，还会向知识库查询。**知识引导教师 (`KnowledgeableTeacher`)** 会比较智能体的决策与知识库中相似情况下的“专家决策”，并给予一个即时的、细粒度的奖励或惩罚。这使得智能体能更快地学会“好”的决策模式。
  * `train_no_rag.py`: **训练一个纯DRL的WASS智能体**。这个版本的智能体不使用知识库，仅依赖于最终工作流完成时间（Makespan）作为奖励信号进行学习，作为对比实验组。

### 阶段三：实验与评估 (`run_experiments.py`)

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
    直接运行 `bash.sh` 脚本即可按顺序完成知识库构建、模型训练和最终实验评估的全部流程。

    ```bash
    bash bash.sh
    ```

4.  **查看结果**:
    实验完成后，所有详细的性能数据和总结报告将保存在 `results/final_experiments/` 目录下。

## 最终目标

从 `liucheng.md` 和实验结果 可以看出，本项目的最终目标是证明：

> WASS-RAG 和 WASS-DRL 调度器的性能（以Makespan衡量）优于传统的 FIFO 调度器。

通过引入RAG，WASS-RAG旨在比纯DRL版本的调度器达到更高的性能和更快的收敛速度。

## 工作流来源与转换流程

本项目的训练与实验使用来自 [WFCommons](https://wfcommons.org/) 的真实科学工作流基准 (epigenomics, montage, seismology 等) 的 JSON 描述。为了让调度与图编码模块使用统一的字段 (runtime, flops, memory, dependencies)，我们提供了标准转换脚本：

1. 原始 WFCommons JSON 位于 `configs/wfcommons/*.json`。
2. 运行 `scripts/0_convert_wfcommons.py`：
  - 读取 execution.machines、execution.tasks、specification.files、specification.tasks。
  - 计算每个任务的 FLOPs: runtimeInSeconds * (avgCPU/100) * cpu_speed_MHz * 1e6。
  - 估算内存: sum(input file sizes) + sum(output file sizes) + 100MB 基础开销。
  - 写入 `task['runtime']` = execution.tasks.runtimeInSeconds，提供给 GNN 编码与 PPO 状态向量。
  - 保持原始结构并补充 `flops` / `memory` / `runtime` 字段，输出到 `data/workflows/*.json`。
3. 训练与推理脚本 (`scripts/2_train_rag_agent.py`, `scripts/3_train_drl_agent.py`, `scripts/4_run_experiments.py`) 直接从 `data/workflows` 读取已转换文件，不再生成内部合成工作流。
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

