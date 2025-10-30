WASS-RAG：一种用于工作流感知调度(Workflow-Aware
Scheduling)的检索增强(Retrieval-Augmented)DRL框架

赵涛

联想（天津）有限公司，zhaotao1@lenovo.com

吴众欣

联想（天津）有限公司，wuzx9@lenovo.com

盛家杰

上海交通大学，jiajie.sheng@sjtu.edu.cn

于洋

联想（北京）有限公司，yuyang5@lenovo.com

韦建文 \*

[^1]weijianwen@sjtu.edu.cn

科学计算过程日益依赖于复杂的工作流。因此，高效的调度对于整体系统性能变得至关重要。尽管基于深度强化学习（DRL）的调度器显示出巨大潜力，但它们通常存在两个严重限制其SBEP署的关键瓶颈：(1)
奖励稀疏性（Reward
Sparsity），即智能体在数千个决策后只能获得一个稀疏的最终奖励，导致信用分配困难；以及
(2) 基线无知（Baseline
Ignorance），即智能体无法判断其获得的绝对奖励对于特定工作流而言是好是坏。这些挑战导致了极低的样本效率和不稳定的收敛性
。为了应对这些挑战，我们提出了
WASS-RAG，一个引入了基于检索的、剧集式（Episodic）奖励塑形的新型调度框架。WASS-RAG的核心创新在于，它允许DRL智能体（学生）首先独立完成一个完整的调度剧集以获得其真实的"智能体总完工时间"。随后，一个"知识引导教师"模块被激活，该模块利用检索增强生成（RAG）技术，从一个由"多教师委员会"（包含多种启发式算法）构建的多样化知识库中，查询并检索出拓扑相似案例的"历史最优总耗时"。这个检索到的"历史最优值"被用作一个动态的、上下文感知的奖励基线。智能体收到的最终奖励被"塑形"为其性能相对于该基线的相对提升值。这种机制将稀疏的奖励信号转化为强烈的相对梯度，极大地加速了DRL智能体的收敛。在高保真仿真环境中的实验表明，与基线调度器相比，WASS-RAG能将工作流完工时间最多减少
（TODO）%。重要的是，RAG组件的引入（通过提供动态基线）相较于非RAG的DRL方法带来了额外的（TODO）%
性能提升
。因此，WASS-RAG为基于AI的调度提供了一条高效、鲁棒且性能卓越的技术路径。

**CCS概念** • 计算方法论 → 人工智能 → 规划与调度 → 调度； • 计算方法论 →
机器学习 → 强化学习； • 计算方法论 → 机器学习 → 神经网络； • 计算理论 →
离散数学 → 图论。

**附加关键词与短语：**工作流调度、深度强化学习、检索增强生成、知识库、Slurm、可解释人工智能

# 引言 {#引言 .Head1}

现代科学计算，从天体物理学 \[1\]、地震模拟 \[2\] 到基因组学
\[3\]，日益依赖于在高性能计算（HPC）集群上执行的复杂科学工作流（Scientific
Workflows）。这些工作流通常被建模为有向无环图（DAGs），其中节点代表计算或I/O任务，边代表它们之间的依赖关系。随着数据量和计算规模的爆炸式增长，高效调度这些工作流以最小化完工时间（Makespan）和最大化资源利用率，已成为HPC系统性能的关键瓶颈。

传统的HPC调度器，如广泛部署的Slurm
\[4\]，通常采用基于队列和先进先出（FCFS）的策略。这种"以作业为中心"的范式忽视了工作流内部复杂的任务依赖性，导致资源利用率低下。另一方面，专为DAG设计的启发式算法，如HEFT（Heterogeneous
Earliest Finish
Time）\[5\]，虽然在静态环境中表现出色，但它们是"流程感知，系统无感知"的
\[6\],
\[7\]。它们无法动态适应HPC集群中不断变化的负载和资源可用性，导致调度决策次优。

为了应对这种动态性，研究人员已转向基于深度强化学习（DRL）的调度器 \[8\],
\[9\]。这些方法通常使用图神经网络（GNN）\[10\]
来编码工作流DAG和系统状态，并通过DRL学习端到端的调度策略
\[11\]。尽管展现了巨大潜力，但DRL调度器在实际应用中面临两大挑战：

1.  **样本效率低：** 它们需要海量的模拟训练才能收敛，即"冷启动"问题。

2.  **可解释性差：** 其"黑盒"决策过程难以在关键的生产环境中获得信任
    \[12\]。为解决上述挑战，本文提出了一种名为**WASS-RAG**（Workflow-Aware
    Scheduling for Slurm using Retrieval-Augmented
    Generation）的**算法框架**。我们的核心创新在于，首次将自然语言处理（NLP）领域的检索增强生成（RAG）范式
    \[13\] 创造性地应用于HPC工作流调度这一复杂的组合优化问题。

WASS-RAG框架的核心是一个"知识引导教师"（Knowledge-Guided
Teacher）模块。该模块维护一个存储了高质量历史调度经验的知识库。在DRL训练期间，该模块通过RAG检索与当前状态最相似的历史经验，并利用**基于势函数（Potential-Based）的奖励塑造**
\[23\]
来提供有理论保障的指导。这种机制不仅显著加速了智能体的收敛速度（解决样本效率问题），也为调度决策提供了可追溯的"历史依据"（解决可解释性问题）。

本文的主要贡献总结如下：

1.  提出了一种将RAG范式 \[13\]
    应用于HPC工作流调度的创新方法，通过检索历史经验来增强DRL，解决了DRL的样本效率和可解释性问题。

```{=html}
<!-- -->
```
2.  设计并实现了一个"知识引导教师"模块，该模块利用**基于RAG的势函数**
    \[23\]（RAG-based Potential Function）显著加速了DRL智能体的收敛。

3.  通过在高保真Wrench模拟器 \[14\] 中对真实科学工作流（如Montage \[3\],
    LIGO \[1\], CyberShake
    \[2\]）进行的严格实证评估，**证明**了WASS-RAG相比HEFT \[5\]
    和纯DRL基线 \[11\] 在完工时间上的显著优势。

# 相关工作 {#相关工作 .Head1}

本节旨在为我们的研究提供上下文，重点关注三个关键领域：传统HPC工作流调度、基于AI的调度方法的演进，以及定位本文所解决的"部署鸿沟"。

**A. 传统工作流调度（Traditional Workflow Scheduling）**

HPC调度是一个经典的NP难组合优化问题
\[15\]。传统的解决方案主要分为两大类：基于队列的系统和基于DAG的启发式算法。

1.  **基于队列的调度器：** 以Slurm
    \[4\]、PBS/Torque为代表的生产调度器，主要采用FCFS（先进先出）或辅以某种形式的优先级（如公平共享）的策略。这些调度器是"以作业为中心"的，它们将整个工作流（可能包含数千个任务）视为一个单一的作业提交脚本。这种范式无法感知工作流内部复杂的DAG结构，导致关键路径上的任务（Critical
    Path
    Tasks）可能因为非关键任务占用了资源而被迫等待，从而造成严重的资源闲置和"队头阻塞"（Head-of-Line
    Blocking）\[7\]。

2.  **DAG感知的启发式算法：**
    为了解决上述问题，学术界提出了多种"流程感知"（Workflow-Aware）的启发式算法。其中，HEFT（Heterogeneous
    Earliest Finish Time）\[5\]
    已被公认为最有效和最广泛使用的基准算法之一。HEFT通过两个阶段工作：首先，它根据任务的平均计算和通信成本，自底向上计算每个任务的"向上排名"（Upward
    Rank）；然后，它按照排名（即关键路径）的降序，将任务贪婪地调度到能提供最早完成时间（EFT）的处理器上。

尽管HEFT在静态和确定性环境（即所有任务成本和数据传输时间均已知）中表现优异
\[5\]，但它有两大局限性：

-   **静态性：**
    它假设计算和网络资源是同构且可预测的，无法适应生产HPC集群中由其他用户作业引起的动态负载变化
    \[6\]。

-   **系统无感知：**
    它的决策模型没有考虑HPC系统的全局状态，例如存储I/O瓶颈或网络拓扑
    \[7\]。

**B. 用于调度的深度强化学习与GNN（DRL and GNN for Scheduling）**

为了克服启发式算法的局限性，研究人员开始探索使用DRL来学习动态调度策略。DRL将调度问题建模为一个马尔可夫决策过程（MDP），其中智能体（Agent）在每个时间步观察系统状态（State），执行一个动作（Action，即调度一个任务），并接收一个奖励（Reward，例如最小化完工时间）\[8\]。

在HPC调度中，一个核心挑战是如何有效地表示复杂的状态（即工作流DAG和集群资源图）。图神经网络（GNN）\[10\]
已成为解决此问题的SOTA方法。GNN能够通过消息传递机制，将节点特征和图的拓扑结构编码为低维嵌入向量（Embedding）\[11\]，\[16\]。

因此，将GNN与DRL结合（GNN-RL）已成为组合优化领域的热点
\[9\]。在调度方面，GrapheonRL \[17\]、DRAS \[18\]
等工作展示了GNN-RL智能体在模拟环境中学习调度策略的潜力。然而，正如引言中所述，这些方法普遍受困于低样本效率（需要数百万次模拟才能收敛）和缺乏可解释性
\[12\]。

**C. 检索增强与"部署鸿沟"（Retrieval Augmentation and the \"Deployment
Gap\"）**

1.  **检索增强生成 (RAG)：** 在NLP领域，RAG \[13\]
    作为一种将参数化知识（神经网络）和非参数化知识（外部知识库）相结合的范式取得了巨大成功。RAG通过检索相关文档来增强大型语言模型的生成过程。我们认为，这一范式可以被创造性地迁移到HPC调度中：即**通过检索高质量的历史调度经验，来增强DRL智能体的决策过程**。这构成了本文方法论的核心（详见第四节）。

2.  **部署鸿沟：** 目前的AI调度研究（如 \[11\], \[17\],
    \[18\]）与真实的HPC生产环境之间存在着显著的"部署鸿沟"（Deployment
    Gap）。现有研究大多假设一个专用的AI调度器可以完全接管集群，这在已经部署了Slurm
    \[4\]
    等成熟管理软件的生产环境中是不可行的。这些AI模型（如GNN-RL）的推理开销也可能给调度器控制器带来无法接受的延迟
    \[20\]。

3.  **本文定位：** WASS-RAG 框架主要致力于解决 GNN-DRL
    调度器在样本效率和可解释性方面（2.B 节）的核心挑战。如
    **表I（已修改）** 所示，与 GrapheonRL \[17\] 等纯 GNN-RL
    方法不同，WASS-RAG 的核心创新在于**将 RAG
    范式引入调度决策过程**。通过利用一个由"多教师"引导的知识库（详见 4.C
    节），我们的方法（详见 4.D
    节）显著加速了收敛，并为调度决策提供了可追溯的依据。

表1： WASS-RAG 与先进调度方法的比较分析

  ------------------------------------------------------------------------------------------------------
  特征               GrapheonRL             DRAS                  WASS-RAG（我们的提案）
  ------------------ ---------------------- --------------------- --------------------------------------
  核心机器学习模型   GNN + RL（PPO）        DRL（分层神经网络）   GNN + DRL（PPO）

  集成模型           整体式                 智能体-模拟器交互     混合式：客户端（训练/编码）+
                                                                  插件（推理）

  核心差异点         模拟中接近最优的调度   学习填补策略          在生产调度器中部署学习策略的实用框架
  ------------------------------------------------------------------------------------------------------

# 知识增强的调度方法论 {#知识增强的调度方法论 .Head1}

本节详细阐述了WASS-RAG框架实现智能调度的核心方法论。我们将HPC工作流调度在数学上建模为一个马尔可夫决策过程（MDP），并引入一个由检索增强生成（RAG）机制所指导的GNN-DRL智能体来求解该MDP。

#### A. 调度马尔可夫决策过程 (Scheduling as an MDP) {#a.-调度马尔可夫决策过程-scheduling-as-an-mdp .unnumbered}

我们遵循标准定义，将调度问题形式化为一个MDP，由元组 $(S,A,P,R,\gamma)$
定义 ：

-   **状态空间 (State Space** $S$**)**: 在任意时间步 $t$，系统状态
    $s_{t} \in S$ 捕获了工作流DAG和HPC集群的瞬时快照 。状态 $s_{t}$
    包含了就绪任务集 $T_{\text{ready}}$、机器资源集 $R$
    及其当前负载、以及任务间的依赖关系 。
-   **动作空间 (Action Space** $A$**)**: 在状态 $s_{t}$
    下，可行的动作空间 $A_{s_{t}}$ 是所有可能的 (任务, 资源)
    配对的集合：
    $A_{s_{t}} = \{\left( T_{j},R_{k} \right) \mid T_{j} \in T_{\text{ready}}\left( s_{t} \right),R_{k} \in R\}$
    。
-   **转移概率 (Transition Probability** $P$**)**:
    $P\left( s_{t + 1} \mid s_{t},a_{t} \right)$ 代表在状态 $s_{t}$
    执行动作 $a_{t}$ 后转移到状态 $s_{t + 1}$ 的概率
    。在我们的仿真环境（WRENCH \[14\]）中，该转移是确定性的 。
-   **奖励函数 (Reward Function** $R$**)**:
    为激励智能体尽快完成工作流，我们定义了一个稠密的即时奖励
    $r_{t} = r\left( s_{t},a_{t} \right)$
    。一个简单而有效的奖励是最小化时间流逝 ：

$$r_{t} = - \Delta t_{t}\quad\quad(1)$$

-   其中 $\Delta t_{t}$ 是执行动作 $a_{t}$（即任务 $T_{j}$ 在 $R_{k}$
    上运行）所消耗的时间 。

```{=html}
<!-- -->
```
-   **目标函数 (Objective)**: 智能体的目标是学习一个策略
    $\pi\left( a_{t} \mid s_{t} \right)$
    ，以最大化折扣累积奖励的期望，即最小化完工时间（Makespan）：

$$\max_{\pi}J(\pi) = \mathbb{E}_{\tau \sim \pi}\left\lbrack \sum_{t = 0}^{T}\gamma^{t}r_{t} \right\rbrack\quad\quad(2)$$

-   其中 $\tau$ 是一个完整的调度轨迹（Episode），
    $\gamma \in \lbrack 0,1\rbrack$ 是折扣因子 。

#### B. 解耦的GNN状态编码架构 (Decoupled GNN State Encoding Architecture) {#b.-解耦的gnn状态编码架构-decoupled-gnn-state-encoding-architecture .unnumbered}

为了解决DRL训练中编码器更新导致的"语义漂移"问题（即当前状态嵌入与知识库历史嵌入不在同一向量空间），我们采用了一种**双编码器架构**。此架构将用于决策的**策略编码器**与用于检索的**检索编码器**解耦。

1.  **异构图构建**: 我们首先将高维、结构化的状态 $s_{t}$
    构建为一个异构图 $G_{t}$ 。该图包含三种类型的节点： 任务节点
    (Task)、 资源节点 (Resource) 和 文件节点 (File) 。节点特征
    $h_{v}^{(0)}$ 基于 RAG-Sched 草稿中的表2 进行初始化 。
2.  **GNN 消息传递**: 两个编码器均采用关系GNN（Relational GNN）\[22\]
    来处理不同类型的节点和边（关系 $\mathcal{R}$）。在第 $l$ 层，节点
    $v$ 的隐藏表示 $h_{v}^{(l)}$ 按如下方式更新 ：

$$h_{v}^{(l)} = \psi^{(l)}\left( h_{v}^{(l - 1)},\text{AGG}\left\{ \phi_{R}^{(l)}\left( h_{u}^{(l - 1)},e_{uv} \right) \mid R\mathcal{\in R,}u \in \mathcal{N}_{R}(v) \right\} \right)\quad\quad(3)$$

-   其中 $\phi_{R}$ 是消息函数， $\psi$ 是更新函数，AGG 是聚合函数 。

3.  **解耦的状态嵌入**: 经过 $L$ 层传播后，我们通过读出（Readout）函数
    聚合节点表示，以获得两个**不同**的图级别嵌入：
    -   **策略嵌入** $\mathbf{h}_{\text{policy},t}$: 由**策略编码器**
        $GNN_{\theta_{\text{policy}}}$（参数 $\theta_{\text{policy}}$
        在DRL训练中**持续更新**）生成。此嵌入被输入DRL智能体的Actor和Critic网络，用于**决策**。
    -   **检索嵌入** $\mathbf{s}_{\text{key},t}$: 由**检索编码器**
        $GNN_{\theta_{\text{key}}}$（参数 $\theta_{\text{key}}$
        在训练开始前**完全冻结**）生成。此嵌入仅用于在知识库
        $\mathcal{K}$ 中进行**检索** 。

这种解耦确保了所有用于检索的"键"（历史
$\mathbf{s}_{\text{key},i}$）和"查询"（当前
$\mathbf{s}_{\text{key},t}$）始终位于同一个、固定的向量空间，使得相似度比较在数学上是有效且一致的。

#### C. 调度知识库 (Scheduling Knowledge Base) {#c.-调度知识库-scheduling-knowledge-base .unnumbered}

为了解决DRL的"冷启动"和样本效率低下的问题
，我们引入了一个非参数化的调度知识库 $\mathcal{K}$ 。

-   **知识模式 (Schema)**: $\mathcal{K}$ 被形式化为一个键值存储
    $\mathcal{K = \{}\left( \mathbf{s}_{\text{key},i},q_{i} \right)\}_{i = 1}^{N}$
    。

    -   **键 (Key)** $\mathbf{s}_{\text{key},i}$:
        一个由**冻结的检索编码器 (GNN-Key)** 编码的历史状态嵌入 。
    -   **值 (Value)** $q_{i}$: **（已修正）** $q_{i}$
        不再是整个轨迹的最终性能，而是对**状态** $s_{i}$
        **的真实价值**的估计，即从该状态出发的**期望未来回报（或剩余成本）**的估计
        $V\left( s_{i} \right)$。这解决了将轨迹级回报错误归因于状态级的"价值污染"问题。
        $q_{i}$ 是一个标量， $q_{i}$
        越高，代表该状态越"好"（即未来成本越低）。

-   **填充策略：多样化知识注入 (Diverse Knowledge Seeding)**:
    为了构建一个丰富且无偏见的知识库，我们采用"多教师"策略来引导启动
    $\mathcal{K}$
    。我们运行一组具有不同特性的调度策略（"教师"）来生成初始经验 ：

    -   **HEFT \[5\]**: 作为一个强大的启发式"教师"，提供高质量经验（高
        $q_{i}$ 值）。
    -   **MIN-MIN \[171\]**:
        作为另一个经典的启发式"教师"，它提供了不同的调度偏好，丰富了KB的多样性
        。
    -   **Random Policy (随机策略)**:
        作为一个"坏教师"，它提供了低质量的经验（低 $q_{i}$
        值），帮助智能体学会识别并"避开"导致性能灾难的状态空间区域 。

```{=html}
<!-- -->
```
-   对于这些教师数据， $q_{i}$ 被计算为：从状态 $s_{i}$
    开始，继续遵循该教师策略（如HEFT）直至结束所产生的**剩余完工时间**
    $M_{\text{remain}}\left( s_{i} \right)$（的标准化值）。

```{=html}
<!-- -->
```
-   **在线策展 (Online Curation)**:
    在DRL训练过程中，DRL智能体（"学生"）本身也会产生经验。当智能体探索到一个新状态
    $s_{t}$ 时，它可以使用其**自身的Critic网络**对该状态的价值估计
    $V_{\theta_{\text{policy}}}\left( s_{t} \right)$ 作为 $q_{t}$
    值，并将 $\left( \mathbf{s}_{\text{key},t},q_{t} \right)$
    键值对添加回 $\mathcal{K}$，实现了知识库的持续改进 。

#### D. 基于RAG的势函数奖励塑造 (RAG-based Potential-Based Reward Shaping) {#d.-基于rag的势函数奖励塑造-rag-based-potential-based-reward-shaping .unnumbered}

这是WASS-RAG方法论的核心 。我们利用这个"多教师"知识库
$\mathcal{K}$，通过 势函数奖励塑造 (PBRS) \[23\] 来指导DRL智能体 。PBRS
\[23\] 证明，通过增加一个形式如下的额外奖励，不会改变MDP的最优策略 ：
$r'_{t} = r_{t} + \gamma\Phi\left( s_{t + 1} \right) - \Phi\left( s_{t} \right)$，其中
$\Phi(s)$ 是一个仅依赖于状态的"势函数"。

我们的核心创新是定义了一个 基于RAG的势函数 $\Phi_{\text{RAG}}(s)$
，它利用知识库 $\mathcal{K}$ 来估计任何给定状态 $s$
的"先验价值"。该过程分为三步 ：

4.  **检索 (Retrieve)**: 给定DRL智能体在 $t$ 时刻的当前状态
    $s_{t}$，我们首先使用**冻结的检索编码器** $\text{GNN-Key}$
    来计算其查询嵌入 $\mathbf{s}_{\text{key},t}$ 。我们使用此
    $\mathbf{s}_{\text{key},t}$ 作为查询，从（"多教师"）知识库
    $\mathcal{K}$ 中检索 $k$ 个最相似的历史状态键
    $\mathbf{s}_{\text{key},j}$ 。

$$\mathcal{K}_{\text{retrieved}}\left( s_{t} \right) = \{\left( \mathbf{s}_{\text{key},j},q_{j} \right)\}_{j = 1}^{k} \leftarrow \text{k-NN}\left( \mathbf{s}_{\text{key},t}\mathcal{,K} \right)$$

-   检索使用高效的向量相似度搜索（如余弦相似度
    $\text{sim}( \cdot , \cdot )$）完成 。

5.  **插值 (Interpolate)**: 我们将势函数
    $\Phi_{\text{RAG}}\left( s_{t} \right)$ 定义为这 $k$
    个检索到的（状态级）经验质量 $q_{j}$ 的加权插值
    （例如，Softmax加权）：

$$w_{j} = \frac{\exp\left( \text{sim}\left( \mathbf{s}_{\text{key},t},\mathbf{s}_{\text{key},j} \right)/\tau \right)}{\sum_{i = 1}^{k}\exp\left( \text{sim}\left( \mathbf{s}_{\text{key},t},\mathbf{s}_{\text{key},i} \right)/\tau \right)}\quad\quad(5)$$

$$\Phi_{\text{RAG}}\left( s_{t} \right) = \sum_{j = 1}^{k}w_{j} \cdot q_{j}\quad\quad(6)$$

-   其中 $\tau$ 是温度系数 。直观地，
    $\Phi_{\text{RAG}}\left( s_{t} \right)$ 通过RAG"估算"了当前状态
    $s_{t}$ 的历史经验价值------如果 $s_{t}$
    接近于历史上（无论来自哪个"教师"）导致"好"结果的状态（高 $q_{i}$
    值），其势函数值就高 。

6.  **塑造 (Shape)**:
    我们将这个RAG势函数代入PBRS公式，定义了我们的增强奖励 $r'_{t}$ ：

$$r'_{t} = r_{t} + \lambda \cdot \left( \gamma\Phi_{\text{RAG}}\left( s_{t + 1} \right) - \Phi_{\text{RAG}}\left( s_{t} \right) \right)\quad\quad(7)$$

-   其中 $\lambda$ 是一个超参数，用于平衡原始奖励和知识引导 。

如果一个动作 $a_{t}$ 使得系统从一个"低知识价值"的状态 $s_{t}$
转移到一个"高知识价值"的状态 $s_{t + 1}$ （即
$\Phi\left( s_{t + 1} \right) > \Phi\left( s_{t} \right)$），智能体就会收到一个
**正的引导奖励**
。这种机制利用了来自多个教师的集体智慧（现在是基于准确的状态价值估计），极大地加速了DRL智能体的收敛
。

# 实验方法论 {#实验方法论 .Head1}

为严谨验证WASS-RAG框架的性能、实用性与可解释性，我们设计了全面的实验方法论。该方案依托高保真仿真环境进行知识库生成与智能体训练，采用多维度基线模型进行对比，并通过多样化的工作负载与集群配置进行评估。

**V. 实证评估协议 (Empirical Evaluation Protocol)**

为了严格评估 WASS-RAG
框架的有效性（特别是"多教师"RAG-PBRS算法的有效性），我们设计了一套详尽的评估协议。

**A. 仿真环境与平台 (Simulation Environment)**

所有实验都将在 **WRENCH** \[14\] 模拟器上进行。WRENCH 基于 SimGrid
\[21\] 构建，是一个高保真、可扩展的模拟平台，专为科学工作流设计。

1.  **HPC 平台建模：** 我们的模拟环境将按照 `RAG-Sched`
    中的定义，建模一个典型的异构HPC集群，其关键特征包括：
    a.  **计算资源：** 包含CPU密集型节点和配备GPU的加速节点。
    b.  **存储系统：**
        建模一个共享的并行文件系统，具有明确的读/写带宽限制。
    c.  **网络拓扑：**
        采用Fat-Tree（胖树）拓扑，精确建模网络延迟和带宽。
2.  **动态负载：**
    为了模拟真实的生产环境，我们还将在集群中引入背景"噪声"负载，即模拟由其他用户提交的、与主工作流竞争资源的作业。

**B. 基准工作流 (Benchmark Workloads)**

为了全面评估调度器在不同负载特性下的鲁棒性，我们采用了 `RAG-Sched`
中选定的三个具有代表性的真实科学工作流：

1.  **Montage (天体物理学) \[3\]：**
    I/O密集型（I/O-intensive）工作流，对调度器的I/O感知能力提出了极高要求。
2.  **LIGO (引力波物理学) \[1\]：**
    计算密集型（CPU-intensive）工作流，测试调度器处理大规模并行任务的能力。
3.  **CyberShake (地震学) \[2\]：**
    异构计算（Heterogeneous）工作流，包含CPU和GPU任务，用于测试异构资源分配能力。

**C. 对比基线 (Comparison Baselines) (已修改)**

为了量化 WASS-RAG
框架（特别是"多教师"RAG机制）的性能增益，我们将与以下**五种**关键基线进行对比：

1.  **FCFS (First-Come, First-Served)：** 传统Slurm \[4\]
    的默认策略，不感知DAG结构（作为性能下限）。

2.  **HEFT (Heterogeneous Earliest Finish Time) \[5\]：**
    静态的、DAG感知的启发式算法。它是我们知识库的"教师"之一，也是一个强大的性能基线。

3.  **MIN-MIN：** 另一种经典的启发式算法，也是我们知识库的"教师"之一。

4.  **WASS-DRL (Vanilla)：** 这是 WASS-RAG
    框架的一个**核心消融**版本。它使用与WASS-RAG完全相同的GNN编码器和DRL智能体（第四节），但**完全移除了RAG模块**（即没有知识库
    $\mathcal{K}$ 和势函数奖励
    $\Phi_{\text{RAG}}$）。此基线用于隔离并证明RAG模块的价值。

```{=html}
<!-- -->
```
2.  **WASS-RAG (HEFT-only)：**
    这是第二个**关键消融**版本。它使用完整的RAG-PBRS机制，但其知识库
    $\mathcal{K}$ **仅由HEFT \[5\]
    单一教师**进行引导启动。此基线用于验证我们的"多教师"（4.C.2节）假设，即**多样化的知识库**优于单一的知识库。

3.  **WASS-RAG (Full) (本文方法)：** 即第四节中描述的完整框架，其知识库
    $\mathcal{K}$ 由"多教师"（HEFT, MIN-MIN, Random）共同引导启动。

**D. 评估指标 (Evaluation Metrics)**

我们将从性能和效率两个维度对调度策略进行全面评估：

1.  **主要指标（性能）：**
    a.  **工作流完工时间 (Makespan)：**
        定义为工作流的最后一个任务完成的时间。这是评估调度性能**最重要**的指标。
    b.  **平均周转时间 (Average Turnaround Time)：**
        工作流中所有任务的平均（完成时间 - 提交时间）。
    c.  **系统资源利用率 (Resource Utilization)：**
        集群在工作流执行期间的平均资源（CPU/GPU）占用率。
2.  **次要指标（效率）：**
    a.  **RL收敛速度 (Convergence Speed)：** 比较
        `WASS-RAG (Full)`、`WASS-RAG (HEFT-only)` 和
        `WASS-DRL (Vanilla)`
        达到稳定策略所需的训练周期（Episodes）数量。此指标用于验证"多教师"RAG机制是否（如预期地）提供了最快的学习效率。

# 预期结果与分析 (Expected Results and Analysis) {#预期结果与分析-expected-results-and-analysis .Head1}

基于第五节中定义的严格评估协议，本节阐述了我们预期的实验结果，并分析这些结果将如何验证
WASS-RAG 框架的有效性。

**A. 性能优越性 (Makespan 与资源利用率)**

我们预期 `WASS-RAG (Full)`（本文方法）在所有三个基准工作流（Montage,
LIGO, CyberShake）上的**工作流完工时间 (Makespan)**
将显著优于所有五个对比基线。

7.  **对比传统基线 (FCFS, HEFT, MIN-MIN)：** 预期 `WASS-RAG (Full)`
    将展现全面优势。与 FCFS 不同，它是DAG感知的。与静态的 HEFT \[5\] 和
    MIN-MIN 不同，WASS-RAG 的 GNN-DRL 智能体是动态的，它能实时感知 5.A.2
    节中引入的"动态背景负载"，从而做出更符合现实的决策。

8.  **对比 WASS-DRL (Vanilla) (消融1)：** 预期 `WASS-RAG (Full)`
    将显著优于 `Vanilla` 版本。这验证了RAG机制的总体价值：`Vanilla`
    智能体在面对复杂组合优化时容易陷入次优解，而RAG-PBRS（第四节D）通过历史经验提供了强有力的引导。

9.  **对比 WASS-RAG (HEFT-only) (消融2)：**
    这是验证我们"多教师"（4.C.2节）假设的**最关键对比**。我们预期
    `WASS-RAG (Full)` 的性能将优于 `WASS-RAG (HEFT-only)`。

    -   **预期分析：** `HEFT-only`
        的知识库是**有偏见的**。在HEFT表现良好的工作流上（如LIGO），二者性能可能相近。但在HEFT表现不佳的工作流上（例如，Montage
        \[3\] 的I/O瓶颈是HEFT的盲点），`HEFT-only`
        智能体将继承其"教师"的偏见。
    -   相比之下，`WASS-RAG (Full)` 的知识库由
        `HEFT`（好经验）、`MIN-MIN`（不同偏好的经验）和
        `Random`（坏经验）共同组成。这种**多样性**（尤其是"坏经验"）使得势函数
        \$\\Phi\_{\\text_RAG}\$
        能够更准确地评估状态空间，引导智能体学会HEFT所不知道的、规避I/O瓶颈的更优策略。

**B. 学习效率 (RL 收敛速度)**

我们预期在比较 `Vanilla`、`HEFT-only` 和 `Full`
三个版本的**RL收敛速度**时，将观察到清晰的层次（如**图Y**所示，*注：预期的学习曲线图*）：

1.  **WASS-DRL (Vanilla)**
    （红色曲线）：收敛最慢。由于"冷启动"，它将经历漫长的随机探索阶段，学习曲线最平坦。
2.  **WASS-RAG (HEFT-only)**
    （绿色曲线）：收敛较快。得益于HEFT的引导启动，它会迅速达到一个较高的性能基线。但由于知识库的偏见，它可能过早收敛到HEFT的次优水平。
3.  **WASS-RAG (Full)**
    （蓝色曲线）：收敛最快且性能最高。得益于"多教师"知识库提供的丰富信号（$q_{i}$有好有坏），其
    \$\\Phi\_{\\text_RAG}\$
    势函数提供的引导最准确，使智能体能最快地找到全局最优策略。

这将有力地证明，我们的"多教师"RAG-PBRS机制是解决"样本效率低下"问题的最有效途径。

**C. 可解释性案例研究 (Interpretability Case Study)**

最后，我们将提供一个案例研究来定性地展示RAG带来的可解释性。

-   **场景：** 我们将从一个 CyberShake \[2\] 工作流的执行中，提取一个
    `WASS-RAG (Full)`
    做出"反直觉"决策的时刻（例如，它没有将一个关键GPU任务分配给最快的A100
    GPU，而是分配给了较慢的V100 GPU）。
-   **分析：**
    我们将展示，通过查询"知识引导教师"（4.D节），该决策是基于从KB
    $\mathcal{K}$ 中检索到的 $k = 3$ 个相似历史经验。这些经验（可能来自
    `Random` 策略的"坏经验"或 `HEFT`
    策略的"好经验"）共同表明：在当时的状态下（例如，A100
    的网络I/O正忙），将任务分配给A100会导致一个（$q_{i}$值很低的）坏结果，而分配给V100反而能实现更高的（$q_{i}$值很高的）预期价值。
-   **结论：**
    这种将决策追溯到具体历史经验（无论好坏）的能力，为"黑盒"的AI调度提供了急需的、基于证据的可解释性。

参考文献

1.  B. P. Abbott et al. (LIGO Scientific Collaboration and Virgo
    Collaboration), \"LIGO: The Laser Interferometer Gravitational-Wave
    Observatory,\" *Rep. Prog. Phys.*, vol. 72, no. 7, p. 076901, 2009.

2.  G. P. B. V. E. K. S. S. P. M. Y. G. C. T. H. J. T. J. P. S. R. W.
    Scott et al., \"Using open-science workflow tools to produce SCEC
    CyberShake physics-based probabilistic seismic hazard models,\"
    *Front. High Perform. Comput.*, vol. 1, 2024.

3.  A. C. Berriman, G. B. et al., \"Montage: A grid enabled engine for
    delivering custom science-grade mosaics on demand,\" *Proc. SPIE*,
    vol. 5493, pp. 221--232, 2004.

4.  A. B. Yoo, M. A. Jette, and M. Grondona, \"SLURM: Simple Linux
    utility for resource management,\" in *Job Scheduling Strategies for
    Parallel Processing (JSSPP 2003)*, LNCS, vol. 2862, Berlin,
    Heidelberg: Springer, 2003, pp. 44--60.

5.  H. Topcuoglu, S. Hariri, and M. Y. Wu, \"Performance-effective and
    low-complexity task scheduling for heterogeneous computing,\" *IEEE
    Trans. Parallel Distrib. Syst.*, vol. 13, no. 3, pp. 260--274, Mar.
    2002.

6.  F. Suter, H. Casanova, and R. Ferreira da Silva, \"How to Use WRENCH
    to Simulate Workflow Management Systems,\" *Tech. Rep.*, 2020.

7.  T. D. Braun et al., \"A comparison of eleven static heuristics for
    mapping a class of independent tasks onto heterogeneous distributed
    computing systems,\" *J. Parallel Distrib. Comput.*, vol. 61, no. 6,
    pp. 810--837, 2001.

8.  G. Rattihalli, A. Dhakal, S. R. Chalamalasetti, D. Milojicic, and E.
    Frachtenberg, \"Opportunistic Energy-Aware Scheduling for Container
    Orchestration Platforms Using Graph Neural Networks,\" in *Proc.
    24th IEEE/ACM Int. Symp. Cluster, Cloud Grid Comput. (CCGrid \'24)*,
    2024, pp. 1--11.

9.  H. Mao, M. Alizadeh, I. Menache, and S. Kandula, \"Resource
    management with deep reinforcement learning,\" in *Proc. 15th ACM
    Workshop on Hot Topics in Networks (HotNets)*, 2016.

10. T. N. Kipf and M. Welling, \"Semi-supervised classification with
    graph convolutional networks,\" in *Proc. 5th Int. Conf. Learn.
    Represent. (ICLR)*, 2017.

11. S. Shah, V. T. P. H. S. V. C. R. A. N. and S. S. G. R. V. A. B.,
    \"GrapheonRL: A Graph-based Approach for Reinforcement Learning in
    HPC Scheduling,\" *arXiv preprint arXiv:2506.00260*, 2025.

12. A. D. M. L. N. V. M. D. J. P. E. A. W. C. S. H. K. P. N. D. F.,
    \"Wander: An Explainable Decision-Support Framework for HPC,\"
    *arXiv preprint arXiv:2506.04049*, 2025.

13. P. Lewis, E. Perez, A. Piktus, et al., \"Retrieval-augmented
    generation for knowledge-intensive NLP tasks,\" in *Proc. Adv.
    Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp.
    9459--9474.

14. H. Casanova, R. Ferreira da Silva, and F. Suter, \"WRENCH: A
    framework for simulating workflow management systems,\" *Proc. 20th
    IEEE/ACM Int. Symp. Cluster, Cloud, Grid Comput. (CCGrid)*, 2020.

15. J. D. Ullman, \"NP-complete scheduling problems,\" *J. Comput. Syst.
    Sci.*, vol. 10, no. 3, pp. 384--393, 1975.

16. W. L. Hamilton, R. Ying, and J. Leskovec, \"Inductive representation
    learning on large graphs,\" in *Proc. Adv. Neural Inf. Process.
    Syst. (NeurIPS)*, 2017, pp. 1024--1034.

17. O. H. Ibarra and S. K. Kim, \"Heuristic algorithms for scheduling
    independent tasks on nonidentical processors,\" *J. ACM*, vol. 24,
    no. 2, pp. 280--289, Apr. 1977.

18. K. Liu, Z. Wan, J. Lin, et al., \"DRAS: Deep Reinforcement Agent for
    Scheduling in HPC,\" *arXiv preprint arXiv:2102.06243*, 2021.

19. M. G. N. G. R. S. S. A. J. S. P. A. B. P. N. G. J. B., \"Analyzing
    the performance overhead of Slurm plugins in production HPC
    environments,\" *J. High Perform. Comput. Appl.*, 2023.

20. H. Casanova, A. Giersch, A. Legrand, M. Quinson, and F. Suter,
    \"SimGrid: A sustained effort for the versatile simulation of large
    scale distributed systems,\" *Concurrency Computat.: Pract. Exper.*,
    vol. 28, no. 14, 2016.

21. X. Wang, H. Ji, C. Shi, et al., \"Heterogeneous graph attention
    network,\" in *Proc. The World Wide Web Conf. (WWW)*, 2019, pp.
    2022--2032.

22. A. Y. Ng, D. Harada, and S. Russell, \"Policy invariance under
    reward transformations: Theory and application to reward shaping,\"
    in *Proc. 16th Int. Conf. Mach. Learn. (ICML)*, 1999, pp. 278--287.

[^1]: 资助项目：由联想校企合作项目资助，项目编号：202405SJTU01-BUBG017
