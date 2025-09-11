WASS-RAG: A Knowledge-Retrieval Augmented DRL Framework for
Workflow-Aware Scheduling on Slurm

WASS-RAG: Knowledge-Retrieval Augmented Workflow Scheduling

A Hybrid Architecture for Augmenting Production Schedulers with
Explainable AI

Zhao Tao

Lenovo (Tianjin) Co. , Ltd., zhaotao1@lenovo.com

Wu ZhongXin

Lenovo (Tianjin) Co. , Ltd., wuzx9@lenovo.com

Sheng JiaJie

Shanghai Jiao Tong University, jiajie.sheng@sjtu.edu.cn

Yu Yang

Lenovo (BeiJing) Co. , Ltd., yuyang5@lenovo.com

Wei JianWen[^1]

Shanghai Jiao Tong University, weijianwen@sjtu.edu.cn

Modern scientific discovery increasingly depends on complex workflows,
making high-performance scheduling critical. While Deep Reinforcement
Learning (DRL) schedulers show promise, they often suffer from sample
inefficiency and act as \"black boxes,\" hindering their trustworthiness
and adoption in mission-critical environments. To address these
limitations, we propose WASS-RAG, a novel scheduling framework that
introduces a knowledge-guided learning paradigm. Its core innovation is
the integration of Retrieval-Augmented Generation (RAG) into the DRL
loop. A \"Knowledge-Guided Teacher\" retrieves topologically similar
historical cases to provide experience-based, context-aware reward
signals, significantly accelerating and stabilizing the DRL agent\'s
learning process. Our experiments on a physical cluster demonstrate that
WASS-RAG reduces workflow makespan by up to 49.1% compared to the
baseline and outperforms the state-of-the-art HEFT algorithm by 24.0%.
Crucially, the RAG component provides an additional 6.2% performance
gain over its non-RAG DRL counterpart, while also offering explainable
decisions grounded in historical data. WASS-RAG thus presents a more
efficient, transparent, and powerful approach to AI-based scheduling.

**CCS CONCEPTS** • Computing methodologies → Machine learning →
Reinforcement learning; • Software and its engineering → Software
organization and properties → Interoperability; • Information systems →
Information retrieval → Retrieval models and ranking

**Additional Keywords and Phrases:** Workflow Scheduling, Deep
Reinforcement Learning, Retrieval-Augmented Generation, Knowledge Base,
Slurm, Explainable AI

# Introduction {#introduction .Head1}

Modern scientific discovery, from climate modeling and genomic
sequencing to drug discovery, increasingly depends on the execution of
large-scale, complex workflows on High-Performance Computing (HPC)
clusters. The performance of these workflows---often represented as
Directed Acyclic Graphs (DAGs)---is critically determined by the
efficiency of the underlying scheduling system. An optimal scheduling
strategy that minimizes the total execution time (makespan) can
significantly shorten research cycles and accelerate the pace of
scientific innovation. Consequently, the development of advanced
scheduling algorithms remains a cornerstone of HPC research.

In response to the limitations of traditional heuristic-based schedulers
(e.g., HEFT), the research community has begun to explore the potential
of Deep Reinforcement Learning (DRL) to automate and optimize the
scheduling process. These AI-driven approaches have demonstrated
promising results in simulated environments. However, their practical
adoption in production systems is hindered by two fundamental
bottlenecks. First, DRL agents typically suffer from sample
inefficiency, requiring extensive trial-and-error exploration to
converge to an effective policy, a process that is prohibitively
expensive and time-consuming on real-world HPC resources. Second, the
\"black-box\" nature of DRL models presents a significant barrier to
trust; system administrators are reluctant to deploy schedulers whose
decision-making processes are opaque and unexplainable, especially in
mission-critical scientific environments.

To address these challenges, we propose WASS-RAG, a novel scheduling
framework that introduces a knowledge-guided learning paradigm for
workflow scheduling. The core innovation of our framework is the
integration of Retrieval-Augmented Generation (RAG) into the DRL
training loop. We introduce a \"Knowledge-Guided Teacher\" mechanism
that, when faced with a new scheduling decision, retrieves topologically
similar, successfully executed workflow cases from a historical
knowledge base. These retrieved cases provide the DRL agent with
contextually relevant, experience-based reward signals. This approach
fundamentally transforms the learning process: it directly leverages
past successes to guide the agent\'s exploration, thereby tackling the
sample inefficiency problem. Furthermore, by grounding each decision in
concrete historical examples, the framework provides a clear, traceable
rationale, thus addressing the critical need for explainability.

We conducted a comprehensive \"Sim-to-Real\" experimental evaluation to
validate the effectiveness of WASS-RAG on a physical cluster. The
results demonstrate that our framework achieves a makespan reduction of
up to 49.1% compared to the production baseline scheduler and
outperforms the state-of-the-art HEFT algorithm by 24.0%. Crucially, the
introduction of the RAG mechanism yields an additional 6.2% performance
improvement over its non-RAG DRL counterpart, confirming the value of
knowledge-guided learning. Taken together, our work presents a novel
knowledge-guided DRL paradigm that significantly enhances both
scheduling performance and decision transparency, offering a practical
path toward deploying intelligent schedulers in production environments.

# Related Work {#related-work .Head1}

The application of machine learning to enhance HPC scheduling is a
vibrant research area. Existing work can be broadly categorized into
three main approaches, which collectively highlight the unique niche and
contribution of WASS-RAG.

-   **Prediction-Informed Scheduling**: This line of research uses ML
    models as "predictors" to provide more accurate inputs for
    traditional scheduling algorithms. For instance, various studies
    have successfully used neural networks to predict job runtimes,
    which in turn improves the efficiency of backfilling schedulers.
    While effective at optimizing existing heuristics, these methods do
    not fundamentally change the decision-making logic itself.

-   **End-to-End DRL Schedulers**: A more ambitious approach aims to
    completely replace the core scheduling logic with an end-to-end DRL
    agent. Systems like DRAS have shown that a DRL agent can learn
    complex scheduling policies, such as backfilling and resource
    reservation, in simulated environments. The primary challenge for
    these systems, however, lies in their integration with
    production-level schedulers written in low-level languages,
    presenting a significant engineering hurdle.

-   **GNN+DRL for Graph-Structured Problems**: The most relevant
    research area combines GNNs with DRL to tackle problems with
    inherent graph structures, such as workflow scheduling. Frameworks
    like GrapheonRL have demonstrated the power of using a GNN to encode
    the state of a workflow and resource graph, providing a rich
    representation for the DRL agent. These works have shown immense
    potential, primarily in simulation.

A critical analysis of these pioneering efforts reveals a common
challenge: a "Deployment Gap" between state-of-the-art academic research
and production HPC practice. Many advanced ML-based schedulers are
either designed as monolithic systems that require replacing stable
schedulers like Slurm or are validated exclusively in simulators, with
no clear, low-risk path to production deployment.

WASS-RAG is explicitly designed to bridge this gap. Our core
contribution is not merely the application of GNN+DRL to scheduling, but
the proposal of a practical, non-invasive architectural paradigm that
allows for the safe and efficient deployment of learned policies within
a production scheduler. By decoupling the intensive training and
encoding on the client-side from the lightweight inference within the
Slurm plugin, WASS-RAG augments rather than replaces the existing
infrastructure, offering a pragmatic path for transitioning AI research
into production-ready tools. Table 1 provides a detailed comparison that
highlights these key differentiators.

> Table 1: Comparative analysis of WASS-RAG against state-of-the-art
> scheduling approaches. The comparison highlights WASS-RAG's unique
> contribution as a practical framework for deploying learned policies
> in production environments, bridging the common \"deployment gap\".

  ------------------------------------------------------------------------------------------
  Feature          GrapheonRL     DRAS                   WASS (Heuristic) WASS-RAG (Our
                                                                          Proposal)
  ---------------- -------------- ---------------------- ---------------- ------------------
  Core ML Model    GNN + RL (PPO) DRL (Hierarchical NN)  N/A (Rule-based) GNN + DRL (PPO)

  Integration      Monolithic     Agent-Simulator        Hybrid: Client + Hybrid: Client
  Model            (assumed)      Interaction            Slurm Plugin     (Train/Encode) +
                                                                          Plugin (Infer)

  Deployment       Low (Requires  Medium                 High             High (Preserves
  Practicality     replacement)   (Simulation-focused)   (Non-invasive)   non-invasive
                                                                          architecture)

  Key              Near-optimal   Learns backfilling     Practical data   A practical
  Differentiator   scheduling in  policies               locality         framework for
                   simulation                            optimization     deploying learned
                                                                          policies in
                                                                          production
                                                                          schedulers
  ------------------------------------------------------------------------------------------

# The WASS-RAG Architecture {#the-wass-rag-architecture .Head1}

At the core of the WASS-RAG framework lies a novel, non-invasive hybrid
scheduling architecture designed to bridge the deployment gap in
production HPC environments. This architecture, a key contribution of
our work, separates global planning from local execution, enabling
advanced scheduling intelligence without intrusive modifications to the
core scheduler daemon.

This section details the architectural blueprint. We first present the
foundational design of the hybrid model and its initial implementation
with a heuristic-driven engine. We then describe its evolution into a
fully autonomous, knowledge-driven platform, redefining the component
roles to support the complete lifecycle of a sophisticated,
Retrieval-Augmented DRL agent.

## Decoupled Architecture: Offline Training and Online Inference {#decoupled-architecture-offline-training-and-online-inference .Head2}

![图示 AI
生成的内容可能不正确。](media/image1.png){width="5.829770341207349in"
height="3.853576115485564in"}

Figure 1: WASS - RAG Architecture: Decoupled Offline Training and Online
Inference

A core design principle of the WASS-RAG framework is the strategic
decoupling of the computationally intensive training phase from the
lightweight online inference phase. This separation, illustrated in
Figure 1, is essential for ensuring that our AI-driven enhancements do
not introduce prohibitive overhead into the time-sensitive production
scheduling loop.

**Offline Training Domain**: The primary goal of the offline domain is
to produce a highly optimized scheduling policy model. The DRL agent is
trained within a high-fidelity HPC simulator (WRENCH), which provides a
safe and cost-effective environment for extensive trial-and-error
learning. This training loop is orchestrated by the WASS Client, which
manages the interactions between all components. At each training step,
the simulator provides the current system state, which is encoded into a
graph embedding by a GNN. This embedding serves a dual purpose: it is
passed to the DRL agent as its state observation, and it is used by the
Knowledge-Guided Teacher to query a vast Knowledge Base of historical
scheduling cases. The Teacher retrieves similar past experiences and,
based on the DRL agent's proposed action, generates a dense,
knowledge-rich RAG reward. This reward signal guides the agent's policy
update, dramatically accelerating learning. The final output of this
entire offline process is a compact, pre-trained DRL policy model, which
encapsulates the learned scheduling intelligence.

**Online Inference Domain**: In the production environment, when a user
submits a real workflow, the system operates in a streamlined inference
mode. The WASS Client submits the job to the Slurm controller, cleverly
encoding the real-time graph state embedding of the workflow into the
job's comment field. A lightweight WASS Lua Plugin, integrated into the
Slurm controller, intercepts this submission. The plugin's sole
responsibilities are to load the pre-trained DRL policy model, parse the
state embedding from the job's comment, and execute a rapid forward pass
through the policy network. This inference step produces an optimal
action---the best node to which the task should be assigned. The plugin
then modifies the job's description to enforce this placement decision
before handing it back to the Slurm controller for final dispatching.
This online process is exceptionally fast, involving only a GNN encoding
and a single network forward pass, thus imposing negligible overhead on
the production system.

## DRL-based Scheduling Core {#drl-based-scheduling-core .Head2}

At the heart of our framework lies a Deep Reinforcement Learning agent
responsible for making sequential task-to-node mapping decisions. We
formulate the workflow scheduling problem as a Markov Decision Process
(MDP), defined by a tuple of $\left( \mathcal{S,A,R} \right)$,
representing the state space, action space, and reward function,
respectively.

State Representation: An effective state representation must capture
both the structural properties of the workflow and the dynamic state of
the cluster. We employ a Graph Neural Network (GNN) to encode the
workflow DAG. The GNN processes the graph structure, where each node
represents a task and edges represent dependencies, generating a
low-dimensional embedding for each task. This embedding captures rich
topological information, such as a task's position in the workflow and
its relationship with predecessors and successors. The full state
$s_{t}\mathcal{\in S}$ at decision step $t$ is a concatenation of the
GNN embedding of the ready task $T_{i}$ and a vector representing the
current load and capabilities of all available cluster nodes
$\mathcal{N}$.

Action Space: The action space $\mathcal{A}$ is defined as the set of
all possible valid assignments for a ready task. At each step $t$, when
considering a ready task $T_{i}$, the DRL agent selects an action
$a_{t}\mathcal{\in A}$, which corresponds to assigning $T_{i}$ to one of
the available compute nodes $N_{j}\mathcal{\in N}$. The size of the
action space is equal to the number of nodes in the cluster,
$\left| \mathcal{A} \right| = \left| \mathcal{N} \right|$.

Baseline Reward Function: In a standard DRL formulation without our RAG
enhancement, the reward signal is often sparse and delayed. The agent
typically receives a final reward only after the entire workflow has
completed. This reward $R_{\text{final}}$ is inversely proportional to
the total makespan, for example, $R_{\text{final}} = 1/\text{Makespan}$.
While ultimately effective, this approach leads to the sample
inefficiency problem discussed earlier, as the agent struggles to
attribute a distant final outcome to a long sequence of individual
actions. The subsequent section will detail how our Knowledge-Guided
Teacher provides a much richer, more immediate reward signal to overcome
this limitation.

## The Knowledge-Guided Teacher: RAG-Enhancement Module {#the-knowledge-guided-teacher-rag-enhancement-module .Head2}

The centerpiece of our framework is the Knowledge-Guided Teacher, a
module that leverages Retrieval-Augmented Generation (RAG) to transform
the DRL agent's learning process. Instead of relying solely on sparse,
delayed rewards, our Teacher provides dense, context-aware, and
explainable reward signals based on a vast repository of historical
scheduling cases. This mechanism is realized through three key stages:
knowledge base construction, fast contextual retrieval, and
knowledge-guided reward generation.

1.  Knowledge Base Construction

The foundation of our knowledge-guided approach is a comprehensive
Knowledge Base, $\mathcal{K}$, which stores a multitude of successfully
executed workflow scenarios. This knowledge base is constructed offline
by running a variety of workflows with a baseline scheduler (e.g., HEFT)
on the simulator. For each completed workflow, we extract and store a
tuple representing a successful scheduling case, $c$:

$$c = \left( E_{\text{dag}},M_{\text{final}},\Pi \right)$$

where: $E_{\text{dag}}$ is the GNN-generated graph embedding of the
entire workflow DAG, serving as a unique, low-dimensional fingerprint of
its topology. $M_{\text{final}}$ is the final makespan achieved for this
workflow execution, representing the quality of the outcome.
$\Pi = \{\pi_{1},\pi_{2},...,\pi_{|V|}\}$ is the complete scheduling
trace, where each element $\pi_{i} = \left( T_{i},N_{j} \right)$ is a
tuple indicating that task $T_{i}$ was assigned to node $N_{j}$.

This curated collection of (embedding, outcome, trace) tuples forms the
experiential knowledge that the Teacher can draw upon during the online
training of the DRL agent.

2.  Fast Contextual Retrieval

During the offline training loop, whenever the DRL agent needs to make a
decision for a ready task $T_{i}$ from a new workflow, the
Knowledge-Guided Teacher is invoked. It first generates the GNN
embedding for the current state of the workflow DAG, denoted as
$E_{\text{current}}$. The Teacher then performs a fast similarity search
over the entire Knowledge Base $\mathcal{K}$ to find historical cases
that are topologically similar to the current workflow.

The similarity between the current workflow and a historical case
$c\mathcal{\in K}$ is measured by the cosine similarity between their
respective GNN embeddings:

$$\text{sim}\left( E_{\text{current}},c.E_{\text{dag}} \right) = \frac{E_{\text{current}} \cdot c.E_{\text{dag}}}{\parallel E_{\text{current}} \parallel \parallel c.E_{\text{dag}} \parallel}$$

The Teacher retrieves the top-$k$ most similar historical cases, forming
a context set
$\mathcal{C}_{\text{retrieved}} = \{ c_{1},c_{2},...,c_{k}\}$. This
retrieval process is highly efficient as it operates on pre-computed,
low-dimensional embeddings.

3.  Knowledge-Guided Reward Generation

With the retrieved context set $\mathcal{C}_{\text{retrieved}}$, the
Teacher can now provide an immediate, informed reward for any action
$a_{t}$ (assigning task $T_{i}$ to node $N_{j}$) proposed by the DRL
agent. This knowledge-guided reward, $R_{\text{rag}}$, is designed to
favor actions that align with the successful strategies observed in
similar past scenarios.

Specifically, we calculate a "wisdom score," $W\left( a_{t} \right)$,
for the proposed action $a_{t}$. This score is the weighted average of
the outcomes of the retrieved cases where a similar action was taken.
Let $\mathcal{C' \subseteq}\mathcal{C}_{\text{retrieved}}$ be the subset
of cases where task $T_{i}$ (or a topologically equivalent task) was
also assigned to node $N_{j}$. The wisdom score is then:

$$W\left( a_{t} \right) = \frac{\sum_{c\mathcal{\in C'}}^{}\text{sim}\left( E_{\text{current}},c.E_{\text{dag}} \right) \times \left( 1/c.M_{\text{final}} \right)}{\sum_{c\mathcal{\in C'}}^{}\text{sim}\left( E_{\text{current}},c.E_{\text{dag}} \right)}$$

If no such cases exist in the retrieved set ($\mathcal{C'}$ is empty),
$W\left( a_{t} \right)$ is zero. The final RAG reward $R_{\text{rag}}$
is then directly proportional to this score. This reward is given to the
agent immediately after its action, providing a dense and informative
learning signal that is grounded in historical evidence. This mechanism
not only accelerates training but also makes the agent's emergent policy
inherently more explainable.

# Experiments {#experiments .Head1}

## Experimental Platform {#experimental-platform .Head2}

Our experimental evaluation follows a \"Sim-to-Real\" methodology to
ensure that our findings are both rigorously tested and practically
relevant. This involves an offline training phase in a high-fidelity
simulator and an online inference and evaluation phase on a physical
hardware cluster.

Physical Cluster: The online evaluation was conducted on a physical HPC
cluster consisting of 16 compute nodes. Each node is equipped with an
Intel Xeon Gold 6132 CPU (14 cores, 2.6 GHz) and 128 GB of RAM. The
nodes are interconnected via a 10 Gbps Ethernet network. The cluster\'s
resource and job management are handled by the Slurm scheduler (version
22.05). This physical setup serves as the ground truth for validating
the real-world performance of our framework.

Simulation Environment: The DRL agent\'s extensive offline training was
performed in the WRENCH simulation framework. WRENCH is a validated,
high-fidelity simulator built atop SimGrid, widely used in the HPC
scheduling research community for its accuracy in modeling computation
and data movement. We meticulously configured the simulator\'s platform
file---including node processing power, network topology, bandwidth, and
latency---to precisely mirror the specifications of our physical
cluster, ensuring a high degree of correspondence between the simulated
and real-world environments.

## Workloads {#workloads .Head2}

To comprehensively evaluate the performance of the scheduling algorithms
across a range of complexities, we used synthetic workflow DAGs
generated by a standard workflow generator. This is a common practice in
the scheduling literature, allowing for controlled and reproducible
experiments. We generated four sets of workflows with varying scales,
containing 10, 20, 49, and 100 tasks, respectively. These workflows
feature diverse topological structures, including variations in task
dependencies and parallelism, which allows us to test the
generalizability and robustness of the schedulers under different
conditions. The computational cost of each task and the data size of
each dependency edge were assigned based on realistic distributions
found in scientific applications.

## Baseline Methods {#baseline-methods .Head2}

To rigorously evaluate the performance of our proposed framework, we
compare WASS-RAG against four baseline scheduling methods, ranging from
standard production schedulers to state-of-the-art heuristic and
learning-based approaches.

-   FIFO (First-In, First-Out): This serves as our primary baseline,
    representing the default scheduling policy in many production
    systems, including Slurm. It processes tasks in the order they
    become available, without considering workflow topology or task
    heterogeneity. Performance improvements over FIFO directly reflect
    the practical value of our approach in a real-world setting.

-   HEFT (Heterogeneous Earliest Finish Time): HEFT is a widely
    recognized, high-performance static list-scheduling heuristic
    algorithm. It prioritizes tasks based on their upward rank and
    schedules them on the compute node that offers the earliest finish
    time. It is considered a strong and classic baseline in academic
    literature for workflow scheduling.

-   WASS (Heuristic): This is a custom heuristic scheduler developed as
    part of our work. It incorporates domain-specific strategies for
    workflow scheduling, serving as an intermediate step between
    traditional methods and our full AI-based approach.

-   WASS-DRL (w/o RAG): This baseline is crucial for our ablation study.
    It is a version of our framework that uses the same GNN-based DRL
    agent but without the enhancement of the Knowledge-Guided Teacher.
    By comparing WASS-RAG directly against WASS-DRL, we can isolate and
    precisely quantify the performance contribution of the novel RAG
    mechanism.

## Evaluation Metrics {#evaluation-metrics .Head2}

We assess the quality of the scheduling strategies using three key
performance indicators, which provide a holistic view of the system's
efficiency.

-   Makespan: This is the primary metric for evaluating scheduling
    performance. It is defined as the total time elapsed from the start
    of the first task to the completion of the last task in a workflow
    ($Makespan = \max_{i \in V}\left( \text{finish\_time}\left( T_{i} \right) \right) - \min_{i \in V}\left( \text{start\_time}\left( T_{i} \right) \right)$).
    A lower makespan signifies a more efficient schedule and is our main
    optimization objective.

-   CPU Utilization: This metric reflects the overall efficiency of
    resource usage across the cluster. It is calculated as the average
    utilization of all CPU cores during the entire workflow makespan.
    Higher utilization indicates that the scheduler is effective at
    keeping the cluster's computational resources busy.

-   Data Locality: This metric measures the percentage of tasks that are
    scheduled to run on the same node where their required input data is
    located. Higher data locality reduces network overhead from data
    transfers, which can significantly impact performance, especially
    for data-intensive workflows.

# Results and Analysis {#results-and-analysis .Head1}

## Overall Performance Comparison {#overall-performance-comparison .Head2}

We conducted a comprehensive set of experiments to evaluate the
performance of WASS-RAG against the four baseline methods. The overall
results, aggregated across all workflow and cluster configurations, are
summarized in Table 2.

> Table 2: Overall comparison of scheduling methods across all
> experiments

  ------------------------------------------------------------------------
  Method              Makespan     Improvement   CPU Util    Data Locality
  ------------------- ------------ ------------- ----------- -------------
  FIFO                155.95       0%            51.3%       50.0%

  HEFT                116.8        25.1%         62.3%       70.0%

  WASS (Heuristic)    97.02        37.8%         56.0%       67.1%

  WASS-DRL (w/o RAG)  89.05        42.9%         57.4%       72.2%

  WASS-RAG            79.41        49.1%         59.1%       78.5%
  ------------------------------------------------------------------------

The results clearly demonstrate the superior performance of the WASS-RAG
framework. On the primary metric of makespan, WASS-RAG achieves a
remarkable 49.1% reduction compared to the standard FIFO scheduler used
in production systems, and a significant 24.0% reduction over the strong
HEFT heuristic. This confirms the effectiveness of our AI-driven
approach in optimizing complex workflow executions. Furthermore, the
table highlights a progressive performance improvement from the simpler
heuristics to the more advanced DRL-based methods, with WASS-RAG
standing out as the top performer. Notably, it also achieves the highest
data locality (78.5%), indicating its effectiveness in minimizing data
movement overhead.

## Performance Across Workloads: A Heatmap Analysis {#performance-across-workloads-a-heatmap-analysis .Head2}

To gain a deeper understanding of where WASS-RAG excels, we analyzed its
performance across different scales of workloads and cluster sizes.
Figure 2 presents a heatmap that visualizes the percentage improvement
in makespan of WASS-RAG relative to the HEFT baseline under each
specific configuration.![](media/image2.png){width="4.10426290463692in"
height="2.2531266404199477in"}

-   Figure 2: Heatmap of Makespan Improvement (%) of WASS-RAG over HEFT

The heatmap reveals a clear trend: the advantage of WASS-RAG becomes
increasingly pronounced in more complex scenarios. For instance, in the
most challenging configuration with 100 tasks and a resource-constrained
4-node cluster, WASS-RAG shows its maximum performance gain. This is
because, in such high-contention environments, naive or simple
heuristic-based decisions can easily lead to suboptimal resource
allocation and prolonged queuing times. In contrast, WASS-RAG\'s ability
to learn from and leverage a vast knowledge base of past experiences
allows it to make more sophisticated and far-sighted decisions,
effectively navigating the complex scheduling trade-offs to find a
near-optimal solution. This result strongly suggests that our
knowledge-guided approach is particularly well-suited for large-scale,
demanding HPC workloads.

## Architectural Overhead Analysis {#architectural-overhead-analysis .Head2}

WASS-RAG maintains the lightweight properties of the architecture.
Plugin Inference Time: The average time for the Lua plugin to perform
inference remains negligible at 1.8 milliseconds. Client-side Encoding
Time: The online, pre-submission step, which includes GNN encoding and
knowledge base retrieval, averaged 185 milliseconds for the 49-task
workflow. This sub-second, one-time cost is entirely practical for
production use.

## Explainable AI in Action: A Decision Walkthrough {#explainable-ai-in-action-a-decision-walkthrough .Head2}

A key contribution of WASS-RAG is its interpretability. To demonstrate
this, we queried the system about a non-intuitive decision in the
genomics workflow. The system explained its choice by presenting the
Top-3 most similar historical cases from its knowledge base . The
retrieved cases clearly showed that workflows with a similar DAG
topology consistently achieved better makespans when they avoided
resource contention on the primary node post-parallelization . This
ability to ground a decision in historical, empirical evidence is a
crucial step towards building trustworthy AI schedulers.

## Sim-to-Real Validation: From Simulation to a Physical Cluster {#sim-to-real-validation-from-simulation-to-a-physical-cluster .Head2}

The ultimate test is transferring the simulation-trained policy to the
real world. The WASS-RAG agent, trained exclusively in the WRENCH
simulator, was deployed on the physical Slurm cluster. The results were
remarkable. On the physical hardware, WASS-RAG completed the genomics
pipeline in an average of 475 seconds. This represents a 38.5% makespan
reduction compared to the physical baseline of Traditional Slurm (772s),
and a 18.0% makespan reduction compared to the physical performance of
the heuristic WASS (579s).

The strong correlation between simulated and real-world performance
gains provides powerful evidence that WASS-RAG offers a viable and
effective Sim-to-Real pathway for deploying advanced, knowledge-driven
AI schedulers in production HPC systems .

# Limitations {#limitations .Head1}

While WASS-RAG demonstrates significant promise, we acknowledge several
limitations that provide important context for our results and open
avenues for future work.

-   Scale of Physical Validation: Our Sim-to-Real experiments were
    conducted on a two-node heterogeneous cluster. While these
    experiments successfully validated the core thesis that a
    simulation-trained policy can be effectively transferred to real
    hardware, the limited scale of the physical deployment means that
    potential real-world challenges at a larger scale (e.g., the impact
    of complex network topologies, filesystem contention) are not fully
    captured. Validating WASS-RAG\'s performance on a larger-scale
    physical infrastructure remains an important next step.

-   Knowledge Base Dynamics and Scalability: The performance of the
    \"Knowledgeable Teacher\" is contingent on the quality and scale of
    the Knowledge Base. This introduces two potential limitations:

    -   Retrieval Latency: As the knowledge base grows to millions of
        entries, the vector retrieval latency, although highly optimized
        by FAISS, may become a non-negligible part of the client-side
        pre-submission overhead.

    -   Generalization to Novel Workflows: The agent\'s ability to
        handle entirely novel workflow structures, for which no similar
        historical cases exist in the knowledge base, has not been
        exhaustively tested. In such \"out-of-distribution\" scenarios,
        the RAG mechanism might struggle to provide a useful reward
        signal, potentially causing the agent to fall back to a less
        optimal, standard DRL policy.

```{=html}
<!-- -->
```
-   Simulation Fidelity: Although we chose a high-fidelity simulator
    (WRENCH) and our Sim-to-Real results show a strong correlation, no
    simulation can perfectly replicate the stochastic and chaotic nature
    of a real-world, multi-tenant HPC environment. Factors such as
    intermittent hardware failures, network jitter, and interference
    from other users\' jobs are not fully modeled, and could affect the
    performance of the deployed policy.

# Conclusion and Future Work {#conclusion-and-future-work .Head1}

## Conclusion {#conclusion .Head2}

This paper introduced WASS-RAG, a novel intelligent scheduling framework
that successfully bridges the critical "deployment gap" between advanced
machine learning research and production HPC systems. Our core
contribution is a new architectural paradigm that enables the safe and
efficient deployment of a sophisticated, learned scheduling policy
within a mature, production-level scheduler like Slurm. By leveraging a
hybrid "client-side training/encoding + server-side lightweight
inference" model, WASS-RAG augments the existing infrastructure rather
than replacing it, providing a practical pathway for infusing AI into
production environments.

The key innovation of WASS-RAG is the integration of a
Retrieval-Augmented Generation (RAG) paradigm into the DRL agent. This
transforms the scheduler from a black-box optimizer into a
knowledge-driven decision-maker. By learning from a "Knowledge-Guided
Teacher" that reasons based on a knowledge base of historical
executions, our agent develops policies that are not only highly
performant---reducing makespan by up to 48.2%---but also trustworthy and
explainable. As demonstrated, scheduling decisions can be traced back to
concrete, similar past cases, a critical feature for adoption in
mission-critical environments.

Ultimately, WASS-RAG represents a tangible step towards the future of
self-optimizing HPC systems. It offers a blueprint for how to safely and
effectively integrate advanced AI that is not only intelligent, but also
interpretable and capable of evolving with experience, into the critical
path of scientific computing.

## Future Work {#future-work .Head2}

The WASS-RAG framework opens up several exciting avenues for future
research, moving towards a truly autonomous scheduling system:

-   **Advanced Retrieval and Knowledge Curation**: Future work can
    explore more sophisticated retrieval algorithms beyond simple vector
    similarity to better capture the nuances of workflow structures.
    Additionally, developing automated methods for curating the
    knowledge base---such as identifying and pruning outdated or
    low-quality historical records---will be crucial for maintaining the
    long-term effectiveness of the \"Knowledge-Guided Teacher.\"

-   **Multi-Objective RAG Framework**: The current system optimizes for
    a single objective: makespan. A significant extension would be to
    evolve the RAG framework to handle multi-objective optimization.
    This would involve training a performance predictor capable of
    estimating outcomes across multiple dimensions (e.g., makespan,
    energy consumption, cost) and retrieving historical cases based on a
    composite score, enabling the agent to learn complex trade-off
    policies.

-   **Online Knowledge Integration and Continual Learning**: While the
    knowledge base allows the system to evolve, the current agent is
    trained offline. The next frontier is to enable true continual
    learning, where the agent can be fine-tuned in an online fashion
    using live performance data from the cluster. This would involve
    developing safe and efficient methods to update the policy and value
    networks without disrupting the production scheduling environment.

REFERENCES

1.  Ahn, D., Garlick, J., and Springmeyer, B. 2021. Flux: A
    Full-Featured Resource Manager for the Exascale Era.  

2.  *LLNL Computing*. Retrieved from
    https://computing.llnl.gov/projects/flux-building-framework-resource-management

3.  Deelman, E., Vahi, K., Juve, G., Rynge, M., Callaghan, S.,
    Maechling, P. J., Sakellariou, R., and Liew, C. S. 2015. Pegasus, a
    workflow management system for science automation.  

4.  *Future Generation Computer Systems* 46 (May 2015), 17--35.
    https://doi.org/10.1016/j.future.2014.10.008

5.  Di Tommaso, P., Chatzou, M., Floden, E. W., Barja, P. P., Palumbo,
    E., and Notredame, C. 2017. Nextflow enables reproducible
    computational workflows.  

6.  *Nature Biotechnology* 35, 4 (April 2017), 316--319.
    https://doi.org/10.1038/nbt.3820

7.  E4 Company. 2025. Future HPC & AI Predictions 2025.  

8.  *E4 Company Blog*. Retrieved from
    https://www.e4company.com/en/2025/01/future-hpc-ai-predictions/  

9.  Koster, J. and Rahmann, S. 2012. Snakemake---a scalable
    bioinformatics workflow engine.  

10. *Bioinformatics* 28, 19 (October 2012), 2520--2522.
    https://doi.org/10.1093/bioinformatics/bts480

11. Kulagina, S., Meyerhenke, H., and Benoit, A. 2024. Mapping Large
    Memory-Constrained Workflows onto Heterogeneous Platforms. In  

12. *Proceedings of the 38th IEEE International Parallel and Distributed
    Processing Symposium (IPDPS \'24)*. IEEE, 456--465.
    https://doi.org/10.1109/IPDPS59217.2024.00051

13. Kwok, Y.-K. and Ahmad, I. 1999. Static scheduling algorithms for
    allocating directed task graphs to multiprocessors.  

14. *ACM Computing Surveys* 31, 4 (Dec. 1999), 406--471.
    https://doi.org/10.1145/344588.344618

15. Lakkaraju, K., Valluru, S. L., and Srivastava, B. 2025. Holistic
    Explainable AI (H-XAI): Extending Transparency Beyond Developers in
    AI-Driven Decision Making.  

16. arXiv preprint arXiv:2508.05792.  

17. Lehmann, F., Bader, J., Tschirpke, F., De Mecquenem, N., Lößer, A.,
    Becker, S., Lewińska, K. E., Thamsen, L., and Leser, U. 2025. WOW:
    Workflow-Aware Data Movement and Task Scheduling for Dynamic
    Scientific Workflows. In  

18. Proceedings of the 25th IEEE/ACM International Symposium on Cluster,
    Cloud and Internet Computing (CCGrid \'25). (Forthcoming).

19. Mao, H., Medidi, M., and Vemulapalli, M. K. 2021. Deep Reinforcement
    Agent for Scheduling in HPC.  

20. arXiv preprint arXiv:2106.13803.  

21. Quaedvlieg, L. C. P. M. 2025. Optimizing Job Allocation using
    Reinforcement Learning with Graph Neural Networks.  

22. arXiv preprint arXiv:2501.19063.  

23. Sharma, C. 2025. Retrieval-Augmented Generation: A Comprehensive
    Survey of Architectures, Enhancements, and Robustness Frontiers.  

24. arXiv preprint arXiv:2506.00054.  

25. Sun, H., Wang, Y., Zhang, Z., and Li, K. 2025. A Comprehensive
    Survey on Deep Reinforcement Learning for Job Scheduling and
    Resource Management in Cloud Computing.  

26. arXiv preprint arXiv:2501.01007.  

27. Topcuoglu, H., Hariri, S., and Wu, M.-Y. 2002. Performance-effective
    and low-complexity task scheduling for heterogeneous computing.  

28. *IEEE Transactions on Parallel and Distributed Systems* 13, 3 (March
    2002), 260--274. https://doi.org/10.1109/71.993206

[^1]: Funding Project: Supported by the Lenovo University-Enterprise
    Cooperation Project, Project Number: 202405SJTU01-BUBG017
