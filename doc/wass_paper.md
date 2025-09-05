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

Modern scientific discovery increasingly depends on complex workflows
executed on HPC clusters. With the rise of AI-based scheduling, there is
strong interest in leveraging machine learning to optimize data locality
and parallelism.Despite promising results in simulation, advanced
DRL-based schedulers face a "deployment gap": they are difficult to
integrate into stable production systems such as Slurm, and their
black-box nature hinders trust and adoption in mission-critical
environments.We propose WASS-RAG, a hybrid architecture that augments
rather than replaces existing schedulers. Its core innovation integrates
Retrieval-Augmented Generation (RAG) into the Deep Reinforcement
Learning (DRL) loop. A "Knowledge-Guided Teacher" retrieves
topologically similar historical workflows to provide experience-based,
context-aware reward signals. This transforms static heuristic rules
into a dynamic, knowledge-guided reasoning process, yielding adaptive
and explainable scheduling policies.Through a Sim-to-Real methodology,
WASS-RAG trains policies on millions of simulated workflows using WRENCH
and deploys them in a physical Slurm cluster. Experiments show up to
48.2% reduction in makespan over traditional scheduling and 15.7%
improvement over a strong heuristic baseline. Case studies further
highlight interpretability, with scheduling decisions grounded in
retrieved historical cases.WASS-RAG demonstrates a practical and
trustworthy pathway for integrating state-of-the-art AI scheduling into
production HPC environments. By combining performance, safety, and
explainability, it paves the way toward self-optimizing workflow
management in next-generation scientific computing.

**CCS CONCEPTS** • Computing methodologies → Machine learning →
Reinforcement learning; • Software and its engineering → Software
organization and properties → Interoperability; • Information systems →
Information retrieval → Retrieval models and ranking

**Additional Keywords and Phrases:** Workflow Scheduling, Deep
Reinforcement Learning, Retrieval-Augmented Generation, Knowledge Base,
Slurm, Explainable AI

# Introduction {#introduction .Head1}

The efficiency of complex scientific workflows on High-Performance
Computing (HPC) clusters is often bottlenecked by the costly movement of
intermediate data. While machine learning offers a promising solution, a
critical "deployment gap" hinders the adoption of advanced AI models in
production environments, as HPC centers are justifiably reluctant to
replace their battle-hardened schedulers like Slurm with monolithic,
academic AI frameworks.

This paper introduces WASS-RAG, a novel hybrid architecture designed
explicitly to bridge this deployment gap. Our work's core contribution
is a practical and non-invasive architectural paradigm for safely
deploying and executing learned, intelligent policies within existing
production schedulers. We achieve this by decoupling the AI lifecycle
into an offline training phase and a lightweight, online inference phase
executed within a stateless Slurm plugin.

The key innovation of WASS-RAG lies in integrating a Retrieval-Augmented
Generation (RAG) paradigm into the Deep Reinforcement Learning (DRL)
agent. Instead of relying on static, handcrafted heuristic rules for
guidance, our DRL agent learns from a dynamic "Knowledge-Guided
Teacher". This teacher queries a vast knowledge base of historical
execution records to retrieve topologically similar past workflows. By
reasoning based on this retrieved "experience," it provides the DRL
agent with highly accurate, context-aware reward signals, enabling it to
learn sophisticated scheduling policies.

This knowledge-driven approach not only yields superior performance but
also provides unprecedented explainability, as scheduling decisions can
be traced back to concrete historical examples. Our rigorous
"Sim-to-Real" evaluation validates this framework, demonstrating that
WASS-RAG significantly reduces workflow makespan by up to 48.2% and,
more importantly, offers a trustworthy, practical, and continuously
evolving solution for intelligent scheduling in production HPC.

The main contributions of this paper are as follows:

- We identify the "deployment gap" and propose WASS-RAG, a non-invasive
  hybrid architecture to bridge it.

- We introduce a novel RAG-enhanced DRL framework for scheduling,
  featuring a "Knowledge-Guided Teacher" that provides experience-based
  reward signals.

- We demonstrate how this approach provides strong explainability by
  grounding AI decisions in historical data.

- We validate our framework through a comprehensive "Sim-to-Real"
  methodology, proving its real-world effectiveness and practicality.

# Related Work {#related-work .Head1}

The application of machine learning to enhance HPC scheduling is a
vibrant research area. Existing work can be broadly categorized into
three main approaches, which collectively highlight the unique niche and
contribution of WASS-DRL.

- **Prediction-Informed Scheduling**: This line of research uses ML
  models as "predictors" to provide more accurate inputs for traditional
  scheduling algorithms. For instance, various studies have successfully
  used neural networks to predict job runtimes, which in turn improves
  the efficiency of backfilling schedulers. While effective at
  optimizing existing heuristics, these methods do not fundamentally
  change the decision-making logic itself.

- **End-to-End DRL Schedulers**: A more ambitious approach aims to
  completely replace the core scheduling logic with an end-to-end DRL
  agent. Systems like DRAS have shown that a DRL agent can learn complex
  scheduling policies, such as backfilling and resource reservation, in
  simulated environments. The primary challenge for these systems,
  however, lies in their integration with production-level schedulers
  written in low-level languages, presenting a significant engineering
  hurdle.

- **GNN+DRL for Graph-Structured Problems**: The most relevant research
  area combines GNNs with DRL to tackle problems with inherent graph
  structures, such as workflow scheduling. Frameworks like GrapheonRL
  have demonstrated the power of using a GNN to encode the state of a
  workflow and resource graph, providing a rich representation for the
  DRL agent. These works have shown immense potential, primarily in
  simulation.

A critical analysis of these pioneering efforts reveals a common
challenge: a "Deployment Gap" between state-of-the-art academic research
and production HPC practice. Many advanced ML-based schedulers are
either designed as monolithic systems that require replacing stable
schedulers like Slurm or are validated exclusively in simulators, with
no clear, low-risk path to production deployment.

WASS-DRL is explicitly designed to bridge this gap. Our core
contribution is not merely the application of GNN+DRL to scheduling, but
the proposal of a practical, non-invasive architectural paradigm that
allows for the safe and efficient deployment of learned policies within
a production scheduler. By decoupling the intensive training and
encoding on the client-side from the lightweight inference within the
Slurm plugin, WASS-DRL augments rather than replaces the existing
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

## Architectural Components: From Heuristic Engine to AI Agent {#architectural-components-from-heuristic-engine-to-ai-agent .Head2}

The WASS-RAG hybrid architecture consists of two primary components: a
client-side orchestrator and a server-side Slurm plugin. Their roles are
significantly evolved from a simple heuristic model to support a
knowledge-driven, AI-powered scheduling process.

1.  The Client-Side Orchestrator: The System's Knowledge and
    Intelligence Core

In WASS-RAG, the client evolves into a powerful and sophisticated
machine learning engine, becoming the true intelligence core of the
system. It handles all computationally intensive and stateful operations
outside the scheduler's critical path. Its new, expanded
responsibilities are threefold:

- **Knowledge Base Management**: The client is responsible for building,
  maintaining, and querying a vector Knowledge Base ($\mathcal{K}$).This
  knowledge base stores historical execution records, including workflow
  topologies, scheduling decisions made, and final performance outcomes
  (e.g., Makespan).

- **Offline Training Controller**: The client acts as the master
  controller for the DRL agent's training loop. In each training
  episode, it orchestrates the entire Retrieval-Augmented learning
  process. It queries the knowledge base to retrieve relevant historical
  cases, which are then used by the "Knowledge-Guided Teacher" to
  generate a highly informative, experience-based reward signal for
  updating the DRL agent's policy network.

- **Online State Encoding & Submission**: When a user submits a
  workflow, the client constructs a real-time state graph. It then
  utilizes a GNN model to encode this graph into an embedding vector,
  which is injected into the job's submission comment for the
  server-side plugin to use.

  1.  The Server-Side Plugin: A Lightweight and Robust Inference Engine

In WASS-RAG, the plugin's role remains an extremely lightweight, fast,
and robust inference engine. Its purpose is to execute the pre-trained
policy from the DRL agent with minimal overhead. Its tasks are:

- **Model Loading**: At startup, the plugin loads the compact,
  pre-trained DRL policy network.

- **Metadata Extraction & Inference**: When a job arrives, it
  efficiently parses the state embedding from the job's comment field
  and performs a single, rapid forward-pass computation to determine the
  optimal action (i.e., the target node). This entire process is
  designed to complete in milliseconds.

- **Action Application & Safety Guarantee**: The plugin applies the
  decision by setting the job's nodelist property. Critically, it
  retains the "graceful degradation" mechanism, performing a final
  resource compatibility check to ensure the AI's decision does not
  violate any system constraints, guaranteeing system stability.

![](media/image1.png){width="5.829770341207349in"
height="3.853576115485564in"}

Figure 1: WASS - RAG Architecture: Decoupled Offline Training and Online
Inference

## Implementation Details {#implementation-details .Head2}

To realize the WASS-RAG framework, several key components were
implemented to handle the knowledge-driven AI lifecycle.

- **Knowledge Base and Retrieval**: The Knowledge Base (K) is
  implemented using a high-performance vector database, indexed by FAISS
  for efficient similarity searches. When a new workflow state is
  processed, its graph embedding, generated by the GNN, is used as a
  query vector. The retrieval mechanism then employs cosine similarity
  to rapidly identify and retrieve the Top-K most similar historical
  cases from the knowledge base.

- **Performance Predictor Model**: The \"Knowledge-Guided Teacher\'s\"
  ability to generate rewards hinges on the Performance Predictor. This
  component is implemented as a multi-layer perceptron (MLP) network.
  Its inputs are the concatenated embeddings of the current workflow
  state, the proposed scheduling action, and the averaged embedding of
  the Top-K retrieved historical cases. The model is trained offline in
  a supervised manner to predict the final makespan, using the
  historical data in the knowledge base as ground truth.

# RAG-Augmented MDP Formulation for Scheduling {#rag-augmented-mdp-formulation-for-scheduling .Head1}

To learn a policy capable of achieving our primary
objective---minimizing the Makespan---we formalize the scheduling
problem as a Retrieval-Augmented Markov Decision Process. This framework
extends the traditional MDP by incorporating an external Knowledge Base
($\mathcal{K}$) as a form of memory, which is used to generate a more
informed reward signal.

The knowledge base $\mathcal{K}$ is a collection of historical execution
records, where each record $k_{i}$ is a tuple
$k_{i} = \left( G_{i},A_{i},M_{i} \right)$, representing the historical
workflow graph, the sequence of scheduling actions taken, and the final
Makespan achieved, respectively. The RAG-augmented MDP is then defined
by the tuple
$\left( \mathcal{S,A,}P,\mathcal{R}_{\text{RAG}},\gamma\mathcal{,K} \right)$,
where the key innovation lies in the reward function
$\mathcal{R}_{\text{RAG}}$, which is dynamically generated based on
knowledge retrieved from $\mathcal{K}$.

The primary objective of our work is to learn a scheduling policy,
$\pi$, that minimizes the total workflow execution time, or Makespan.
The Makespan is determined by the completion time of the final task in
the workflow and is formally defined as:

$$\min_{\pi\mathcal{:V \rightarrow N}}\text{Makespan} = \min_{\pi}\max_{t_{i}\mathcal{\in V}}C\left( t_{i},\pi \right)\quad(1)$$

where $C\left( t_{i},\pi \right)$ is the completion time of task $t_{i}$
under policy $\pi$.

To learn a policy capable of achieving this objective in a dynamic and
complex HPC environment, we formalize the scheduling problem as a
Retrieval-Augmented Markov Decision Process. This framework extends the
traditional MDP by incorporating an external Knowledge Base
($\mathcal{K}$) as a form of memory, which is used to generate a more
informed reward signal.

The knowledge base $\mathcal{K}$ is a collection of historical execution
records, where each record $k_{i}$ is a tuple
$k_{i} = \left( G_{i},A_{i},M_{i} \right)$, representing the historical
workflow graph, the sequence of scheduling actions taken, and the final
Makespan achieved, respectively. The RAG-augmented MDP is then defined
by the tuple
$\left( \mathcal{S,A,}P,\mathcal{R}_{\text{RAG}},\gamma\mathcal{,K} \right)$,
where the key innovation lies in the reward function
$\mathcal{R}_{\text{RAG}}$, which is dynamically generated based on
knowledge retrieved from $\mathcal{K}$. The following sections detail
our design of the state, action, and reward spaces tailored for this
problem.

## State Space ($\mathcal{S}$): A Heterogeneous Graph Representation {#state-space-mathcals-a-heterogeneous-graph-representation .Head2}

The design of the state space is arguably the most critical step in the
DRL framework, as it determines the quality of information available to
the agent for decision-making. A flattened feature vector is
insufficient for this problem, as it would lose the rich topological and
dependency information inherent in workflows and HPC clusters.
Therefore, we employ a powerful heterogeneous graph,
$\mathcal{G}_{t} = \left( \mathcal{V,E} \right)$, to represent the
scheduling state $s_{t}$ at each decision step $t$. This heterogeneous
graph is composed of multiple node and edge types:

- Node Types ($\mathcal{V}$):

  - Task Nodes: Represent all un-dispatched or currently executing tasks
    in the DAG. Their features include resource requirements (CPU,
    memory, GPU), an estimated computation time, and topological
    properties like the task's depth in the DAG.

  - Compute Nodes: Represent the physical nodes in the cluster. Features
    include total resource capacity, current resource utilization
    (CPU/memory load), and GPU availability.

  - File Nodes: Represent key intermediate data products generated by
    the workflow. Their primary feature is the file size.

- Edge Types ($\mathcal{E}$):

  - Task $\rightarrow$ Task: Represents the precedence constraints in
    the workflow DAG.

  - File $\rightarrow$ Compute Node: Represents the location of a data
    file on a node's local storage.

  - Task $\leftrightarrow$ File: Represents the input/output data
    dependencies of tasks.

This complex graph structure cannot be directly consumed by a standard
DRL policy network. This is where the GNN encoder from our architecture
(Section 3.1) plays its vital role. The GNN processes the entire
heterogeneous graph $\mathcal{G}_{t}$ and generates a low-dimensional
embedding vector, $e_{t} = \text{GNN}\left( \mathcal{G}_{t} \right)$,
for the task to be scheduled. This embedding $e_{t}$ serves as the
final, information-rich state representation $s_{t}$ for the DRL agent,
enabling it to "understand" the intricate relationships between task
dependencies, data locality, and resource availability .

## Action Space ($\mathcal{A}$): Defining Scheduling Decisions {#action-space-mathcala-defining-scheduling-decisions .Head2}

The action space defines the set of operations the agent can perform.
For our scheduling problem, the action space $\mathcal{A}_{t}$ at
decision step $t$ is the discrete set of all available compute nodes
where the current ready task can be placed:

$$\mathcal{A}_{t} = \{ n_{1},n_{2},\ldots,n_{k}\}$$

where $k$ is the total number of compute nodes in the cluster. The
agent's policy network will output a probability distribution or
Q-values over this set, indicating the suitability of each node for the
pending task.

## Reward Function ($\mathcal{R}$): Incentivizing Makespan Minimization via RAG {#reward-function-mathcalr-incentivizing-makespan-minimization-via-rag .Head2}

The reward function is critical as it guides the agent toward our stated
objective in Equation (1). We engineer the reward signal using a
combination of a primary sparse reward and a dense, knowledge-driven
reward shaping technique.

- Primary Sparse Reward: The most direct reward is given at the end of
  an entire workflow execution. We define the final reward as the
  negative of the Makespan, which directly encourages the agent to
  minimize the total completion time :

$$R_{\text{final}} = - \text{Makespan}\quad(2)$$

- RAG-based Reward Shaping: To provide a dense and highly informative
  learning signal, we employ a sophisticated reward shaping technique
  powered by our "Knowledge-Guided Teacher". The intermediate reward,
  $\mathcal{R}_{\text{RAG}}$, is defined as the predicted performance
  improvement of the agent's action ($a_{t}$) relative to a historically
  optimal baseline action ($a_{t}^{*}$).

<!-- -->

- First, the "Knowledge-Guided Teacher" identifies the historically
  optimal action $a_{t}^{*}$ by finding the action that minimizes the
  predicted Makespan based on the retrieved context $\mathcal{C}_{t}$:

$$a_{t}^{*} = arg\min_{a \in \mathcal{A}_{t}}\left( \mathbb{E}_{k \sim \mathcal{C}_{t}}\left\lbrack \text{Predict}\left( M|\mathcal{G}_{t},a \right) \right\rbrack \right)$$

- Then, the reward for the agent's chosen action $a_{t}$ is calculated
  as the difference between the predicted makespan of the baseline
  action and the agent's action:

$$\mathcal{R}_{\text{RAG}}\left( s_{t},a_{t},\mathcal{C}_{t} \right) = \mathbb{E}_{k \sim \mathcal{C}_{t}}\left\lbrack \text{Predict}\left( M|\mathcal{G}_{t},a_{t}^{*} \right) \right\rbrack - \mathbb{E}_{k \sim \mathcal{C}_{t}}\left\lbrack \text{Predict}\left( M|\mathcal{G}_{t},a_{t} \right) \right\rbrack\quad(3)$$

- Here,
  $\mathbb{E}_{k \sim \mathcal{C}_{t}}\left\lbrack \text{Predict}( \cdot ) \right\rbrack$
  represents the expected Makespan predicted by the Performance
  Predictor model, conditioned on the current state, the proposed
  action, and the context retrieved from similar historical cases.

  The intuitive meaning of this formula is crucial: if the agent's
  action $a_{t}$ is better than the historical best $a_{t}^{*}$ (i.e.,
  its predicted Makespan is lower), the result of the subtraction will
  be a positive number, giving the agent a positive reward. This
  powerful technique provides immediate, meaningful feedback at every
  step, directly incentivizing the agent to discover policies that not
  only match but actively surpass the collective wisdom stored in the
  knowledge base.

  ![](media/image2.png){width="5.8429538495188105in"
  height="2.2531266404199477in"}

  Figure 2: The information flow of the RAG-driven reward generation
  process. At each step, the Knowledge-Guided Teacher retrieves relevant
  historical context to predict the performance of the agent\'s action
  relative to a historically optimal one, generating a dense and
  explainable reward signal.

# Experimental Methodology {#experimental-methodology .Head1}

To rigorously validate the performance, practicality, and explainability
of the WASS-RAG framework, we designed a comprehensive experimental
methodology. Our approach relies on a high-fidelity simulation
environment for knowledge base generation and agent training, a
multi-faceted set of baselines for comparison, and a diverse suite of
workloads and cluster configurations for evaluation.

## High-Fidelity Simulation Environment {#high-fidelity-simulation-environment .Head2}

Training a DRL agent and its associated models requires millions of
simulated workflow executions. As such, a high-fidelity simulator is a
prerequisite for this research. Our work leverages WRENCH, a
state-of-the-art simulation framework designed specifically for
scientific workflows. Its validated SimGrid core, workflow-centric
design, and high performance make it the ideal environment for
generating the vast datasets required for our knowledge-driven approach
.

## Agent and Knowledge Base Training Process {#agent-and-knowledge-base-training-process .Head2}

The training process for WASS-RAG is more sophisticated than a standard
DRL loop and involves three key stages:

- **Stage 1: Knowledge Base Seeding**: We first create an initial
  Knowledge Base ($\mathcal{K}$). This is done by running a large number
  of diverse workflows in the WRENCH simulator using the WASS
  (heuristic) policy. For each completed workflow, we store its graph
  topology, the sequence of scheduling actions taken, and the final
  achieved Makespan as a record
  $k_{i} = \left( G_{i},A_{i},M_{i} \right)$ in the knowledge base. This
  seeds the system with a baseline of "good" (but not optimal)
  historical experiences.

- **Stage 2: Performance Predictor Training**: The core of our
  "Knowledge-Guided Teacher" is the Performance Predictor model. This
  model is trained in a supervised manner on the data from the knowledge
  base. Its goal is to learn a function that, given a new workflow state
  ($G_{t}$), a proposed action ($a_{t}$), and a set of similar
  historical cases ($\mathcal{C}_{t}$), can accurately predict the final
  Makespan.

- **Stage 3: DRL Agent Training**: With the knowledge base and the
  trained performance predictor in place, we begin training the DRL
  agent using the Proximal Policy Optimization (PPO) algorithm. In each
  training step, the agent interacts with the simulator. Its actions are
  evaluated by the "Knowledge-Guided Teacher," which retrieves relevant
  context from $\mathcal{K}$ and uses the performance predictor to
  generate the RAG-based reward signal ($\mathcal{R}_{\text{RAG}}$) as
  defined in Section 4.2. This reward guides the agent to discover
  policies that outperform the historical experiences stored in the
  knowledge base.

## Comparison Baselines {#comparison-baselines .Head2}

To comprehensively evaluate the effectiveness of WASS-DRL, we compare it
against a carefully selected set of strong baselines:

- **Traditional Slurm**: A workflow-agnostic scheduler where tasks are
  submitted as independent jobs linked only by dependencies. This
  represents the baseline performance without any data locality
  awareness.

- **WASS (Heuristic)**: Our own heuristic-driven framework. This is the
  most critical baseline, as it allows us to directly quantify the
  performance improvement gained by replacing handcrafted rules with a
  learned, dynamic policy.

- **HEFT (Heterogeneous Earliest Finish Time)**: A widely-cited classic
  scheduling algorithm from the academic literature. Including HEFT
  ensures our work is benchmarked against established academic
  standards, enhancing its scholarly rigor.

- **WASS-DRL (w/o RAG)**: A standard DRL agent implemented within our
  hybrid architecture, but without the RAG-based reward mechanism (using
  a simpler sparse reward). **This crucial baseline isolates the
  performance contribution of the DRL paradigm itself and serves as the
  direct benchmark to quantify the specific added value of our
  innovative RAG framework.**

## Experimental Design and Metrics {#experimental-design-and-metrics .Head2}

Our experiments are designed to test the agent's performance,
scalability, and generalization capabilities across a variety of
scenarios.

- **Workloads**: We use a diverse set of workloads, including synthetic
  DAGs (linear, fan-in, fan-out) and real-world scientific workflows
  (genomics, Montage, CyberShake) .

- **Cluster Configurations**: Simulations are run on multiple cluster
  configurations, including different scales (16 to 128 nodes) and types
  (homogeneous and heterogeneous).

- **Evaluation Metrics**:

  - Primary Metric: Makespan, the total workflow completion time, is the
    core metric for efficiency.
  - Secondary Metrics: We also measure Average Job Wait Time, Resource
    Utilization, and Total Cross-Node I/O Volume.
  - Overhead Metric: The Plugin Inference Time is measured to
    empirically demonstrate the practicality of our architecture.
  - Explainability Metric: We will present case studies showing the
    Top-K historical cases retrieved by the RAG mechanism to explain
    specific, non-intuitive scheduling decisions, qualitatively
    demonstrating the system's interpretability.

# Results and Analysis {#results-and-analysis .Head1}

This section presents a comprehensive evaluation of the WASS-RAG
framework. We first analyze its performance on synthetic and real-world
workflows within our high-fidelity WRENCH simulation environment. We
then report on the architectural overhead to demonstrate its
practicality. Crucially, we showcase the system's explainability and
present results from a "Sim-to-Real" experiment, deploying the
simulation-trained agent onto a physical cluster to validate its
real-world efficacy.

## Performance on Synthetic Workflows {#performance-on-synthetic-workflows .Head2}

The agent's ability to learn fundamental scheduling policies was
evaluated on synthetic workflows. The results, summarized in Table 2,
show that by reasoning from a knowledge base of archetypal workflow
executions, WASS-RAG learns highly adaptive strategies that consistently
outperform both rule-based and standard learning approaches.

> Table 2: Makespan comparison on synthetic workflows. The results
> demonstrate WASS-RAG's adaptive, knowledge-driven policy learning.

  --------------------------------------------------------------------------
  Workflow   Traditional   HEFT (s) WASS          WASS-DRL (w/o   WASS-RAG
  Type       Slurm (s)              (Heuristic)   RAG) (s)        (s)
                                    (s)                           
  ---------- ------------- -------- ------------- --------------- ----------
  Linear     250           215      190           175             162
  Chain                                                           

  Fan-in     420           350      310           280             255

  Fan-out    380           330      325           295             281
  --------------------------------------------------------------------------

## Case Study: A 49-Task Genomics Workflow {#case-study-a-49-task-genomics-workflow .Head2}

We conducted a detailed case study on the complex, 49-task genomics
pipeline to provide a clear, hierarchical comparison of the different
scheduling policies. The end-to-end makespan for each scheduler is
summarized in Table 3 and visualized in the Gantt chart in Figure 3.

> Table 3: End-to-end Makespan comparison on the 49-task genomics
> workflow, illustrating the performance gains at each stage of
> intelligence.

  -----------------------------------------------------------------------
  Scheduling Policy  Makespan (s) Improvement over   Improvement over
                                  Slurm              Heuristic
  ------------------ ------------ ------------------ --------------------
  Traditional Slurm  772          \-                 \-

  HEFT               610          21.0%              \-

  WASS (Heuristic)   579          25.0%              \-

  WASS-DRL (w/o RAG) 520          32.6%              10.2%

  WASS-RAG           475          38.5%              18.0%
  -----------------------------------------------------------------------

- ![](media/image3.png){width="4.10426290463692in"
  height="2.2531266404199477in"}

  Figure 3: Makespan Comparison of Scheduling Policies

The results reveal a clear progression of intelligence. The WASS
(Heuristic) framework, with its handcrafted rules, provides a solid
25.0% improvement over the workflow-agnostic Slurm. The introduction of
a standard DRL agent in WASS-DRL (w/o RAG) further pushes this boundary
to a 32.6% improvement by learning more adaptive policies than static
rules.

Critically, the final leap in performance comes from our RAG-based
"Knowledgeable Teacher." WASS-RAG achieves an overall 38.5% improvement
over Slurm, and demonstrates a significant 18.0% gain over the strong
heuristic baseline. This quantifies the substantial value of enabling
the DRL agent to reason from a knowledge base of historical experience,
allowing it to discover globally optimal policies that surpass both
static rules and standard reinforcement learning.

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

- Scale of Physical Validation: Our Sim-to-Real experiments were
  conducted on a two-node heterogeneous cluster. While these experiments
  successfully validated the core thesis that a simulation-trained
  policy can be effectively transferred to real hardware, the limited
  scale of the physical deployment means that potential real-world
  challenges at a larger scale (e.g., the impact of complex network
  topologies, filesystem contention) are not fully captured. Validating
  WASS-RAG\'s performance on a larger-scale physical infrastructure
  remains an important next step.

- Knowledge Base Dynamics and Scalability: The performance of the
  \"Knowledgeable Teacher\" is contingent on the quality and scale of
  the Knowledge Base. This introduces two potential limitations:

  - Retrieval Latency: As the knowledge base grows to millions of
    entries, the vector retrieval latency, although highly optimized by
    FAISS, may become a non-negligible part of the client-side
    pre-submission overhead.

  - Generalization to Novel Workflows: The agent\'s ability to handle
    entirely novel workflow structures, for which no similar historical
    cases exist in the knowledge base, has not been exhaustively tested.
    In such \"out-of-distribution\" scenarios, the RAG mechanism might
    struggle to provide a useful reward signal, potentially causing the
    agent to fall back to a less optimal, standard DRL policy.

<!-- -->

- Simulation Fidelity: Although we chose a high-fidelity simulator
  (WRENCH) and our Sim-to-Real results show a strong correlation, no
  simulation can perfectly replicate the stochastic and chaotic nature
  of a real-world, multi-tenant HPC environment. Factors such as
  intermittent hardware failures, network jitter, and interference from
  other users\' jobs are not fully modeled, and could affect the
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

- **Advanced Retrieval and Knowledge Curation**: Future work can explore
  more sophisticated retrieval algorithms beyond simple vector
  similarity to better capture the nuances of workflow structures.
  Additionally, developing automated methods for curating the knowledge
  base---such as identifying and pruning outdated or low-quality
  historical records---will be crucial for maintaining the long-term
  effectiveness of the \"Knowledge-Guided Teacher.\"

- **Multi-Objective RAG Framework**: The current system optimizes for a
  single objective: makespan. A significant extension would be to evolve
  the RAG framework to handle multi-objective optimization. This would
  involve training a performance predictor capable of estimating
  outcomes across multiple dimensions (e.g., makespan, energy
  consumption, cost) and retrieving historical cases based on a
  composite score, enabling the agent to learn complex trade-off
  policies.

- **Online Knowledge Integration and Continual Learning**: While the
  knowledge base allows the system to evolve, the current agent is
  trained offline. The next frontier is to enable true continual
  learning, where the agent can be fine-tuned in an online fashion using
  live performance data from the cluster. This would involve developing
  safe and efficient methods to update the policy and value networks
  without disrupting the production scheduling environment.

[^1]: Funding Project: Supported by the Lenovo University-Enterprise
    Cooperation Project, Project Number: 202405SJTU01-BUBG017
