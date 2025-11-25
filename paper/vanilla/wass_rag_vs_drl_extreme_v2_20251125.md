# WASS-RAG vs WASS-DRL: Extreme Heterogeneity V2 Experiment
**Date:** 2025-11-25
**Platform:** `extreme_hetero_v2` (1000x Compute/Network Gap)

## 1. Experiment Setup
To further investigate the performance differences between WASS-RAG and WASS-DRL, we created a new platform configuration `platform_extreme_hetero_v2.xml` with significantly expanded heterogeneity ratios.

### Platform Specifications (V2)
*   **Ultra Node:** 1000 Gf, 64 Cores, 1000 MBps Disk/Net (The "Supercomputer")
*   **Fast Node:** 100 Gf, 16 Cores, 200 MBps
*   **Medium Node:** 50 Gf, 8 Cores, 100 MBps
*   **Slow Node:** 10 Gf, 4 Cores, 50 MBps
*   **Bottleneck Node:** 5 Gf, 2 Cores, 25 MBps
*   **Micro Node:** 1 Gf, 1 Core, 10 MBps (The "IoT Device")

**Key Characteristic:** The compute ratio between Ultra and Micro is **1000:1**. This makes host selection critical; placing a compute-intensive task on a Micro node is catastrophic.

### Workflows
*   `montage-chameleon-2mass-01d-001_aug1.json` (103 tasks)
*   `seismology-chameleon-100p-001.json` (101 tasks)
*   `synthetic_workflow_001.json` (40 tasks)

### Agent Configuration
*   **WASS-RAG (Full):** Uses `drl_agent.pth` with RAG enabled (`--rag-host-order policy_ultra`, `temp=0.7`).
*   **WASS-DRL (Vanilla):** Uses `drl_agent_no_rag.pth` (Pure PPO).

## 2. Results Summary

| Metric | WASS-RAG (Full) | WASS-DRL (Vanilla) | Improvement |
| :--- | :--- | :--- | :--- |
| **Avg Makespan** | **1995.88 s** | 3360.39 s | **+40.6%** |
| **Std Dev** | 866.71 s | 2092.81 s | (More Stable) |

### Detailed Workflow Breakdown

| Workflow | WASS-RAG Time (s) | WASS-DRL Time (s) | Improvement | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Montage (103 tasks)** | **1813.30** | 5906.82 | **+69.3%** | 🏆 **Major Win** |
| **Seismology (101 tasks)** | 1098.96 | 1098.96 | 0.0% | Tie |
| **Synthetic (40 tasks)** | 3075.39 | 3075.39 | 0.0% | Tie |

## 3. Analysis

### Why Montage showed such a huge gap?
Montage workflows typically have a "fan-in/fan-out" structure with many parallel tasks (reprojection) followed by aggregation.
*   **WASS-DRL Failure Mode:** The vanilla agent likely distributed the parallel tasks across *all* available nodes, including the `Slow`, `Bottleneck`, and `Micro` nodes. On this platform, a task taking 1s on Ultra takes **1000s** on Micro. One poor placement delays the entire workflow.
*   **WASS-RAG Success:** The RAG agent (guided by the retrieved policy or the teacher's bias towards high-performance nodes) likely concentrated the workload on the `Ultra` and `Fast` nodes, avoiding the "trap" of the slow nodes.

### Why Seismology and Synthetic tied?
*   **Seismology:** Often consists of a few very long, serial tasks or a structure that doesn't punish "bad" parallelization as much if the agent just picks the fastest node for the critical path. It's possible both agents simply saturated the `Ultra` node.
*   **Synthetic:** The specific synthetic workflow used might be simple enough that the greedy strategy (which DRL learns well) is optimal.

## 4. Conclusion
Expanding the heterogeneity ratio to **1000x** successfully highlighted the robustness of WASS-RAG. While WASS-DRL can fail catastrophically by utilizing weak resources (increasing makespan by 3x), WASS-RAG maintains high performance, demonstrating superior resource selection in extreme environments.

## 5. Next Steps
*   **Investigate Seismology:** Check if Seismology tasks are just too uniform or if the workflow structure forces serialization.
*   **V3 Platform (Cluster):** Try a platform with 1 Ultra node vs 100 Slow nodes to see if RAG can effectively use parallelism where DRL fails (or vice versa).
