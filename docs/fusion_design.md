# 融合决策模型设计 (R-2.3)

目标: 将 DRL 的全局策略价值估计与 RAG 的基于相似历史案例的经验推荐结合, 获得更鲁棒的调度决策。

## 1. 输入信号
- DRL: Q-values 向量 Q[nodes]
- RAG: 对应节点的相似度加权置信度 scores[nodes]
- 运行态上下文: 节点当前负载负向因子 load_penalty[nodes]

## 2. 归一化
- Q': (Q - mean(Q)) / (std(Q)+ε)
- S': scores / max(scores)
- L': 1 - normalized_load  (鼓励低负载)

## 3. 融合公式 (初稿)
`F = α * Q' + β * S' + γ * L'`
其中 α, β, γ 动态调节:
- 早期训练: α 较低 (鼓励利用案例)  -> α=0.4 β=0.4 γ=0.2
- 中后期: α 提升 (信任学到的策略) -> α=0.6 β=0.25 γ=0.15

调节依据: episode_progress ∈ [0,1]
`α = 0.4 + 0.2 * progress`
`β = 0.4 - 0.15 * progress`
`γ = 1 - α - β`

## 4. 置信度门控
如果 `max(S') < θ_low` 则忽略 RAG 分量 (β=0)
如果 `std(Q) < τ_low` 且 RAG 有高置信节点, 提升 β

## 5. 决策
- 选择 argmax(F)
- 记录三种分量及最终权重用于分析

## 6. 回退策略
若融合过程中出现 NaN / 所有节点被过滤: 回退 DRL argmax。

## 7. 输出
```
{
  'node': selected_node,
  'weights': {'alpha':α,'beta':β,'gamma':γ},
  'components': {'q':Q', 'rag':S', 'load':L'},
  'final': F
}
```

## 8. 验证计划
1. 单批次可视化 3 层分量热力图。
2. 统计融合后与单独 DRL 决策差异率及makespan变化。
3. Ablation: 禁用 RAG 或禁用负载项对比。
