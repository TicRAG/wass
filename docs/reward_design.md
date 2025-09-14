# DRL 奖励函数重构设计 (R-2.1)

目标: 使智能体优化目标与项目总体目标 (最小化工作流 makespan) 对齐，同时保持学习信号的稳定性与样本效率。

## 1. 问题回顾
当前最终奖励聚焦单任务完成时间, 无法引导策略全局优化。需要引入基于完整 episode 的终局稀疏奖励, 并通过 shaping 减缓稀疏性带来的学习困难。

## 2. 奖励结构概览
- 终局稀疏奖励: `R_final = - normalized_makespan`
- 中间 shaping 组件 (逐步计算, 不改变最优策略):
  1. 关键路径进展奖励 (Critical Path Progress, CPP)
  2. 负载均衡奖励 (Load Balance, LB)
  3. 任务并行度利用奖励 (Parallelism Utilization, PU)
  4. 延迟惩罚 (Queue Delay Penalty, QD)

合成: `R_step = w_cpp*CPP + w_lb*LB + w_pu*PU + w_qd*QD`
并在 episode 结束追加: `R_final`。

## 3. 归一化与稳定策略
- 对 makespan 使用运行窗口均值/方差标准化: `norm_makespan = (M - mean_M)/std_M`
- 对每类 shaping 先单独归一化到 [-1, 1]
- 使用温度参数 τ 平滑融合: `R_step = tanh( (Σ w_i * comp_i)/τ )`

## 4. 关键指标定义
| 指标 | 定义 | 范围 | 正向含义 |
|------|------|------|----------|
| CPP | (已完成关键路径任务数 / 关键路径总任务数) 增量 | [0,1] | 越大越好 |
| LB | 1 - Gini(节点累计忙碌时间) | [0,1] | 越均衡越好 |
| PU | (当前可并行执行任务数 / 节点数) 裁剪到[0,1] | [0,1] | 越高说明并行潜力利用好 |
| QD | - (平均排队等待时间 / 基准) | 负值 | 绝对值越小越好 |

## 5. 权重建议 (初始)
- w_cpp = 0.35
- w_lb = 0.2
- w_pu = 0.25
- w_qd = 0.2

## 6. 与最优策略一致性说明
所有 shaping 组件满足潜在函数或对称性条件, 不会在无限步数下改变最优策略选择 (需在实现中确保 QD 惩罚不随调度历史路径不可逆偏置)。

## 7. 实现接口
`compute_step_reward(state_context)` 返回 (reward, metrics_dict)
`compute_final_reward(episode_stats)` 返回 final_reward

## 8. 风险与缓解
| 风险 | 缓解 |
|------|------|
| 稀疏终局奖励导致收敛慢 | 重放缓存优先采样高 makespan 改善样本; 可选 value baseline | 
| Shaping 不当引入偏差 | 单元测试验证在相同 makespan 下策略无偏 | 
| 指标计算成本高 | 缓存关键路径集合; 增量维护节点负载统计 | 

## 9. 验证计划
1. 单工作流小规模仿真: 打印 reward 分解。
2. 对比旧版本: 100 episodes 学习曲线(makespan vs episodes)。
3. Ablation: 逐步移除 shaping 组件评估影响。

---
(后续迭代将补充实现细节与指标缓存策略)
