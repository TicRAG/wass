# 系统架构草案 (WASS + GNN + DRL + RAG)

## 目标
提供一个模块化、可扩展的弱监督+图学习+强化学习+检索增强的统一实验框架。

## 总体流程
```
Raw Data -> (Label Functions) -> Label Matrix L  ---->  标签模型  --->  软标签 Y_hat
                               |                                   
                               v                                    
                           元数据/关系  --> 图构建(GraphBuilder) --> G (节点/边特征)

G + Y_hat + 文本/知识库 --> (RAG 检索增强特征) --> 表征融合 --> GNN 训练

GNN 表征/预测 + 交互代价/不确定性指标 --> DRL 环境(Active Policy) --> 选择操作(查询/增强/扩展) -> 迭代优化
```

## 关键模块
- data: 数据集抽象与适配器
- labeling: 标签函数、标签矩阵生成、与 Wrench 的对接包装
- label_model: 弱监督标签模型训练 (Generative Model / MajorityVote /其他)
- graph: 图构建、特征融合、GNN模型
- rag: 知识库索引 (向量/倒排) 与检索、融合策略 (late / cross-attn 占位)
- drl: 强化学习策略 (环境Env, 策略Policy, 经验缓存ReplayBuffer)
- configs: YAML/JSON 配置
- experiments: 运行脚本与批量调度
- eval: 指标与报告

## 接口草图
见 `src/interfaces.py` (将创建)。

## 配置层次
```
configs/
  dataset.yaml
  label_model.yaml
  gnn.yaml
  rag.yaml
  drl.yaml
  experiment.yaml  # 组合引用
```

## 日志 & 结果
- 统一使用 `results/<exp_name>/` 目录：保存配置副本、指标、模型权重、检索索引摘要。

## 后续
1. 创建接口文件与占位类
2. 创建一个最小 end2end pipeline stub (`run_pipeline.py`) 仅打印阶段顺序
3. 编写示例配置

## 已实现的占位模块 (2025-09-05)
- data/jsonl_adapter.py: JSONLAdapter 基础加载
- labeling/lf_base.py: 注册 & 关键词LF 构造器
- labeling/label_matrix.py: 简单标签矩阵生成 + 统计
- label_model/majority_vote.py: MajorityVote 标签模型
- graph/graph_builder.py: 共现窗口图构建
- graph/gnn_model.py: DummyGNN 伪训练/预测
- rag/retriever.py: SimpleBM25Retriever 占位实现
- rag/fusion.py: ConcatFusion 融合示例
- drl/env.py: ActiveLearningEnv 简单交互
- drl/policy.py: RandomPolicy
- eval/metrics.py: accuracy / f1 二分类指标

## 下一步整合计划
1. 工厂/注册：根据配置字符串创建上述组件 (factory 模块)
2. 统一 Pipeline 组装：读取配置 -> 依次执行：数据加载 -> 构建 L -> 训练 LabelModel -> 软标签统计 -> 图构建 -> GNN 伪训练 -> DRL 回合 (若启用) -> RAG 检索示例 -> 评估
3. 结果输出：将中间统计 (coverage/abstain_rate 等) 写入 JSON
4. 预留 Wrench 包装：`label_model/wrench_wrapper.py` (延迟导入, 若 ImportError 则提示未安装)
