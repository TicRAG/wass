# WASS: 弱监督学习 + 图神经网络 + 强化学习 + RAG 实验框架

## 项目概述

WASS是一个集成了弱监督学习(Weak Supervision)、图神经网络(GNN)、强化学习(DRL)和检索增强生成(RAG)的模块化实验框架。该项目旨在为复杂的机器学习研究提供统一的实验平台。

## 核心特性

- 🧩 **模块化架构**: 各组件独立设计，易于扩展和替换
- 🏷️ **多种Label Function**: 支持关键词、正则表达式、长度等多种标注函数
- 🔗 **Wrench集成**: 支持Wrench弱监督学习库(可选)
- 📊 **详细统计**: 提供覆盖率、冲突率等关键指标
- 📋 **完整日志**: 记录每个阶段的执行时间和结果
- ⚙️ **配置驱动**: 基于YAML的灵活配置系统
- 🚀 **快速演示**: 内置演示脚本展示完整流程

## 快速开始

### 1. 生成演示数据
```bash
python scripts/gen_fake_data.py --out_dir data --train 200 --valid 50 --test 50
```

### 2. 运行基础pipeline
```bash
python -m src.pipeline_enhanced configs_example.yaml
```

### 3. 运行完整演示
```bash
python demo.py
```

## 项目结构

```
wass/
├── configs/              # 配置文件
│   ├── data.yaml
│   ├── labeling.yaml
│   ├── label_model.yaml
│   ├── graph.yaml
│   ├── rag.yaml
│   ├── drl.yaml
│   └── experiment.yaml
├── src/                  # 源代码
│   ├── data/            # 数据适配器
│   ├── labeling/        # 标签函数和矩阵
│   ├── label_model/     # 标签模型
│   ├── graph/           # 图构建和GNN
│   ├── rag/             # 检索增强
│   ├── drl/             # 强化学习
│   ├── eval/            # 评估指标
│   └── utils.py         # 工具函数
├── experiments/         # 实验脚本
├── scripts/            # 数据生成等脚本
├── notes/              # 开发日志
└── results/            # 实验结果
```

## 系统架构

```
Raw Data → Label Functions → Label Matrix → Label Model → Soft Labels
    ↓                                                           ↓
Graph Builder → Graph → GNN → Node Representations
    ↓                                    ↓
RAG Retrieval → Knowledge Enhancement → Final Predictions
    ↓
DRL Policy → Active Learning → Iterative Improvement
```

## 配置说明

### 基础配置 (configs_example.yaml)
```yaml
experiment_name: demo_wass_pipeline
paths:
  data_dir: data/
  results_dir: results/demo_wass_pipeline/

data:
  adapter: simple_jsonl
  train_file: train.jsonl
  
labeling:
  lfs:
    - name: keyword_positive
      type: keyword
      keywords: ["good", "excellent"]
      label: 1

label_model:
  type: majority_vote  # 或 wrench
  
graph:
  builder: cooccurrence
  gnn_model: gcn
  
# ... 更多配置
```

### 多文件配置
可以将配置拆分为多个文件，使用`configs/experiment.yaml`作为入口：
```bash
python -m src.pipeline_enhanced configs/experiment.yaml
```

## 支持的组件

### Label Functions
- `keyword`: 关键词匹配
- `regex`: 正则表达式
- `length`: 文本长度过滤
- `contains_url`: URL检测

### Label Models
- `majority_vote`: 多数投票
- `wrench`: Wrench库集成(需安装wrench)

### Graph Builders
- `cooccurrence`: 共现图构建

### GNN Models
- `gcn`: 图卷积网络(占位)

### RAG Components
- `simple_bm25`: 简化BM25检索
- `concat`: 拼接融合

### DRL Components
- `active_learning`: 主动学习环境
- `random`: 随机策略
- `dqn`: DQN策略(占位)

## 开发状态

### ✅ 已完成
- 核心架构和接口设计
- 所有模块的占位实现
- 完整的pipeline流程
- 配置系统和日志
- 统计指标计算
- 演示脚本

### 🚧 开发中
- Wrench真实集成(需要在有wrench的环境中完善)
- 真实GNN模型实现
- 更复杂的RAG策略
- 智能DRL策略

### 📋 待办
- 更多评估指标
- 配置验证
- 单元测试
- 性能优化

## 实验结果

运行后在`results/`目录下会生成：
- `summary.json`: 关键指标汇总
- `config_used.yaml`: 使用的配置备份
- `pipeline.log`: 详细执行日志

### 关键指标示例
```json
{
  "experiment_name": "demo_wass_pipeline",
  "data_stats": {
    "train_size": 200,
    "valid_size": 50,
    "test_size": 50
  },
  "labeling_stats": {
    "coverage": 0.412,
    "conflict_rate": 0.000,
    "lf_coverage": [0.36, 0.465]
  },
  "eval_stats": {
    "accuracy": 1.000,
    "f1": 1.000
  }
}
```

## 与Wrench集成

项目设计为可以与[Wrench](https://wrench-python-api.readthedocs.io/)无缝集成：

1. 在有Wrench的环境中，标签模型会自动使用Wrench实现
2. 在没有Wrench的环境中，会使用占位实现并给出警告
3. 支持多种Wrench标签模型：MajorityVoting, Snorkel等

```yaml
label_model:
  type: wrench
  model_name: Snorkel
  params:
    lr: 0.01
    epochs: 100
```

## 贡献指南

1. 查看`notes/dev_log.md`了解开发进展
2. 遵循模块化设计原则
3. 添加新组件时更新工厂函数
4. 编写相应的配置示例

## 论文实验

这个框架是为了支持WASS论文的实验而开发的。论文实验部分将使用这个框架在真实环境中运行。

详见`doc/wass_paper.md`了解论文背景。

## 许可证

MIT License
