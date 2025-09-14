# WASS-RAG 项目

WASS-RAG (Workflow Scheduling System with Retrieval-Augmented Generation) 是一个基于深度强化学习和检索增强生成的工作流调度系统。

## 项目结构

- `src/`: 核心源代码
  - `ai_schedulers.py`: AI调度器实现
  - `drl_agent.py`: 深度强化学习智能体
  - `performance_predictor.py`: 性能预测器
  - `knowledge_base/`: 知识库相关组件
  - `scheduling/`: 调度算法实现
  - `environment/`: 环境适配器

- `configs/`: 配置文件
  - `platform.xml`: 平台配置
  - `platform.yaml`: 平台参数
  - `experiment.yaml`: 实验配置
  - `drl.yaml`: DRL参数配置
  - `rag.yaml`: RAG相关配置

- `scripts/`: 实用脚本
  - `workflow_generator.py`: 工作流生成器
  - `platform_generator.py`: 平台生成器
  - `improved_drl_trainer.py`: DRL训练器

- `experiments/`: 实验数据和结果
  - `benchmark_validation/`: 基准验证
  - `ccr_results/`: CCR实验结果
  - `test_benchmark/`: 测试基准

- `models/`: 训练好的模型文件

- `data/`: 数据集和中间结果

## 运行实验

### 完整实验流程
```bash
./run_complete_experiment.sh
```

### 基准验证
```bash
./run_benchmark_validation.sh
```

### 系统测试
```bash
./test_curated_system.sh
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 项目特点

- 基于WRENCH仿真框架
- 支持多种工作流模式（montage, ligo, cybershake）
- 集成GNN性能预测器
- 实现RAG增强的DRL调度器
- 支持CCR参数化控制
- 模块化设计，易于扩展