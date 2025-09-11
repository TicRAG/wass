# WASS-RAG 完整实验指南

## 概述

WASS-RAG (Workflow-Aware Scheduling System with Retrieval-Augmented Generation) 是一个基于WRENCH仿真的智能工作流调度系统。本文档描述了完整的训练和实验流程。

## 系统架构

```
WASS-RAG 系统组件：
├── 知识库生成 (generate_kb_dataset.py)     - 基于WRENCH仿真
├── 性能预测器 (train_predictor_from_kb.py) - 基于WRENCH数据
├── DRL智能体 (train_drl_wrench.py)         - 基于WRENCH环境
├── RAG知识库 (train_rag_wrench.py)         - 基于WRENCH仿真
└── 实验框架 (wrench_real_experiment.py)    - 真实WRENCH实验对比
```

## 实验环境要求

### 软件依赖
```bash
# Python环境
python >= 3.8

# 核心依赖包
numpy
torch
pyyaml
matplotlib
pandas

# WRENCH仿真框架
wrench-python-api
```

### 平台配置
- 4节点异构集群仿真
- 不同CPU容量: 2.0, 3.0, 2.5, 4.0 GHz
- 统一存储服务
- 网络和I/O建模

## 完整实验流程

### 第1步: 环境准备

```bash
# 1. 激活虚拟环境
source wrench-env/bin/activate

# 2. 验证WRENCH环境
cd /data/workspace/wass
python test_simple_wrech.py
```

**预期输出:**
```
WRENCH仿真开始...
创建工作流：Task_A -> Task_B
调度作业到计算节点
工作流完成时间: 4.0016s
✅ WRENCH测试成功
```

### 第2步: 知识库生成

```bash
python scripts/generate_kb_dataset.py configs/experiment.yaml
```

**功能:** 
- 使用HEFT和FIFO调度器
- 生成240个真实仿真样本
- 输出: `data/kb_training_dataset.json`

**预期输出:**
```
🚀 开始生成知识库数据集...
WRENCH环境初始化完成
生成调度器案例: HEFT
生成调度器案例: FIFO
✅ 知识库生成完成: 240个样本
```

### 第3步: 性能预测器训练

```bash
python scripts/train_predictor_from_kb.py configs/experiment.yaml
```

**功能:**
- 训练神经网络性能预测器
- 使用知识库数据进行监督学习
- 输出: `models/wass_models.pth` (性能预测器部分)

**预期输出:**
```
🧠 开始训练性能预测器...
训练样本: 192, 验证样本: 48
Epoch 50/100: Loss=0.045, Val R²=0.8901
Epoch 100/100: Loss=0.021, Val R²=0.9313
✅ 性能预测器训练完成: R²=0.9313
```

### 第4步: DRL智能体训练

```bash
python scripts/train_drl_wrench.py configs/experiment.yaml
```

**功能:**
- 在WRENCH环境中训练深度强化学习智能体
- 使用DQN算法学习调度策略
- 输出: `models/wass_models.pth` (DRL智能体部分)

**预期输出:**
```
🚀 开始基于WRENCH的DRL智能体训练...
WRENCH环境初始化完成: 4 计算节点, 状态维度: 17
Episode 0: 平均奖励=5.46, 平均Makespan=7.13s, ε=1.000
Episode 50: 平均奖励=4.96, 平均Makespan=16.80s, ε=0.786
Episode 100: 平均奖励=4.65, 平均Makespan=18.04s, ε=0.643
✅ DRL模型已保存
```

### 第5步: RAG知识库训练

```bash
python scripts/train_rag_wrench.py configs/experiment.yaml
```

**功能:**
- 基于WRENCH仿真构建RAG知识库
- 包含工作流相似度检索和案例推荐
- 输出: `data/wrench_rag_knowledge_base.pkl`

**预期输出:**
```
🚀 开始生成 600 个WRENCH知识案例...
已生成 600/600 个案例...
构建检索索引，共 600 个案例...
索引构建完成：20 个聚类
📊 检索质量评估 - 调度器一致性: 0.460
✅ RAG检索器训练完成
```

### 第6步: 运行完整实验

```bash
python experiments/wrench_real_experiment.py
```

**功能:**
- 对比不同调度算法性能
- 生成实验数据和统计结果
- 输出: `results/final_experiments_discrete_event/`

**对比算法:**
1. FIFO - 先进先出
2. HEFT - 异构最早完成时间
3. WASS (Heuristic) - 启发式调度
4. WASS-DRL (w/o RAG) - 纯DRL调度
5. WASS-RAG - 完整系统

### 第7步: 生成实验图表

```bash
python charts/paper_charts.py
```

**功能:**
- 生成学术论文所需的图表
- 包括性能对比、收敛曲线等
- 输出: `charts/` 目录下的PNG文件

**生成图表:**
- 调度器性能对比柱状图
- 不同工作流规模的性能曲线
- 训练收敛曲线
- 系统架构图

## 实验配置文件

### 主配置 (`configs/experiment.yaml`)
```yaml
experiment_name: demo_wass_pipeline
random_seed: 42
include:
  - data.yaml
  - platform.yaml
  - labeling.yaml
  - label_model.yaml
  - graph.yaml
  - rag.yaml
  - drl.yaml
  - eval.yaml
```

### 平台配置 (`configs/platform.yaml`)
```yaml
platform:
  platform_file: "configs/platform.xml"
  controller_host: "ControllerHost"
  storage_host: "StorageHost"
  compute_nodes: ["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"]
```

### DRL配置 (`configs/drl.yaml`)
```yaml
drl:
  episodes: 100
  max_steps: 25
  network:
    hidden_dim: 128
    learning_rate: 0.001
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.1
```

### RAG配置 (`configs/rag.yaml`)
```yaml
rag:
  retriever: "wrench_similarity"
  top_k: 5
  num_cases: 600
  embedding_dim: 64
  similarity_weights:
    workflow: 0.7
    task: 0.3
```

## 实验结果解释

### 性能指标
- **Makespan**: 工作流完成总时间 (越小越好)
- **CPU利用率**: 计算资源利用效率
- **调度时间**: 调度决策计算时间
- **成功率**: 成功完成的工作流比例

### 预期性能排序 (Makespan)
1. WASS-RAG (最佳) - 结合预测器、DRL和历史经验
2. WASS-DRL (w/o RAG) - 纯强化学习调度
3. HEFT - 经典启发式算法
4. WASS (Heuristic) - 简单启发式
5. FIFO (最差) - 无智能优化

### 实验数据文件
```
results/final_experiments_discrete_event/
├── experiment_results.json      # 详细实验数据
├── performance_summary.json     # 性能汇总
└── charts/                      # 生成的图表
    ├── scheduler_comparison.png
    ├── training_curves.png
    └── system_architecture.png
```

## 故障排除

### 常见问题

1. **WRENCH导入错误**
   ```bash
   # 确保在正确的虚拟环境中
   source wrench-env/bin/activate
   python -c "import wrench; print('WRENCH OK')"
   ```

2. **内存不足**
   - 减少训练episode数量
   - 调整batch_size
   - 使用较小的工作流规模

3. **训练不收敛**
   - 检查学习率设置
   - 增加训练episode
   - 调整奖励函数权重

4. **实验数据不一致**
   - 确保随机种子固定
   - 验证配置文件一致性
   - 检查平台XML配置

### 验证检查列表

**训练完成验证:**
```bash
# 检查所有训练输出
ls -la models/wass_models.pth
ls -la data/kb_training_dataset.json
ls -la data/wrench_rag_knowledge_base.pkl

# 验证模型完整性
python -c "
import torch
checkpoint = torch.load('models/wass_models.pth', map_location='cpu', weights_only=False)
print('训练组件:', list(checkpoint.keys()))
print('性能预测器R²:', checkpoint['metadata']['performance_predictor']['validation_results']['r2'])
print('DRL最终性能:', checkpoint['drl_metadata']['avg_makespan'])
"
```

**实验结果验证:**
```bash
# 检查实验输出
ls -la results/final_experiments_discrete_event/
python -c "
import json
with open('results/final_experiments_discrete_event/experiment_results.json') as f:
    results = json.load(f)
print('实验配置:', results['experiment_config']['name'])
print('调度器数量:', len(results['experiment_config']['scheduling_methods']))
"
```

## 性能基准

### 硬件要求
- **CPU**: 4核以上推荐
- **内存**: 8GB以上
- **存储**: 5GB可用空间
- **运行时间**: 完整流程约30-60分钟

### 性能基准数据
```
组件                    | 训练时间  | 输出大小
--------------------- | -------- | --------
知识库生成              | ~5分钟   | ~500KB
性能预测器训练          | ~3分钟   | ~2MB
DRL智能体训练          | ~15分钟  | ~5MB
RAG知识库训练          | ~10分钟  | ~10MB
完整实验运行           | ~20分钟  | ~1MB
图表生成               | ~2分钟   | ~500KB
```

## 扩展指南

### 添加新调度器
1. 在 `src/ai_schedulers.py` 中实现调度逻辑
2. 更新 `experiments/wrench_real_experiment.py` 中的调度器列表
3. 重新运行实验对比

### 修改平台配置
1. 编辑 `configs/platform.xml`
2. 更新节点容量和网络拓扑
3. 重新生成知识库和训练模型

### 调整训练参数
1. 修改对应的配置文件 (drl.yaml, rag.yaml等)
2. 重新运行相应的训练脚本
3. 验证性能改进

---

**最后更新**: 2025-09-11  
**版本**: 1.0  
**作者**: WASS-RAG 开发团队
