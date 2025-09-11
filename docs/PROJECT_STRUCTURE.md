# WASS-RAG 项目结构 (清理后)

## 核心实验文件

```
wass/
├── docs/
│   ├── WASS_RAG_EXPERIMENT_GUIDE.md     # 完整实验指南
│   └── target.md                        # 目标文档
│
├── configs/                             # 配置文件
│   ├── experiment.yaml                  # 主实验配置
│   ├── platform.yaml                   # 平台配置  
│   ├── drl.yaml                        # DRL训练配置
│   ├── rag.yaml                        # RAG配置
│   ├── data.yaml                       # 数据配置
│   ├── labeling.yaml                   # 标注配置
│   ├── label_model.yaml                # 标签模型配置
│   ├── graph.yaml                      # 图配置
│   ├── eval.yaml                       # 评估配置
│   └── platform.xml                    # WRENCH平台定义
│
├── scripts/                             # 训练脚本 (按执行顺序)
│   ├── generate_kb_dataset.py          # [1] 生成知识库数据
│   ├── train_predictor_from_kb.py      # [2] 训练性能预测器
│   ├── train_drl_wrench.py             # [3] 训练DRL智能体
│   ├── train_rag_wrench.py             # [4] 训练RAG知识库
│   └── retrain_performance_predictor.py # 重训性能预测器
│
├── experiments/                         # 实验框架
│   ├── wrench_real_experiment.py       # 真实WRENCH实验框架
│   └── wrench_real_experiment.py       # [5] 真实WRENCH实验对比
│
├── charts/                              # 图表生成
│   ├── paper_charts.py                 # [6] 生成学术论文图表
│   └── README.md                       # 图表说明
│
├── src/                                 # 核心源码
│   ├── __init__.py                     # 包初始化
│   ├── config_loader.py                # 配置加载器
│   ├── utils.py                        # 工具函数
│   ├── performance_predictor.py        # 性能预测器
│   ├── wrench_schedulers.py            # WRENCH调度器
│   ├── ai_schedulers.py                # AI调度器
│   └── interfaces.py                   # 接口定义
│
├── wrenchtest/                          # WRENCH测试
│   ├── test_simple_wrech.py            # WRENCH环境验证
│   └── examples/                       # WRENCH示例
│
├── data/                                # 数据文件
│   ├── train.jsonl                     # 训练数据
│   ├── valid.jsonl                     # 验证数据  
│   ├── test.jsonl                      # 测试数据
│   ├── kb_training_dataset.json        # 知识库训练数据
│   └── wrench_rag_knowledge_base.pkl   # RAG知识库
│
├── models/                              # 训练模型
│   └── wass_models.pth                 # 所有训练好的模型
│
├── results/                             # 实验结果
│   └── final_experiments_discrete_event/
│       └── experiment_results.json     # 实验数据
│
├── requirements.txt                     # Python依赖
└── readme.md                           # 项目说明
```

## 实验执行顺序

### 核心训练流程 (必须按顺序执行)
```bash
# 0. 验证WRENCH环境
python wrenchtest/test_simple_wrech.py

# 1. 生成知识库数据 (240个样本)
python scripts/generate_kb_dataset.py configs/experiment.yaml

# 2. 训练性能预测器 (R²=0.9313)
python scripts/train_predictor_from_kb.py configs/experiment.yaml

# 3. 训练DRL智能体 (100 episodes)
python scripts/train_drl_wrench.py configs/experiment.yaml

# 4. 训练RAG知识库 (600个案例)
python scripts/train_rag_wrench.py configs/experiment.yaml

# 5. 运行完整实验对比
python experiments/wrench_real_experiment.py

# 6. 生成学术论文图表
python charts/paper_charts.py
```

### 输出文件检查
```bash
# 训练输出
ls -la models/wass_models.pth                    # 所有模型
ls -la data/kb_training_dataset.json             # 知识库数据
ls -la data/wrench_rag_knowledge_base.pkl        # RAG知识库

# 实验结果
ls -la results/final_experiments_discrete_event/ # 实验数据
ls -la charts/*.png                              # 生成图表
```

## 删除的无用文件

### 已清理的文件列表
```
根目录:
- wass_wrench_simulator.py              # 旧版仿真器

scripts/:
- train_drl_agent.py                    # 旧版DRL训练
- train_rag_retriever.py                # 旧版RAG训练  
- simple_drl_training.py                # 简化DRL训练
- trained_model_experiment.py           # 模型实验
- drl_agent.py                          # 独立DRL模块

experiments/:
- paper_ready_experiment.py             # 重复实验框架
- run_pipeline.py                       # 流水线脚本

charts/:
- verify_real_data.py                   # 数据验证
- acm_standards.py                      # ACM标准

src/:
- simple_schedulers.py                  # 简单调度器
- factory.py                            # 工厂模式
- encoding_constants.py                 # 编码常量
```

### 保留原因说明

**核心训练脚本 (基于WRENCH)**:
- `generate_kb_dataset.py` - 使用WRENCH生成真实仿真数据
- `train_predictor_from_kb.py` - 基于WRENCH数据训练预测器
- `train_drl_wrench.py` - 在WRENCH环境中训练DRL
- `train_rag_wrench.py` - 基于WRENCH构建RAG知识库

**实验和可视化**:
- `wrench_real_experiment.py` - 完整的WRENCH性能对比实验
- `paper_charts.py` - 生成学术论文所需图表

**核心源码**:
- `ai_schedulers.py` - AI调度器实现
- `wrench_schedulers.py` - WRENCH调度器封装
- `performance_predictor.py` - 性能预测器模型
- `config_loader.py` - 配置文件加载
- `utils.py` - 工具函数
- `interfaces.py` - 接口定义

**测试和示例**:
- `test_simple_wrech.py` - WRENCH环境验证
- `examples/` - WRENCH使用示例

## 项目特点

1. **完全基于WRENCH**: 所有训练组件都使用真实WRENCH仿真
2. **模块化设计**: 每个训练步骤独立，便于调试和修改
3. **配置驱动**: 通过YAML文件统一管理实验参数
4. **结果可重现**: 固定随机种子，确保实验一致性
5. **清晰文档**: 完整的实验指南和API文档

## 文件完整性检查

```bash
# 检查核心文件存在性
files=(
    "docs/WASS_RAG_EXPERIMENT_GUIDE.md"
    "scripts/generate_kb_dataset.py"
    "scripts/train_predictor_from_kb.py" 
    "scripts/train_drl_wrench.py"
    "scripts/train_rag_wrench.py"
    "experiments/wrench_real_experiment.py"
    "charts/paper_charts.py"
    "configs/experiment.yaml"
    "configs/platform.xml"
)

for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file"
    else  
        echo "❌ $file 缺失"
    fi
done
```

---
**清理完成时间**: 2025-09-11  
**保留文件数**: 核心文件25个  
**删除文件数**: 无用文件9个
