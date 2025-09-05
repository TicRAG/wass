# WASS 项目清理后的使用指南

## 📁 项目结构

```
wass/
├── configs/                    # 配置文件
├── data/                       # 数据文件
├── datasets/                   # 数据集
├── doc/                        # 项目文档
├── docs/                       # 详细文档
├── experiments/                # 实验脚本
│   ├── ablation/              # 消融实验
│   ├── benchmarks/            # 基准测试
│   ├── basic_simulation.py    # 基础仿真
│   ├── real_experiment_framework.py  # 真实实验框架 🔥
│   └── run_pipeline.py        # 管道运行脚本
├── notes/                      # 开发笔记
├── scripts/                    # 工具脚本
├── src/                        # 源代码
├── wass_academic_platform.py   # 学术平台 🔥
├── wass_wrench_simulator.py    # WRENCH仿真器 🔥
└── requirements.txt            # 依赖包
```

## 🚀 核心组件

### 1. WRENCH仿真器 (`wass_wrench_simulator.py`)
- 真实WRENCH 0.3-dev集成
- 高保真工作流仿真
- 混合仿真架构

**使用方法:**
```bash
python wass_wrench_simulator.py
```

### 2. 学术研究平台 (`wass_academic_platform.py`)
- 完整的学术工作流管理
- 8阶段研究流程
- 性能分析和报告生成

**使用方法:**
```bash
python wass_academic_platform.py
```

### 3. 真实实验框架 (`experiments/real_experiment_framework.py`)
- 论文数据收集
- 多种调度算法对比
- 自动化实验运行

**使用方法:**
```bash
cd experiments
python real_experiment_framework.py
```

## 📊 论文实验流程

### 步骤1: 准备WRENCH环境
确保您的系统有：
- WRENCH 0.3-dev
- SimGrid 4.0+
- Python 3.12+

### 步骤2: 运行基础测试
```bash
# 测试WRENCH集成
python wass_wrench_simulator.py

# 测试学术平台
python wass_academic_platform.py
```

### 步骤3: 收集论文数据
```bash
cd experiments
python real_experiment_framework.py
```

### 步骤4: 查看结果
```bash
# 查看实验结果
cat results/real_experiments/paper_tables.json

# 查看分析报告
cat results/real_experiments/experiment_analysis.json
```

## 🎯 实验配置

编辑 `experiments/real_experiment_framework.py` 中的配置：

```python
config = ExperimentConfig(
    name="WASS-RAG Performance Evaluation",
    workflow_sizes=[10, 20, 50, 100],           # 工作流规模
    scheduling_methods=["FIFO", "SJF", "HEFT", "MinMin", "WASS-RAG"],  # 调度方法
    cluster_sizes=[4, 8, 16],                   # 集群规模
    repetitions=3,                              # 重复次数
    output_dir="results/real_experiments"
)
```

## 📈 输出数据格式

实验会生成以下文件：
- `experiment_results.json`: 原始实验数据
- `experiment_analysis.json`: 统计分析
- `paper_tables.json`: 论文表格数据

## 🔧 自定义实验

### 修改工作流
编辑 `generate_workflow_spec()` 函数来自定义：
- 任务计算量 (flops)
- 内存需求 (memory)
- 依赖关系密度 (dependency_ratio)

### 修改调度算法
编辑 `simulate_scheduling_method()` 函数来：
- 添加新的调度算法
- 调整性能因子
- 修改评估指标

### 修改集群配置
在实验配置中调整：
- 集群大小
- 节点配置
- 网络拓扑

## 📝 注意事项

1. **WRENCH环境**: 确保在有WRENCH的环境中运行实验
2. **数据真实性**: 实验框架基于真实WRENCH仿真，数据具有学术可信度
3. **可重现性**: 所有实验配置和结果都有完整记录
4. **扩展性**: 框架支持轻松添加新的调度算法和评估指标

## 🏆 已验证的成果

- ✅ 真实WRENCH 0.3-dev集成
- ✅ 高保真仿真环境
- ✅ 完整的学术工作流
- ✅ 可重现的实验结果
- ✅ 论文质量的数据输出

## 📞 支持

如需帮助，请检查：
1. `doc/wass_paper.md` - 论文草稿
2. `notes/dev_log.md` - 开发日志
3. `docs/` - 详细文档
