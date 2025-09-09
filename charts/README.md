# WASS-RAG 论文图表生成器使用指南

## 🎨 功能概述

这个图表生成器为 WASS-RAG 论文创建高质量的学术图表，包括：

1. **热力图** - 全景性能对比
2. **雷达图** - 调度器能力指纹  
3. **箱形图** - 结果稳定性分析
4. **甘特图** - 调度案例研究
5. **综合摘要** - 多维度性能总览

## ⚠️ 重要说明

**此图表生成器只使用真实实验数据，确保学术严谨性！**

- ✅ **基于真实实验结果**：所有图表基于实际WRENCH仿真实验
- ❌ **不使用模拟数据**：拒绝生成基于假数据的图表  
- 🔬 **数据验证机制**：自动验证实验数据格式和完整性
- 📊 **学术标准**：符合ACM出版要求

## 🚀 使用流程

### 步骤1: 运行实验获取真实数据
```bash
# 首先运行完整实验
cd experiments
python real_experiment_framework.py
```

### 步骤2: 生成图表
```bash
# 切换到图表目录
cd charts

# 方法1: 完整图表生成
python paper_charts.py

# 方法2: 完整系统测试 (推荐)
python test_complete_system.py

# 方法3: 验证数据要求
python verify_real_data.py

# 方法4: 测试真实数据处理
python test_real_charts.py
```

### 如果没有实验数据
```bash
# 验证系统要求
cd charts
python verify_real_data.py

# 会看到详细的错误提示和运行指导
```

### 方法2: 实验后自动生成
在实验脚本中添加：
```python
# 在 experiments/real_experiment_framework.py 末尾添加
from charts.paper_charts import PaperChartGenerator

# 实验完成后生成图表
print("📊 Generating paper charts...")
chart_generator = PaperChartGenerator(results_dir="results")
charts = chart_generator.generate_all_charts()
```

## 📁 输出结构

```
charts/output/
├── heatmaps/
│   └── performance_improvement_heatmap.png
├── radar/
│   └── scheduler_radar_chart.png  
├── boxplots/
│   └── stability_analysis.png
├── gantt/
│   └── scheduling_comparison.png
├── combined/
│   └── performance_summary.png
├── data/  # 📊 新增：图表数据JSON文件
│   ├── performance_improvement_data.json
│   ├── scheduler_radar_data.json
│   ├── stability_analysis_data.json
│   ├── scheduling_comparison_data.json
│   └── performance_summary_data.json
└── chart_index.html  # 图表总览页面
```

## 🎯 快速开始

### 运行方式
```bash
cd charts
python paper_charts.py
```

### 输出结果
运行完成后会生成：
- **📈 图表文件**：5种ACM标准的学术图表（PNG/PDF格式）
- **📊 数据文件**：每个图表对应的JSON数据文件
- **🌐 索引页面**：HTML总览页面，方便查看所有图表

### 数据透明度 🔍
每个图表都会生成对应的JSON数据文件，包含：
- 原始实验数据和预处理后的数据
- 数据处理过程信息
- 图表元数据（生成时间、ACM标准配置等）  
- 统计信息摘要（均值、标准差等）

这确保了研究结果的**完全可重现性**和**数据透明度**。

## 📊 图表说明
- **用途**: 展示WASS-RAG在不同场景下的性能优势
- **解读**: 颜色越深，性能提升越大
- **论文价值**: 直观回答"什么情况下WASS-RAG最有效"

### 2. 雷达图 (Radar Chart)  
- **用途**: 多维度对比不同调度器
- **解读**: 面积越大，综合性能越好
- **论文价值**: 展示WASS-RAG的全面优势

### 3. 箱形图 (Box Plot)
- **用途**: 展示结果稳定性和分布
- **解读**: 箱体越窄，结果越稳定
- **论文价值**: 证明方法的可靠性

### 4. 甘特图 (Gantt Chart)
- **用途**: 具体调度案例分析
- **解读**: 任务分配和时间安排的直观展示
- **论文价值**: 展示智能调度决策

### 5. 综合摘要 (Summary)
- **用途**: 多角度性能总览
- **解读**: 一图看懂所有关键指标
- **论文价值**: 结论部分的有力支撑

## ⚙️ 自定义配置

### 修改颜色方案
```python
COLORS = {
    'WASS-RAG': '#1f77b4',    # 蓝色
    'WASS-DRL': '#ff7f0e',    # 橙色  
    'HEFT': '#2ca02c',        # 绿色
    # 添加更多颜色...
}
```

### 调整图表尺寸
```python
# 在对应函数中修改
fig, ax = plt.subplots(figsize=(12, 8))  # 宽x高
```

### 更改输出格式
```python
# 支持多种格式
plt.savefig(output_path, dpi=300, format='pdf')  # PDF格式
plt.savefig(output_path, dpi=300, format='svg')  # SVG格式
```

## 📊 数据要求

图表生成器**仅接受真实实验结果**：

### ✅ 支持的数据源
- `results/real_experiments/experiment_results.json` (主要)
- `results/experiment_results.json`
- `results/wass_academic_results.json`
- `results/demo_wass_pipeline/wass_academic_results.json`

### 📋 必需的数据格式
```json
{
  "experiments": [
    {
      "experiment_id": "exp_001",
      "scheduling_method": "WASS-RAG",    // 调度方法名称
      "workflow_spec": {
        "task_count": 49                  // 工作流任务数
      },
      "cluster_size": 8,                  // 集群规模
      "makespan": 125.3,                  // 完工时间 (必需)
      "cpu_utilization": 0.85,            // CPU利用率
      "data_locality_score": 0.78,        // 数据局部性
      "timestamp": "2025-09-09T10:30:00"
    }
  ]
}
```

### ❌ 拒绝的数据类型
- ❌ 模拟/合成数据
- ❌ 手工编造的数据  
- ❌ 不完整的实验结果
- ❌ 缺少关键字段的数据

## 🔧 故障排除

### 常见问题

**图表生成失败**:
```bash
# 检查依赖
pip install matplotlib seaborn pandas numpy
```

**Colorbar布局冲突错误**:
```python
# 已修复：使用constrained_layout而非tight_layout
# 如果仍有问题，尝试：
matplotlib.use('Agg')  # 非交互式后端
```

**字体显示问题**:
```python
# 在代码开头添加
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
```

**内存不足**:
```python
# 减小图片分辨率
plt.savefig(output_path, dpi=150)  # 从300降到150
```

**方法名错误 (AttributeError)**:
```python
# 正确的方法名：
generator.generate_performance_heatmap(data)  # ✅
generator.generate_radar_chart(data)          # ✅  
generator.generate_stability_boxplot(data)    # ✅
generator.generate_gantt_chart(data)          # ✅
```

## 💡 论文使用建议

### 1. 图表选择
- **Introduction**: 使用综合摘要图
- **Method**: 使用雷达图展示设计思路
- **Evaluation**: 使用热力图和箱形图
- **Case Study**: 使用甘特图

### 2. 图表质量
- 所有图表默认300 DPI，适合印刷
- 支持矢量格式 (SVG/PDF)
- 学术配色方案，适合黑白打印

### 3. 数据引用
```latex
% LaTeX引用示例
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{charts/output/heatmaps/performance_improvement_heatmap.png}
\caption{WASS-RAG Performance Improvement Heatmap}
\label{fig:heatmap}
\end{figure}
```

## 🎯 扩展功能

可以轻松添加新的图表类型：

```python
def generate_custom_chart(self, results):
    # 自定义图表逻辑
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... 绘制逻辑 ...
    output_path = os.path.join(self.output_dir, 'custom', 'my_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return output_path
```

---

**Happy charting! 🎨📊**
