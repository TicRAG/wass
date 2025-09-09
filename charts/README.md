# WASS-RAG 论文图表生成器使用指南

## 🎨 功能概述

这个图表生成器为 WASS-RAG 论文创建高质量的学术图表，包括：

1. **热力图** - 全景性能对比
2. **雷达图** - 调度器能力指纹  
3. **箱形图** - 结果稳定性分析
4. **甘特图** - 调度案例研究
5. **综合摘要** - 多维度性能总览

## 🚀 快速使用

### 方法1: 独立运行
```bash
cd charts
python paper_charts.py
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
└── chart_index.html  # 图表总览页面
```

## 🎯 图表说明

### 1. 热力图 (Heatmap)
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

图表生成器可以自动处理以下数据源：

1. **实际实验结果** (推荐)
   - `results/wass_academic_results.json`
   - `results/experiment_results.json`

2. **自动生成模拟数据** (演示用)
   - 如果没有实际数据，会自动生成用于测试

### 数据格式示例
```json
{
  "experiments": [
    {
      "scheduler": "WASS-RAG",
      "cluster_size": 8,
      "workflow_size": 49,
      "makespan": 125.3,
      "cpu_utilization": 0.85,
      "data_locality": 0.78,
      "decision_time": 0.05
    }
  ]
}
```

## 🔧 故障排除

### 常见问题

**图表生成失败**:
```bash
# 检查依赖
pip install matplotlib seaborn pandas numpy
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
