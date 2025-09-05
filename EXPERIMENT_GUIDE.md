# WASS 实验指南

本文档将指导你如何使用WASS框架进行弱监督学习+图神经网络+强化学习+RAG的实验研究。

## 🎯 实验概述

WASS框架支持以下类型的实验：
- **弱监督学习实验**：比较不同标签模型的效果
- **图学习实验**：评估不同图构建策略和GNN模型
- **主动学习实验**：研究DRL策略的采样效果
- **检索增强实验**：测试RAG对预测性能的提升
- **端到端实验**：完整pipeline的综合评估

## 🚀 快速开始实验

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd wass

# 创建并激活虚拟环境 (推荐)
python -m venv wass_env

# Windows
.\wass_env\Scripts\activate
# 或者直接运行
activate_env.bat

# Linux/macOS
source wass_env/bin/activate
# 或者直接运行
./activate_env.sh

# 安装依赖
pip install -r requirements.txt

# 可选：安装Wrench (如果要使用真实的弱监督模型)
pip install wrench-ml
```

**快速启动**: 双击 `activate_env.bat` (Windows) 或运行 `./activate_env.sh` (Linux/macOS) 即可快速激活环境并查看可用命令。

### 2. 生成实验数据

**注意**: 确保已激活虚拟环境 (运行 `activate_env.bat` 或 `./activate_env.sh`)

```bash
# 生成小规模测试数据
python scripts/gen_fake_data.py --out_dir data --train 100 --valid 30 --test 30

# 生成中等规模数据
python scripts/gen_fake_data.py --out_dir data --train 1000 --valid 200 --test 200

# 生成大规模数据
python scripts/gen_fake_data.py --out_dir data --train 5000 --valid 1000 --test 1000
```

### 3. 运行基础实验

```bash
# 使用默认配置运行
python -m src.pipeline_enhanced configs_example.yaml

# 使用分模块配置运行
python -m src.pipeline_enhanced configs/experiment.yaml

# 运行演示（包含多个配置测试）
python demo.py
```

## 📋 实验类型详解

### 实验类型1：弱监督学习对比实验

**目标**：比较不同标签模型的性能

#### 1.1 创建实验配置

创建 `configs/exp_label_models.yaml`：
```yaml
experiment_name: label_model_comparison
paths:
  data_dir: data/
  results_dir: results/label_model_exp/

data:
  adapter: simple_jsonl
  train_file: train.jsonl
  valid_file: valid.jsonl
  test_file: test.jsonl

labeling:
  abstain: -1
  lfs:
    - name: keyword_positive
      type: keyword
      keywords: ["good", "excellent", "amazing", "great", "wonderful"]
      label: 1
    - name: keyword_negative
      type: keyword
      keywords: ["bad", "terrible", "awful", "poor", "horrible"]
      label: 0
    - name: length_filter
      type: length
      min_length: 5
      max_length: 50
      label: 1
    - name: regex_excitement
      type: regex
      pattern: "!{2,}|wow|amazing"
      label: 1

# 实验变量：不同的标签模型
label_model:
  type: majority_vote  # 或者 wrench
  params: {}

graph:
  builder: cooccurrence
  params:
    window_size: 5
  gnn_model: gcn
  gnn_params:
    hidden_dim: 64

rag:
  retriever: simple_bm25
  fusion: concat
  top_k: 5

drl:
  env: active_learning
  policy: random
  episodes: 3

eval:
  metrics: ["accuracy", "f1", "precision", "recall"]
```

#### 1.2 运行对比实验

```bash
# 1. 运行MajorityVote
cp configs/exp_label_models.yaml configs/exp_majority_vote.yaml
python -m src.pipeline_enhanced configs/exp_majority_vote.yaml

# 2. 创建Wrench配置
sed 's/type: majority_vote/type: wrench\n  model_name: MajorityVoting/' configs/exp_label_models.yaml > configs/exp_wrench_mv.yaml
python -m src.pipeline_enhanced configs/exp_wrench_mv.yaml

# 3. 创建Snorkel配置
sed 's/model_name: MajorityVoting/model_name: Snorkel/' configs/exp_wrench_mv.yaml > configs/exp_wrench_snorkel.yaml
python -m src.pipeline_enhanced configs/exp_wrench_snorkel.yaml
```

#### 1.3 结果分析

```bash
# 比较结果
python scripts/compare_results.py results/label_model_exp/ --metric accuracy
```

### 实验类型2：Label Function 设计实验

**目标**：研究不同Label Function组合的效果

#### 2.1 创建LF变体配置

```bash
# 创建多个LF配置变体
mkdir -p configs/lf_experiments
```

创建 `configs/lf_experiments/lf_keyword_only.yaml`：
```yaml
labeling:
  lfs:
    - name: keyword_positive
      type: keyword
      keywords: ["good", "excellent"]
      label: 1
    - name: keyword_negative
      type: keyword
      keywords: ["bad", "terrible"]
      label: 0
```

创建 `configs/lf_experiments/lf_keyword_regex.yaml`：
```yaml
labeling:
  lfs:
    - name: keyword_positive
      type: keyword
      keywords: ["good", "excellent", "amazing"]
      label: 1
    - name: keyword_negative
      type: keyword
      keywords: ["bad", "terrible", "awful"]
      label: 0
    - name: regex_positive
      type: regex
      pattern: "\\b(great|awesome|fantastic)\\b"
      label: 1
    - name: regex_negative
      type: regex
      pattern: "\\b(hate|disgusting|worst)\\b"
      label: 0
```

#### 2.2 批量运行实验

创建 `scripts/run_lf_experiments.py`：
```python
#!/usr/bin/env python3
"""批量运行Label Function实验."""

import os
import yaml
from pathlib import Path
from src.pipeline_enhanced import run_enhanced_pipeline

def run_lf_experiments():
    """运行所有LF配置实验."""
    base_config = yaml.safe_load(Path('configs_example.yaml').read_text())
    lf_configs = Path('configs/lf_experiments').glob('*.yaml')
    
    results = {}
    for lf_config in lf_configs:
        print(f"运行实验: {lf_config.name}")
        
        # 合并配置
        lf_data = yaml.safe_load(lf_config.read_text())
        config = base_config.copy()
        config['labeling'] = lf_data['labeling']
        config['experiment_name'] = f"lf_exp_{lf_config.stem}"
        config['paths']['results_dir'] = f"results/lf_experiments/{lf_config.stem}/"
        
        # 保存临时配置
        temp_config = f"temp_{lf_config.stem}.yaml"
        Path(temp_config).write_text(yaml.dump(config))
        
        try:
            # 运行实验
            result = run_enhanced_pipeline(temp_config)
            results[lf_config.stem] = result
            print(f"✓ {lf_config.name} 完成")
        except Exception as e:
            print(f"✗ {lf_config.name} 失败: {e}")
        finally:
            # 清理临时文件
            if Path(temp_config).exists():
                Path(temp_config).unlink()
    
    return results

if __name__ == '__main__':
    run_lf_experiments()
```

运行批量实验：
```bash
python scripts/run_lf_experiments.py
```

### 实验类型3：图构建策略实验

**目标**：比较不同图构建方法的效果

#### 3.1 扩展图构建器

首先扩展 `src/graph/graph_builder.py`：
```python
class SimilarityGraphBuilder:
    """基于相似度的图构建器."""
    def __init__(self, similarity_threshold: float = 0.5, field: str = 'text'):
        self.threshold = similarity_threshold
        self.field = field
    
    def build(self, data: List[Dict[str, Any]], labels):
        # 计算文本相似度并构建图
        # 这里可以用TF-IDF + 余弦相似度
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        texts = [sample.get(self.field, '') for sample in data]
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        graph = defaultdict(lambda: defaultdict(float))
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                sim = similarity_matrix[i, j]
                if sim > self.threshold:
                    graph[f"node_{i}"][f"node_{j}"] = sim
                    graph[f"node_{j}"][f"node_{i}"] = sim
        
        return graph
```

#### 3.2 创建图实验配置

创建 `configs/graph_experiments/`目录，包含不同图配置：

`cooccurrence_graph.yaml`:
```yaml
graph:
  builder: cooccurrence
  params:
    window_size: 5
```

`similarity_graph.yaml`:
```yaml
graph:
  builder: similarity
  params:
    similarity_threshold: 0.3
```

#### 3.3 运行图实验

```bash
python scripts/run_graph_experiments.py
```

### 实验类型4：端到端系统实验

**目标**：评估完整系统在真实场景下的表现

#### 4.1 准备真实数据

```python
# scripts/prepare_real_data.py
"""准备真实数据集的脚本."""

def convert_imdb_to_jsonl():
    """将IMDB数据转换为JSONL格式."""
    # 假设你有IMDB数据
    import pandas as pd
    
    # 读取数据
    df = pd.read_csv('path/to/imdb.csv')
    
    # 转换格式
    for split in ['train', 'valid', 'test']:
        split_df = df[df['split'] == split]
        with open(f'data/{split}.jsonl', 'w') as f:
            for _, row in split_df.iterrows():
                item = {
                    'text': row['review'],
                    'label': row['sentiment'],  # 真实标签，用于评估
                    'id': row['id']
                }
                f.write(json.dumps(item) + '\n')

def convert_amazon_to_jsonl():
    """转换Amazon评论数据."""
    # 类似实现
    pass
```

#### 4.2 创建真实数据实验配置

`configs/real_data_exp.yaml`:
```yaml
experiment_name: real_data_evaluation
paths:
  data_dir: data/real/
  results_dir: results/real_data_exp/

data:
  adapter: simple_jsonl
  train_file: train.jsonl
  valid_file: valid.jsonl
  test_file: test.jsonl

labeling:
  lfs:
    # 基于领域知识设计的LF
    - name: positive_words
      type: keyword
      keywords: ["excellent", "outstanding", "wonderful", "fantastic", "amazing", "great", "love", "perfect", "brilliant"]
      label: 1
    - name: negative_words
      type: keyword
      keywords: ["terrible", "awful", "horrible", "disgusting", "hate", "worst", "bad", "poor", "disappointing"]
      label: 0
    - name: rating_patterns
      type: regex
      pattern: "5\\s*(stars?|/5|out of 5)"
      label: 1
    - name: short_negative
      type: length
      max_length: 10
      label: 0

label_model:
  type: wrench
  model_name: Snorkel
  params:
    lr: 0.01
    l2: 0.01
    n_epochs: 100

# 其他配置...
```

#### 4.3 运行完整评估

```bash
python -m src.pipeline_enhanced configs/real_data_exp.yaml
```

## 📊 实验分析与报告

### 1. 结果比较脚本

创建 `scripts/analyze_results.py`：
```python
"""实验结果分析脚本."""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_results(results_dir: str):
    """加载所有实验结果."""
    results = {}
    results_path = Path(results_dir)
    
    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir():
            summary_file = exp_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    results[exp_dir.name] = json.load(f)
    
    return results

def create_comparison_table(results):
    """创建对比表格."""
    data = []
    for exp_name, result in results.items():
        row = {
            'experiment': exp_name,
            'train_size': result.get('data_stats', {}).get('train_size', 0),
            'coverage': result.get('labeling_stats', {}).get('coverage', 0),
            'conflict_rate': result.get('labeling_stats', {}).get('conflict_rate', 0),
            'accuracy': result.get('eval_stats', {}).get('accuracy', 0),
            'f1': result.get('eval_stats', {}).get('f1', 0),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def plot_results(df):
    """绘制结果图表."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 准确率对比
    axes[0, 0].bar(df['experiment'], df['accuracy'])
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1对比
    axes[0, 1].bar(df['experiment'], df['f1'])
    axes[0, 1].set_title('F1 Score Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 覆盖率 vs 准确率
    axes[1, 0].scatter(df['coverage'], df['accuracy'])
    axes[1, 0].set_xlabel('Coverage')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Coverage vs Accuracy')
    
    # 冲突率 vs F1
    axes[1, 1].scatter(df['conflict_rate'], df['f1'])
    axes[1, 1].set_xlabel('Conflict Rate')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Conflict Rate vs F1')
    
    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 分析结果
    results = load_experiment_results('results/')
    df = create_comparison_table(results)
    
    print("实验结果对比:")
    print(df.to_string(index=False))
    
    # 保存到CSV
    df.to_csv('experiment_comparison.csv', index=False)
    
    # 绘制图表
    plot_results(df)
```

### 2. 自动化报告生成

创建 `scripts/generate_report.py`：
```python
"""生成实验报告."""

def generate_markdown_report(results_df):
    """生成Markdown格式的实验报告."""
    report = f"""# WASS 实验报告
    
## 实验概述
本报告包含了 {len(results_df)} 个实验的结果对比。

## 实验结果

### 整体性能对比
{results_df.to_markdown(index=False)}

### 最佳性能实验
- **最高准确率**: {results_df.loc[results_df['accuracy'].idxmax(), 'experiment']} ({results_df['accuracy'].max():.3f})
- **最高F1**: {results_df.loc[results_df['f1'].idxmax(), 'experiment']} ({results_df['f1'].max():.3f})
- **最高覆盖率**: {results_df.loc[results_df['coverage'].idxmax(), 'experiment']} ({results_df['coverage'].max():.3f})
- **最低冲突率**: {results_df.loc[results_df['conflict_rate'].idxmin(), 'experiment']} ({results_df['conflict_rate'].min():.3f})

### 性能分析
1. **覆盖率与准确率的关系**: 
   - 相关系数: {results_df['coverage'].corr(results_df['accuracy']):.3f}
   
2. **冲突率与性能的关系**:
   - 冲突率与F1相关系数: {results_df['conflict_rate'].corr(results_df['f1']):.3f}

## 结论与建议
[根据实验结果填写结论]

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    return report
```

## 🔧 高级实验技巧

### 1. 超参数搜索

创建 `scripts/hyperparameter_search.py`：
```python
"""超参数搜索脚本."""

from itertools import product
import yaml

def grid_search_label_model():
    """标签模型超参数网格搜索."""
    if model_type == 'wrench':
        param_grid = {
            'lr': [0.001, 0.01, 0.1],
            'l2': [0.001, 0.01, 0.1],
            'n_epochs': [50, 100, 200]
        }
        
        for lr, l2, epochs in product(*param_grid.values()):
            config = base_config.copy()
            config['label_model']['params'] = {
                'lr': lr, 'l2': l2, 'n_epochs': epochs
            }
            config['experiment_name'] = f"grid_search_lr{lr}_l2{l2}_ep{epochs}"
            
            # 运行实验
            run_experiment(config)
```

### 2. 交叉验证

```python
def k_fold_validation(k=5):
    """K折交叉验证."""
    from sklearn.model_selection import KFold
    
    # 加载数据
    data = load_data()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        # 创建fold特定的数据文件
        create_fold_data(data, train_idx, val_idx, fold)
        
        # 运行实验
        config = create_fold_config(fold)
        result = run_enhanced_pipeline(config)
        results.append(result)
    
    # 汇总结果
    return aggregate_cv_results(results)
```

### 3. 统计显著性测试

```python
def statistical_significance_test(results1, results2):
    """统计显著性测试."""
    from scipy import stats
    
    # 提取性能指标
    acc1 = [r['eval_stats']['accuracy'] for r in results1]
    acc2 = [r['eval_stats']['accuracy'] for r in results2]
    
    # t检验
    t_stat, p_value = stats.ttest_ind(acc1, acc2)
    
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
```

## 📝 实验最佳实践

### 1. 实验设计原则
- **控制变量**: 每次实验只改变一个变量
- **多次运行**: 使用不同随机种子运行多次
- **基线对比**: 始终包含简单基线方法
- **统计检验**: 进行显著性测试验证结果

### 2. 结果记录
- 详细记录实验设置和超参数
- 保存中间结果和模型权重
- 记录实验环境信息
- 备份原始数据和代码版本

### 3. 可复现性
```bash
# 设置随机种子
export PYTHONHASHSEED=0
python -c "import random; random.seed(42)"

# 记录环境信息
pip freeze > requirements.txt
python --version > python_version.txt
git rev-parse HEAD > git_commit.txt
```

## 🎓 示例：完整实验工作流

```bash
# 1. 准备环境
python scripts/setup_experiment_env.py

# 2. 生成数据
python scripts/gen_fake_data.py --out_dir data --train 1000 --valid 200 --test 200

# 3. 运行基线实验
python -m src.pipeline_enhanced configs/baseline.yaml

# 4. 运行对比实验
python scripts/run_comparison_experiments.py

# 5. 分析结果
python scripts/analyze_results.py

# 6. 生成报告
python scripts/generate_report.py

# 7. 提交结果
git add results/ reports/
git commit -m "Add experiment results for [实验名称]"
```

## 🐛 常见问题与解决方案

### 1. Wrench环境问题
```bash
# 如果Wrench导入失败
pip install wrench-ml==1.2.0

# 如果版本冲突
conda create -n wrench python=3.8
conda activate wrench
pip install wrench-ml
```

### 2. 内存不足
```python
# 减少数据规模
python scripts/gen_fake_data.py --train 100 --valid 20 --test 20

# 或调整批处理大小
config['training']['batch_size'] = 16
```

### 3. 实验重现问题
```python
# 确保设置随机种子
import random
import numpy as np
random.seed(42)
np.random.seed(42)
```

---

这个实验指南提供了从基础到高级的完整实验流程。根据你的研究需求，可以选择相应的实验类型和分析方法。

需要针对特定实验场景的更详细指导吗？
