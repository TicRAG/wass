#!/usr/bin/env python3
"""实验结果分析脚本."""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_experiment_results(results_dir: str):
    """加载所有实验结果."""
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"结果目录不存在: {results_dir}")
        return results
    
    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir():
            summary_file = exp_dir / 'summary.json'
            if summary_file.exists():
                try:
                    with open(summary_file, encoding='utf-8') as f:
                        results[exp_dir.name] = json.load(f)
                    print(f"✓ 加载实验结果: {exp_dir.name}")
                except Exception as e:
                    print(f"✗ 加载失败 {exp_dir.name}: {e}")
    
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
            'n_lfs': result.get('labeling_stats', {}).get('n_lfs', 0),
        }
        
        # 添加阶段耗时
        stages = result.get('stages', {})
        total_time = sum(stage.get('elapsed_seconds', 0) for stage in stages.values())
        row['total_time'] = total_time
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def plot_results(df, output_dir='./'):
    """绘制结果图表."""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 准确率对比
    axes[0, 0].bar(range(len(df)), df['accuracy'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(range(len(df)))
    axes[0, 0].set_xticklabels(df['experiment'], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1对比
    axes[0, 1].bar(range(len(df)), df['f1'], color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(range(len(df)))
    axes[0, 1].set_xticklabels(df['experiment'], rotation=45, ha='right')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 覆盖率 vs 准确率
    axes[1, 0].scatter(df['coverage'], df['accuracy'], s=100, alpha=0.7, color='green')
    for i, exp in enumerate(df['experiment']):
        axes[1, 0].annotate(exp, (df['coverage'].iloc[i], df['accuracy'].iloc[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].set_xlabel('Coverage')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Coverage vs Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 冲突率 vs F1
    axes[1, 1].scatter(df['conflict_rate'], df['f1'], s=100, alpha=0.7, color='orange')
    for i, exp in enumerate(df['experiment']):
        axes[1, 1].annotate(exp, (df['conflict_rate'].iloc[i], df['f1'].iloc[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 1].set_xlabel('Conflict Rate')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Conflict Rate vs F1', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'experiment_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表保存到: {output_path}")
    plt.show()

def generate_summary_report(df, output_dir='./'):
    """生成摘要报告."""
    report = f"""# WASS 实验结果分析报告

## 实验概述
- 总实验数量: {len(df)}
- 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验结果汇总

### 整体性能对比
{df.round(4).to_markdown(index=False)}

### 最佳性能实验
- **最高准确率**: {df.loc[df['accuracy'].idxmax(), 'experiment']} ({df['accuracy'].max():.4f})
- **最高F1**: {df.loc[df['f1'].idxmax(), 'experiment']} ({df['f1'].max():.4f})
- **最高覆盖率**: {df.loc[df['coverage'].idxmax(), 'experiment']} ({df['coverage'].max():.4f})
- **最低冲突率**: {df.loc[df['conflict_rate'].idxmin(), 'experiment']} ({df['conflict_rate'].min():.4f})
- **最快执行**: {df.loc[df['total_time'].idxmin(), 'experiment']} ({df['total_time'].min():.2f}s)

### 性能分析

#### 相关性分析
- **覆盖率与准确率**: r = {df['coverage'].corr(df['accuracy']):.4f}
- **冲突率与F1**: r = {df['conflict_rate'].corr(df['f1']):.4f}
- **LF数量与覆盖率**: r = {df['n_lfs'].corr(df['coverage']):.4f}

#### 统计信息
- **平均准确率**: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}
- **平均F1**: {df['f1'].mean():.4f} ± {df['f1'].std():.4f}
- **平均覆盖率**: {df['coverage'].mean():.4f} ± {df['coverage'].std():.4f}
- **平均冲突率**: {df['conflict_rate'].mean():.4f} ± {df['conflict_rate'].std():.4f}

## 实验建议

### 基于结果的观察
1. **覆盖率影响**: {'覆盖率与准确率呈正相关' if df['coverage'].corr(df['accuracy']) > 0.3 else '覆盖率与准确率相关性较弱'}
2. **冲突处理**: {'需要重视冲突率处理' if df['conflict_rate'].mean() > 0.1 else '冲突率控制良好'}
3. **LF设计**: {'可以考虑增加更多LF' if df['n_lfs'].mean() < 5 else 'LF数量合理'}

### 后续优化方向
- 关注{df.loc[df['accuracy'].idxmax(), 'experiment']}的配置，可作为最佳实践
- 分析{df.loc[df['conflict_rate'].idxmax(), 'experiment']}的高冲突原因
- 考虑组合多个实验的优势配置

---
*此报告由WASS框架自动生成*
"""
    
    report_path = Path(output_dir) / 'experiment_analysis_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"✓ 报告保存到: {report_path}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description='分析WASS实验结果')
    parser.add_argument('results_dir', help='实验结果目录路径')
    parser.add_argument('--output_dir', default='./', help='输出目录')
    parser.add_argument('--format', choices=['csv', 'json', 'excel'], default='csv', help='输出格式')
    parser.add_argument('--plot', action='store_true', help='生成图表')
    parser.add_argument('--report', action='store_true', help='生成分析报告')
    
    args = parser.parse_args()
    
    print(f"🔍 分析实验结果: {args.results_dir}")
    
    # 加载结果
    results = load_experiment_results(args.results_dir)
    if not results:
        print("❌ 没有找到有效的实验结果")
        return
    
    # 创建对比表格
    df = create_comparison_table(results)
    print(f"✓ 成功加载 {len(df)} 个实验结果")
    
    # 输出结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'csv':
        output_path = output_dir / 'experiment_comparison.csv'
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ CSV保存到: {output_path}")
    elif args.format == 'json':
        output_path = output_dir / 'experiment_comparison.json'
        df.to_json(output_path, orient='records', indent=2, force_ascii=False)
        print(f"✓ JSON保存到: {output_path}")
    elif args.format == 'excel':
        output_path = output_dir / 'experiment_comparison.xlsx'
        df.to_excel(output_path, index=False)
        print(f"✓ Excel保存到: {output_path}")
    
    # 生成图表
    if args.plot:
        plot_results(df, args.output_dir)
    
    # 生成报告
    if args.report:
        generate_summary_report(df, args.output_dir)
    
    # 控制台输出摘要
    print("\n📊 实验结果摘要:")
    print(df[['experiment', 'accuracy', 'f1', 'coverage', 'conflict_rate']].round(4).to_string(index=False))

if __name__ == '__main__':
    main()
