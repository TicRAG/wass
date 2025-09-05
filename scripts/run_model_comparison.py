#!/usr/bin/env python3
"""比较不同标签模型的性能."""

import os
import sys
import yaml
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline_enhanced import run_enhanced_pipeline

def create_label_model_configs():
    """创建不同标签模型的配置."""
    
    configs_dir = Path('configs/label_model_experiments')
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # 基础配置模板
    base_labeling = {
        'abstain': -1,
        'lfs': [
            {
                'name': 'keyword_positive',
                'type': 'keyword',
                'keywords': ['good', 'excellent', 'amazing', 'great', 'wonderful'],
                'label': 1
            },
            {
                'name': 'keyword_negative',
                'type': 'keyword', 
                'keywords': ['bad', 'terrible', 'awful', 'poor', 'horrible'],
                'label': 0
            },
            {
                'name': 'regex_positive',
                'type': 'regex',
                'pattern': 'love|perfect|brilliant',
                'label': 1
            },
            {
                'name': 'length_filter',
                'type': 'length',
                'min_length': 5,
                'max_length': 50, 
                'label': 1
            }
        ]
    }
    
    # 不同标签模型配置
    model_configs = {
        'majority_vote.yaml': {
            'label_model': {
                'type': 'majority_vote',
                'params': {}
            }
        },
        'wrench_majority.yaml': {
            'label_model': {
                'type': 'wrench',
                'model_name': 'MajorityVoting',
                'params': {}
            }
        },
        'wrench_snorkel.yaml': {
            'label_model': {
                'type': 'wrench',
                'model_name': 'Snorkel',
                'params': {
                    'lr': 0.01,
                    'l2': 0.01,
                    'n_epochs': 100
                }
            }
        }
    }
    
    # 保存配置文件
    for filename, config in model_configs.items():
        config['labeling'] = base_labeling
        config_path = configs_dir / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        print(f"✓ 创建配置: {config_path}")
    
    return list(model_configs.keys())

def run_label_model_comparison():
    """运行标签模型对比实验."""
    
    # 加载基础配置
    base_config_path = Path('configs_example.yaml')
    if not base_config_path.exists():
        print("❌ 基础配置文件不存在: configs_example.yaml")
        return
    
    with open(base_config_path, encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 创建模型配置
    print("🔧 创建标签模型配置...")
    model_configs = create_label_model_configs()
    
    # 运行实验
    results = {}
    configs_dir = Path('configs/label_model_experiments')
    
    for model_config_file in model_configs:
        model_config_path = configs_dir / model_config_file
        exp_name = model_config_path.stem
        
        print(f"\n🚀 运行实验: {exp_name}")
        
        try:
            # 加载模型配置
            with open(model_config_path, encoding='utf-8') as f:
                model_data = yaml.safe_load(f)
            
            # 合并配置
            config = base_config.copy()
            config.update(model_data)
            config['experiment_name'] = f"label_model_{exp_name}"
            config['paths']['results_dir'] = f"results/label_model_experiments/{exp_name}/"
            
            # 保存临时配置
            temp_config_path = f"temp_model_{exp_name}.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            # 运行实验
            result = run_enhanced_pipeline(temp_config_path)
            results[exp_name] = result
            
            print(f"✓ {exp_name} 完成")
            print(f"  - 模型类型: {config['label_model']['type']}")
            if 'model_name' in config['label_model']:
                print(f"  - 模型名称: {config['label_model']['model_name']}")
            print(f"  - 准确率: {result.get('eval_stats', {}).get('accuracy', 0):.4f}")
            print(f"  - F1: {result.get('eval_stats', {}).get('f1', 0):.4f}")
            
        except Exception as e:
            print(f"✗ {exp_name} 失败: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理临时文件
            temp_config_path = f"temp_model_{exp_name}.yaml"
            if Path(temp_config_path).exists():
                Path(temp_config_path).unlink()
    
    # 保存汇总结果
    summary_path = Path('results/label_model_experiments/summary_all.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n📊 实验汇总保存到: {summary_path}")
    return results

def compare_model_results(results):
    """比较标签模型实验结果."""
    if not results:
        print("❌ 没有实验结果可比较")
        return
    
    print("\n📈 标签模型性能对比:")
    print("-" * 90)
    print(f"{'模型':<20} {'准确率':<10} {'F1':<10} {'覆盖率':<10} {'冲突率':<10} {'训练时间':<10}")
    print("-" * 90)
    
    for exp_name, result in results.items():
        accuracy = result.get('eval_stats', {}).get('accuracy', 0)
        f1 = result.get('eval_stats', {}).get('f1', 0)
        coverage = result.get('labeling_stats', {}).get('coverage', 0)
        conflict_rate = result.get('labeling_stats', {}).get('conflict_rate', 0)
        
        # 获取训练时间
        label_model_stage = result.get('stages', {}).get('label_model', {})
        train_time = label_model_stage.get('elapsed_seconds', 0)
        
        print(f"{exp_name:<20} {accuracy:<10.4f} {f1:<10.4f} {coverage:<10.4f} {conflict_rate:<10.4f} {train_time:<10.3f}s")
    
    print("-" * 90)
    
    # 分析结果
    print(f"\n📊 性能分析:")
    
    # 找出最佳模型
    best_accuracy = max(results.items(), key=lambda x: x[1].get('eval_stats', {}).get('accuracy', 0))
    best_f1 = max(results.items(), key=lambda x: x[1].get('eval_stats', {}).get('f1', 0))
    fastest = min(results.items(), key=lambda x: x[1].get('stages', {}).get('label_model', {}).get('elapsed_seconds', float('inf')))
    
    print(f"  🏆 最高准确率: {best_accuracy[0]} ({best_accuracy[1]['eval_stats']['accuracy']:.4f})")
    print(f"  🏆 最高F1: {best_f1[0]} ({best_f1[1]['eval_stats']['f1']:.4f})")
    print(f"  ⚡ 最快训练: {fastest[0]} ({fastest[1]['stages']['label_model']['elapsed_seconds']:.3f}s)")
    
    # 计算平均性能
    avg_accuracy = sum(r.get('eval_stats', {}).get('accuracy', 0) for r in results.values()) / len(results)
    avg_f1 = sum(r.get('eval_stats', {}).get('f1', 0) for r in results.values()) / len(results)
    
    print(f"\n📐 平均性能:")
    print(f"  平均准确率: {avg_accuracy:.4f}")
    print(f"  平均F1: {avg_f1:.4f}")
    
    # Wrench vs 内置模型对比
    wrench_results = {k: v for k, v in results.items() if 'wrench' in k}
    builtin_results = {k: v for k, v in results.items() if 'majority' in k and 'wrench' not in k}
    
    if wrench_results and builtin_results:
        wrench_avg_acc = sum(r.get('eval_stats', {}).get('accuracy', 0) for r in wrench_results.values()) / len(wrench_results)
        builtin_avg_acc = sum(r.get('eval_stats', {}).get('accuracy', 0) for r in builtin_results.values()) / len(builtin_results)
        
        print(f"\n🔄 Wrench vs 内置模型:")
        print(f"  Wrench平均准确率: {wrench_avg_acc:.4f}")
        print(f"  内置模型平均准确率: {builtin_avg_acc:.4f}")
        print(f"  性能提升: {((wrench_avg_acc - builtin_avg_acc) / builtin_avg_acc * 100):+.2f}%")

def main():
    """主函数."""
    print("🏷️ WASS 标签模型对比实验")
    print("=" * 50)
    
    # 检查数据是否存在
    data_dir = Path('data')
    if not (data_dir / 'train.jsonl').exists():
        print("⚠️ 训练数据不存在，正在生成...")
        os.system('python scripts/gen_fake_data.py --out_dir data --train 500 --valid 100 --test 100')
    
    # 运行实验
    results = run_label_model_comparison()
    
    # 比较结果
    compare_model_results(results)
    
    print(f"\n✨ 标签模型对比实验完成！")
    print(f"📁 查看 results/label_model_experiments/ 目录获取详细结果")
    print(f"💡 运行以下命令进行深入分析:")
    print(f"   python scripts/analyze_results.py results/label_model_experiments/ --plot --report")

if __name__ == '__main__':
    main()
