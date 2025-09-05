#!/usr/bin/env python3
"""批量运行Label Function实验."""

import os
import sys
import yaml
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline_enhanced import run_enhanced_pipeline

def create_lf_configurations():
    """创建不同的Label Function配置."""
    
    # 确保目录存在
    lf_dir = Path('configs/lf_experiments')
    lf_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置1: 仅关键词
    lf_keyword_only = {
        'labeling': {
            'abstain': -1,
            'lfs': [
                {
                    'name': 'keyword_positive',
                    'type': 'keyword',
                    'keywords': ['good', 'excellent', 'amazing'],
                    'label': 1
                },
                {
                    'name': 'keyword_negative', 
                    'type': 'keyword',
                    'keywords': ['bad', 'terrible', 'awful'],
                    'label': 0
                }
            ]
        }
    }
    
    # 配置2: 关键词 + 正则表达式
    lf_keyword_regex = {
        'labeling': {
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
                    'name': 'regex_excitement',
                    'type': 'regex',
                    'pattern': '!{2,}|wow|awesome|fantastic',
                    'label': 1
                },
                {
                    'name': 'regex_disappointment',
                    'type': 'regex', 
                    'pattern': 'disappointed|waste.*time|money.*wasted',
                    'label': 0
                }
            ]
        }
    }
    
    # 配置3: 全功能 (关键词 + 正则 + 长度 + URL检测)
    lf_full_featured = {
        'labeling': {
            'abstain': -1,
            'lfs': [
                {
                    'name': 'keyword_positive',
                    'type': 'keyword',
                    'keywords': ['good', 'excellent', 'amazing', 'great', 'wonderful', 'perfect', 'love'],
                    'label': 1
                },
                {
                    'name': 'keyword_negative', 
                    'type': 'keyword',
                    'keywords': ['bad', 'terrible', 'awful', 'poor', 'horrible', 'hate', 'worst'],
                    'label': 0
                },
                {
                    'name': 'regex_positive',
                    'type': 'regex',
                    'pattern': r'\b(outstanding|brilliant|superb|magnificent)\b',
                    'label': 1
                },
                {
                    'name': 'regex_negative',
                    'type': 'regex',
                    'pattern': r'\b(disgusting|pathetic|useless|garbage)\b', 
                    'label': 0
                },
                {
                    'name': 'length_meaningful',
                    'type': 'length',
                    'min_length': 10,
                    'max_length': 100,
                    'label': 1
                },
                {
                    'name': 'contains_spam_url',
                    'type': 'contains_url',
                    'label': 0
                }
            ]
        }
    }
    
    # 配置4: 精确关键词
    lf_precise_keywords = {
        'labeling': {
            'abstain': -1,
            'lfs': [
                {
                    'name': 'strong_positive',
                    'type': 'keyword',
                    'keywords': ['excellent', 'outstanding', 'magnificent', 'brilliant'],
                    'label': 1
                },
                {
                    'name': 'strong_negative',
                    'type': 'keyword', 
                    'keywords': ['terrible', 'horrible', 'disgusting', 'pathetic'],
                    'label': 0
                }
            ]
        }
    }
    
    # 保存配置文件
    configs = {
        'lf_keyword_only.yaml': lf_keyword_only,
        'lf_keyword_regex.yaml': lf_keyword_regex, 
        'lf_full_featured.yaml': lf_full_featured,
        'lf_precise_keywords.yaml': lf_precise_keywords
    }
    
    for filename, config in configs.items():
        config_path = lf_dir / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        print(f"✓ 创建配置: {config_path}")
    
    return list(configs.keys())

def run_lf_experiments():
    """运行所有Label Function实验."""
    
    # 加载基础配置
    base_config_path = Path('configs_example.yaml')
    if not base_config_path.exists():
        print("❌ 基础配置文件不存在: configs_example.yaml")
        return
    
    with open(base_config_path, encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 创建LF配置
    print("🔧 创建Label Function配置...")
    lf_configs = create_lf_configurations()
    
    # 运行实验
    results = {}
    lf_dir = Path('configs/lf_experiments')
    
    for lf_config_file in lf_configs:
        lf_config_path = lf_dir / lf_config_file
        exp_name = lf_config_path.stem
        
        print(f"\n🚀 运行实验: {exp_name}")
        
        try:
            # 加载LF配置
            with open(lf_config_path, encoding='utf-8') as f:
                lf_data = yaml.safe_load(f)
            
            # 合并配置
            config = base_config.copy()
            config['labeling'] = lf_data['labeling'] 
            config['experiment_name'] = f"lf_exp_{exp_name}"
            config['paths']['results_dir'] = f"results/lf_experiments/{exp_name}/"
            
            # 保存临时配置
            temp_config_path = f"temp_{exp_name}.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            # 运行实验
            result = run_enhanced_pipeline(temp_config_path)
            results[exp_name] = result
            
            print(f"✓ {exp_name} 完成")
            print(f"  - 准确率: {result.get('eval_stats', {}).get('accuracy', 0):.4f}")
            print(f"  - F1: {result.get('eval_stats', {}).get('f1', 0):.4f}")
            print(f"  - 覆盖率: {result.get('labeling_stats', {}).get('coverage', 0):.4f}")
            
        except Exception as e:
            print(f"✗ {exp_name} 失败: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理临时文件
            temp_config_path = f"temp_{exp_name}.yaml"
            if Path(temp_config_path).exists():
                Path(temp_config_path).unlink()
    
    # 保存汇总结果
    summary_path = Path('results/lf_experiments/summary_all.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n📊 实验汇总保存到: {summary_path}")
    return results

def compare_lf_results(results):
    """比较Label Function实验结果."""
    if not results:
        print("❌ 没有实验结果可比较")
        return
    
    print("\n📈 Label Function实验对比:")
    print("-" * 80)
    print(f"{'实验名称':<20} {'准确率':<10} {'F1':<10} {'覆盖率':<10} {'冲突率':<10} {'LF数量':<8}")
    print("-" * 80)
    
    for exp_name, result in results.items():
        accuracy = result.get('eval_stats', {}).get('accuracy', 0)
        f1 = result.get('eval_stats', {}).get('f1', 0) 
        coverage = result.get('labeling_stats', {}).get('coverage', 0)
        conflict_rate = result.get('labeling_stats', {}).get('conflict_rate', 0)
        n_lfs = result.get('labeling_stats', {}).get('n_lfs', 0)
        
        print(f"{exp_name:<20} {accuracy:<10.4f} {f1:<10.4f} {coverage:<10.4f} {conflict_rate:<10.4f} {n_lfs:<8}")
    
    print("-" * 80)
    
    # 找出最佳实验
    best_accuracy = max(results.items(), key=lambda x: x[1].get('eval_stats', {}).get('accuracy', 0))
    best_f1 = max(results.items(), key=lambda x: x[1].get('eval_stats', {}).get('f1', 0))
    best_coverage = max(results.items(), key=lambda x: x[1].get('labeling_stats', {}).get('coverage', 0))
    
    print(f"\n🏆 最佳表现:")
    print(f"  准确率最高: {best_accuracy[0]} ({best_accuracy[1]['eval_stats']['accuracy']:.4f})")
    print(f"  F1最高: {best_f1[0]} ({best_f1[1]['eval_stats']['f1']:.4f})")
    print(f"  覆盖率最高: {best_coverage[0]} ({best_coverage[1]['labeling_stats']['coverage']:.4f})")

def main():
    """主函数."""
    print("🧪 WASS Label Function 批量实验")
    print("=" * 50)
    
    # 检查数据是否存在
    data_dir = Path('data')
    if not (data_dir / 'train.jsonl').exists():
        print("⚠️ 训练数据不存在，正在生成...")
        os.system('python scripts/gen_fake_data.py --out_dir data --train 500 --valid 100 --test 100')
    
    # 运行实验
    results = run_lf_experiments()
    
    # 比较结果
    compare_lf_results(results)
    
    print(f"\n✨ 所有实验完成！查看 results/lf_experiments/ 目录获取详细结果")
    print(f"💡 运行以下命令进行深入分析:")
    print(f"   python scripts/analyze_results.py results/lf_experiments/ --plot --report")

if __name__ == '__main__':
    main()
