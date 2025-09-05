"""演示脚本 - 展示WASS项目的完整流程."""
from __future__ import annotations
import json
from pathlib import Path
from src.pipeline_enhanced import run_enhanced_pipeline
from src.utils import setup_logger

def main():
    """运行完整的演示流程."""
    print("=== WASS项目演示 ===")
    print("该演示将展示弱监督学习+图神经网络+强化学习+RAG的完整pipeline")
    
    # 1. 生成演示数据
    print("\n1. 生成演示数据...")
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, 'scripts/gen_fake_data.py',
            '--out_dir', 'data',
            '--train', '200',
            '--valid', '50', 
            '--test', '50'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 演示数据生成成功")
        else:
            print(f"✗ 数据生成失败: {result.stderr}")
            return
    except Exception as e:
        print(f"✗ 数据生成失败: {e}")
        return
    
    # 2. 运行基础pipeline
    print("\n2. 运行基础pipeline...")
    try:
        results = run_enhanced_pipeline('configs_example.yaml')
        print("✓ 基础pipeline运行成功")
        
        # 显示关键指标
        print(f"  - 训练数据: {results['data_stats']['train_size']}样本")
        print(f"  - 标签覆盖率: {results['labeling_stats']['coverage']:.3f}")
        print(f"  - 标签冲突率: {results['labeling_stats']['conflict_rate']:.3f}")
        print(f"  - 评估准确率: {results['eval_stats']['accuracy']:.3f}")
        
    except Exception as e:
        print(f"✗ Pipeline运行失败: {e}")
        return
    
    # 3. 创建Wrench配置并测试
    print("\n3. 创建Wrench配置测试...")
    wrench_config = {
        'experiment_name': 'demo_wrench_test',
        'paths': {
            'data_dir': 'data/',
            'results_dir': 'results/demo_wrench_test/'
        },
        'data': {
            'adapter': 'simple_jsonl',
            'train_file': 'train.jsonl',
            'valid_file': 'valid.jsonl', 
            'test_file': 'test.jsonl'
        },
        'labeling': {
            'abstain': -1,
            'lfs': [
                {'name': 'keyword_positive', 'type': 'keyword', 'keywords': ['good', 'excellent', 'amazing'], 'label': 1},
                {'name': 'keyword_negative', 'type': 'keyword', 'keywords': ['bad', 'terrible', 'awful'], 'label': 0},
                {'name': 'length_filter', 'type': 'length', 'min_length': 3, 'max_length': 20, 'label': 1}
            ]
        },
        'label_model': {
            'type': 'wrench',
            'model_name': 'MajorityVoting',
            'params': {}
        },
        'graph': {
            'builder': 'cooccurrence',
            'params': {'window_size': 5},
            'gnn_model': 'gcn',
            'gnn_params': {'hidden_dim': 64, 'num_layers': 2}
        },
        'rag': {
            'retriever': 'simple_bm25',
            'fusion': 'concat',
            'top_k': 5
        },
        'drl': {
            'env': 'active_learning',
            'policy': 'dqn',
            'episodes': 5
        }
    }
    
    # 保存Wrench配置
    wrench_config_path = Path('configs_wrench_demo.yaml')
    import yaml
    wrench_config_path.write_text(yaml.dump(wrench_config, allow_unicode=True), encoding='utf-8')
    
    try:
        results_wrench = run_enhanced_pipeline(str(wrench_config_path))
        print("✓ Wrench配置测试成功 (使用占位实现)")
    except ImportError as e:
        print(f"⚠ Wrench不可用 (预期行为): {e}")
        print("  在有Wrench环境中将使用真实实现")
    except Exception as e:
        print(f"✗ Wrench配置测试失败: {e}")
    
    # 4. 展示结果结构
    print("\n4. 结果文件结构:")
    results_dir = Path('results')
    if results_dir.exists():
        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir():
                print(f"  {exp_dir.name}/")
                for file in exp_dir.iterdir():
                    print(f"    {file.name}")
    
    print("\n=== 演示完成 ===")
    print("主要特性:")
    print("✓ 模块化架构设计")
    print("✓ 多种Label Function支持")  
    print("✓ 完整的pipeline流程")
    print("✓ 详细的日志和统计")
    print("✓ Wrench集成准备")
    print("✓ 配置文件系统")
    print("\n查看 results/ 目录了解详细结果")

if __name__ == '__main__':
    main()
