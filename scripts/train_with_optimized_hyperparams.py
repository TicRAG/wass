#!/usr/bin/env python3
"""
使用调优后的最佳超参数重新训练WASS-DRL模型

基于超参数调优结果，使用最优配置进行正式训练
"""

import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.append('/data/workspace/wass/src')
sys.path.append('/data/workspace/wass/scripts')

from improved_drl_trainer import ImprovedDRLTrainer, DRLSchedulingEnvironment


def load_optimized_config():
    """加载调优后的最佳超参数配置"""
    config_path = "/data/workspace/wass/results/local_hyperparameter_tuning/best_hyperparameters_for_training.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("📊 使用调优后的最佳配置:")
    print(f"  学习率: {config['training']['learning_rate']}")
    print(f"  Gamma: {config['training']['gamma']}")
    print(f"  网络结构: {config['model']['hidden_layers']}")
    print(f"  批次大小: {config['training']['batch_size']}")
    print(f"  奖励权重: {config['reward_weights']}")
    print(f"  调优得分: {config['tuning_metadata']['best_score']:.4f}")
    
    return config


def train_optimized_model():
    """使用最优超参数训练模型"""
    print("🚀 开始使用调优后配置训练WASS-DRL模型...")
    
    # 加载最优配置
    config = load_optimized_config()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建环境
    env = DRLSchedulingEnvironment()
    
    # 使用最优超参数创建训练器
    trainer = ImprovedDRLTrainer(
        state_dim=env.get_state_dimension(),
        action_dim=env.get_action_dimension(),
        hidden_layers=config['model']['hidden_layers'],
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        epsilon_start=config['training']['epsilon_start'],
        epsilon_end=config['training']['epsilon_end'],
        epsilon_decay=config['training']['epsilon_decay'],
        dropout_rate=config['model']['dropout_rate'],
        batch_size=config['training']['batch_size'],
        memory_size=config['training']['memory_size'],
        target_update_freq=config['training']['target_update_freq'],
        data_locality_weight=config['reward_weights']['data_locality_weight'],
        waiting_time_weight=config['reward_weights']['waiting_time_weight'],
        critical_path_weight=config['reward_weights']['critical_path_weight'],
        load_balancing_weight=config['reward_weights']['load_balancing_weight']
    )
    
    print("\n🎯 开始训练 (使用密集奖励函数)...")
    
    # 进行完整训练 (更多episodes)
    training_metrics = trainer.train(
        episodes=500,  # 增加训练episodes
        max_steps_per_episode=300,
        verbose=True,
        save_interval=50
    )
    
    # 保存训练好的模型
    model_save_path = "/data/workspace/wass/models/wass_optimized_models.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    trainer.save_model(model_save_path)
    
    print(f"\n✅ 模型训练完成并保存到: {model_save_path}")
    
    # 保存训练报告
    training_report = {
        'hyperparameters': config,
        'training_metrics': training_metrics,
        'model_path': model_save_path,
        'training_completed': True
    }
    
    report_path = "/data/workspace/wass/results/local_hyperparameter_tuning/optimized_training_report.json"
    with open(report_path, 'w') as f:
        json.dump(training_report, f, indent=2, default=str)
    
    print(f"📊 训练报告保存到: {report_path}")
    
    return training_metrics


def main():
    """主函数"""
    try:
        # 训练优化模型
        metrics = train_optimized_model()
        
        print("\n🎉 优化训练完成!")
        print("💡 下一步: 运行完整实验验证调优效果")
        print("   python experiments/wrench_real_experiment.py")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
