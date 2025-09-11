#!/usr/bin/env python3
"""
使用调优后的最佳超参数运行完整WASS-RAG实验

基于超参数调优结果，运行完整的5调度器对比实验
"""

import os
import yaml
import json
import time
from pathlib import Path
import sys

# 添加项目路径
sys.path.append('/data/workspace/wass/src')
sys.path.append('/data/workspace/wass/scripts')
sys.path.append('/data/workspace/wass/experiments')


def update_drl_config_with_optimized_params():
    """使用调优后的最佳参数更新DRL配置文件"""
    
    # 读取调优后的最佳配置
    tuned_config_path = "/data/workspace/wass/results/local_hyperparameter_tuning/best_hyperparameters_for_training.yaml"
    
    if not os.path.exists(tuned_config_path):
        print("❌ 未找到调优后的配置文件")
        return False
    
    with open(tuned_config_path, 'r') as f:
        tuned_config = yaml.safe_load(f)
    
    # 读取原始DRL配置
    original_config_path = "/data/workspace/wass/configs/drl.yaml"
    with open(original_config_path, 'r') as f:
        drl_config = yaml.safe_load(f)
    
    # 更新配置
    print("🔄 更新DRL配置文件...")
    print(f"  原配置学习率: {drl_config.get('learning_rate', 'N/A')}")
    print(f"  调优后学习率: {tuned_config['training']['learning_rate']}")
    
    # 更新核心训练参数
    drl_config.update({
        'learning_rate': tuned_config['training']['learning_rate'],
        'gamma': tuned_config['training']['gamma'],
        'epsilon_start': tuned_config['training']['epsilon_start'],
        'epsilon_end': tuned_config['training']['epsilon_end'],
        'epsilon_decay': tuned_config['training']['epsilon_decay'],
        'batch_size': tuned_config['training']['batch_size'],
        'memory_size': tuned_config['training']['memory_size'],
        'target_update_freq': tuned_config['training']['target_update_freq']
    })
    
    # 更新网络结构
    drl_config['model'] = tuned_config['model']
    
    # 更新奖励权重
    drl_config['reward_weights'] = tuned_config['reward_weights']
    
    # 备份原配置
    backup_path = "/data/workspace/wass/configs/drl_backup.yaml"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(original_config_path, backup_path)
        print(f"📦 原配置已备份到: {backup_path}")
    
    # 保存更新后的配置
    with open(original_config_path, 'w') as f:
        yaml.dump(drl_config, f, default_flow_style=False, indent=2)
    
    print("✅ DRL配置已更新为调优后的最佳参数")
    
    # 显示关键配置更新
    print("\n📊 关键配置更新摘要:")
    print(f"  学习率: {tuned_config['training']['learning_rate']}")
    print(f"  折扣因子: {tuned_config['training']['gamma']}")
    print(f"  网络结构: {tuned_config['model']['hidden_layers']}")
    print(f"  批次大小: {tuned_config['training']['batch_size']}")
    print(f"  关键路径权重: {tuned_config['reward_weights']['critical_path_weight']}")
    
    return True


def run_drl_training_with_optimized_config():
    """使用调优后的配置进行DRL训练"""
    print("\n🚀 开始使用调优配置训练WASS-DRL模型...")
    
    # 检查是否存在训练脚本
    training_scripts = [
        "/data/workspace/wass/scripts/train_drl_agent.py",
        "/data/workspace/wass/scripts/retrain_performance_predictor.py"
    ]
    
    for script in training_scripts:
        if os.path.exists(script):
            print(f"📁 找到训练脚本: {script}")
            
            # 运行训练
            cmd = f"cd /data/workspace/wass && python {script}"
            print(f"🔄 执行: {cmd}")
            
            result = os.system(cmd)
            if result == 0:
                print(f"✅ {script} 执行成功")
            else:
                print(f"⚠️ {script} 执行失败 (返回码: {result})")
        else:
            print(f"❌ 未找到: {script}")


def run_complete_experiment():
    """运行完整的5调度器对比实验"""
    print("\n🧪 开始运行完整的WASS-RAG实验...")
    
    # 检查实验脚本
    experiment_script = "/data/workspace/wass/experiments/wrench_real_experiment.py"
    
    if not os.path.exists(experiment_script):
        print(f"❌ 未找到实验脚本: {experiment_script}")
        return False
    
    print(f"📁 找到实验脚本: {experiment_script}")
    
    # 执行实验
    cmd = f"cd /data/workspace/wass && python {experiment_script}"
    print(f"🔄 执行完整实验: {cmd}")
    
    start_time = time.time()
    result = os.system(cmd)
    end_time = time.time()
    
    if result == 0:
        print(f"✅ 完整实验执行成功 (耗时: {end_time - start_time:.1f}秒)")
        return True
    else:
        print(f"❌ 实验执行失败 (返回码: {result})")
        return False


def analyze_experiment_results():
    """分析实验结果"""
    print("\n📊 分析实验结果...")
    
    # 查找结果文件
    results_dirs = [
        "/data/workspace/wass/results/final_experiments_discrete_event/",
        "/data/workspace/wass/results/"
    ]
    
    results_found = False
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            print(f"📁 检查结果目录: {results_dir}")
            
            # 列出结果文件
            for file in os.listdir(results_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(results_dir, file)
                    print(f"  📄 找到结果文件: {file}")
                    
                    # 简单分析
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list) and len(data) > 0:
                                print(f"    📈 结果记录数: {len(data)}")
                                
                                # 统计各调度器性能
                                methods = set()
                                for record in data:
                                    if 'method' in record:
                                        methods.add(record['method'])
                                
                                print(f"    🎯 调度器种类: {', '.join(methods)}")
                                results_found = True
                                
                    except Exception as e:
                        print(f"    ⚠️ 读取失败: {e}")
    
    if not results_found:
        print("❌ 未找到有效的实验结果文件")
    
    return results_found


def generate_performance_comparison():
    """生成性能对比报告"""
    print("\n📈 生成性能对比报告...")
    
    # 检查结果分析脚本
    analysis_scripts = [
        "/data/workspace/wass/charts/verify_real_data.py",
        "/data/workspace/wass/analyze_data_issues.py",
        "/data/workspace/wass/fix_chart_issues.py"
    ]
    
    for script in analysis_scripts:
        if os.path.exists(script):
            print(f"📁 找到分析脚本: {script}")
            
            cmd = f"cd /data/workspace/wass && python {script}"
            result = os.system(cmd)
            
            if result == 0:
                print(f"✅ {os.path.basename(script)} 执行成功")
            else:
                print(f"⚠️ {os.path.basename(script)} 执行失败")


def main():
    """主函数：运行完整的调优后实验流程"""
    print("🎯 开始运行调优后的完整WASS-RAG实验流程")
    print("=" * 60)
    
    success_steps = 0
    total_steps = 5
    
    # 步骤1: 更新配置
    print(f"\n📋 步骤 1/{total_steps}: 更新DRL配置")
    if update_drl_config_with_optimized_params():
        success_steps += 1
        print("✅ 步骤1完成")
    else:
        print("❌ 步骤1失败")
    
    # 步骤2: DRL训练 (可选，如果需要重新训练)
    print(f"\n📋 步骤 2/{total_steps}: DRL模型训练")
    print("💡 提示: 如果已有训练好的模型，可以跳过此步骤")
    user_input = input("是否重新训练DRL模型? (y/N): ").strip().lower()
    
    if user_input == 'y':
        run_drl_training_with_optimized_config()
        success_steps += 1
    else:
        print("⏭️ 跳过DRL训练，使用现有模型")
        success_steps += 1
    
    # 步骤3: 运行完整实验
    print(f"\n📋 步骤 3/{total_steps}: 运行完整5调度器对比实验")
    if run_complete_experiment():
        success_steps += 1
        print("✅ 步骤3完成")
    else:
        print("❌ 步骤3失败")
    
    # 步骤4: 分析结果
    print(f"\n📋 步骤 4/{total_steps}: 分析实验结果")
    if analyze_experiment_results():
        success_steps += 1
        print("✅ 步骤4完成")
    else:
        print("❌ 步骤4失败")
    
    # 步骤5: 生成对比报告
    print(f"\n📋 步骤 5/{total_steps}: 生成性能对比报告")
    generate_performance_comparison()
    success_steps += 1
    print("✅ 步骤5完成")
    
    # 总结
    print("\n" + "=" * 60)
    print(f"🎉 实验流程完成! 成功步骤: {success_steps}/{total_steps}")
    
    if success_steps == total_steps:
        print("🏆 所有步骤都成功完成!")
        print("📊 实验结果已生成，可以开始论文撰写")
    else:
        print(f"⚠️ 有 {total_steps - success_steps} 个步骤需要注意")
    
    # 显示结果位置
    print("\n📁 关键文件位置:")
    print("  - 调优配置: /data/workspace/wass/results/local_hyperparameter_tuning/")
    print("  - 实验结果: /data/workspace/wass/results/")
    print("  - DRL配置: /data/workspace/wass/configs/drl.yaml")


if __name__ == "__main__":
    main()
