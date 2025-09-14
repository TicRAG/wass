#!/usr/bin/env python3
"""
简化净化知识库生成器 - 仅包含HEFT和WassHeuristicScheduler
使用模拟数据避免wrench API兼容性问题
"""

import json
import os
import random
import numpy as np

def generate_simple_curated_kb():
    """生成简化的净化知识库"""
    
    print("🧹 生成净化知识库 - 仅HEFT和WassHeuristic...")
    
    # 节点配置
    nodes = ['ComputeHost1', 'ComputeHost2', 'ComputeHost3', 'ComputeHost4']
    node_speeds = {'ComputeHost1': 2.0, 'ComputeHost2': 3.0, 'ComputeHost3': 2.5, 'ComputeHost4': 4.0}
    
    # 工作流模式
    patterns = ['montage', 'ligo', 'cybershake', 'sipht', 'genome']
    
    samples = []
    
    # 生成HEFT样本
    print("📊 生成HEFT样本...")
    for i in range(1200):  # 1200个HEFT样本
        # 随机任务特征
        task_flops = random.uniform(50, 500)
        input_files = random.randint(0, 5)
        children = random.randint(0, 3)
        avg_speed = np.mean(list(node_speeds.values()))
        
        # 为每个节点选择生成样本
        for node_idx, node in enumerate(nodes):
            speed = node_speeds[node]
            finish_time = task_flops / speed
            
            state_features = [task_flops, input_files, children, avg_speed, len(nodes)]
            action_features = [0.0] * len(nodes)
            action_features[node_idx] = 1.0
            context_features = [0.0] * 8
            
            samples.append({
                'scheduler': 'HEFT',
                'state_features': state_features,
                'action_features': action_features,
                'context_features': context_features,
                'achieved_finish_time': finish_time,
                'meta': {
                    'task_id': f'heft_task_{i}_{node}',
                    'host': node,
                    'workflow_id': f'heft_wf_{i//20}'
                }
            })
    
    # 生成WassHeuristic样本
    print("🎯 生成WassHeuristic样本...")
    for i in range(1200):  # 1200个WassHeuristic样本
        # 随机任务特征
        task_flops = random.uniform(50, 500)
        input_files = random.randint(0, 5)
        children = random.randint(0, 3)
        avg_speed = np.mean(list(node_speeds.values()))
        
        # 为每个节点选择生成样本（使用启发式分数）
        for node_idx, node in enumerate(nodes):
            speed = node_speeds[node]
            # WassHeuristic启发式分数：考虑数据局部性和计算能力
            heuristic_score = (speed / avg_speed) * (1.0 / (1.0 + input_files * 0.1))
            finish_time = task_flops / speed
            
            state_features = [task_flops, input_files, children, avg_speed, len(nodes)]
            action_features = [0.0] * len(nodes)
            action_features[node_idx] = 1.0
            context_features = [0.0] * 8
            
            samples.append({
                'scheduler': 'WassHeuristic',
                'state_features': state_features,
                'action_features': action_features,
                'context_features': context_features,
                'heuristic_score': heuristic_score,
                'achieved_finish_time': finish_time,
                'meta': {
                    'task_id': f'heuristic_task_{i}_{node}',
                    'host': node,
                    'workflow_id': f'heuristic_wf_{i//20}'
                }
            })
    
    # 保存净化知识库
    os.makedirs('data', exist_ok=True)
    output_path = 'data/curated_kb_training_dataset.json'
    
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # 生成统计信息
    scheduler_counts = {}
    for sample in samples:
        sched = sample['scheduler']
        scheduler_counts[sched] = scheduler_counts.get(sched, 0) + 1
    
    print(f"✅ 净化知识库生成完成!")
    print(f"📊 总计样本: {len(samples)}")
    print(f"🎯 调度器分布:")
    for sched, count in scheduler_counts.items():
        print(f"   {sched}: {count} 个样本")
    
    # 创建元数据文件
    metadata = {
        'total_samples': len(samples),
        'scheduler_distribution': scheduler_counts,
        'features_dim': {
            'state': 5,
            'action': 4,
            'context': 8
        },
        'generated_at': '2025-09-14',
        'description': '净化后的知识库 - 仅包含HEFT和WassHeuristicScheduler'
    }
    
    with open('data/curated_kb_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_path

if __name__ == '__main__':
    generate_simple_curated_kb()