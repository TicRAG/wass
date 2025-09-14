#!/usr/bin/env python3
"""
净化系统设置脚本
1. 生成净化后的知识库（仅HEFT和WassHeuristicScheduler）
2. 验证R_RAG动态奖励机制
3. 准备完整实验环境
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行命令并捕获输出"""
    print(f"运行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False
    print(f"输出: {result.stdout}")
    return True

def setup_curated_system():
    """设置净化后的系统"""
    
    print("🧹 开始净化系统设置...")
    
    # 1. 生成净化后的知识库
    print("\n📚 生成净化知识库...")
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 检查配置文件
    config_path = 'configs/experiment.yaml'
    if not os.path.exists(config_path):
        # 创建基础配置文件
        basic_config = {
            'platform': {
                'platform_file': 'configs/platform.xml',
                'controller_host': 'ControllerHost'
            },
            'kb_generation': {
                'num_tasks': 15,
                'num_workflows': 40,
                'context_dim': 8
            }
        }
        os.makedirs('configs', exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(basic_config, f, indent=2)
    
    # 运行净化知识库生成
    success = run_command(f"{sys.executable} scripts/generate_curated_kb.py configs/experiment.yaml")
    if not success:
        print("❌ 净化知识库生成失败")
        return False
    
    # 2. 验证知识库内容
    print("\n🔍 验证净化知识库...")
    try:
        with open('data/curated_kb_training_dataset.json', 'r') as f:
            kb_data = json.load(f)
        
        # 统计调度器分布
        schedulers = {}
        for sample in kb_data:
            sched = sample.get('scheduler', 'Unknown')
            schedulers[sched] = schedulers.get(sched, 0) + 1
        
        print(f"📊 净化知识库统计:")
        for sched, count in schedulers.items():
            print(f"   {sched}: {count} 个样本")
        
        # 确保只包含HEFT和WassHeuristic
        allowed_schedulers = {'HEFT', 'WassHeuristic'}
        actual_schedulers = set(schedulers.keys())
        
        if actual_schedulers.issubset(allowed_schedulers):
            print("✅ 知识库净化成功 - 仅包含HEFT和WassHeuristic")
        else:
            print(f"⚠️  发现额外调度器: {actual_schedulers - allowed_schedulers}")
            
    except Exception as e:
        print(f"❌ 知识库验证失败: {e}")
        return False
    
    # 3. 创建平台配置文件（如果不存在）
    platform_xml = """<?xml version="1.0"?>
<platform version="4.1">
    <zone id="AS0" routing="Full">
        <host id="ComputeHost1" speed="2Gf" core="4"/>
        <host id="ComputeHost2" speed="3Gf" core="4"/>
        <host id="ComputeHost3" speed="2.5Gf" core="4"/>
        <host id="ComputeHost4" speed="4Gf" core="4"/>
        <link id="link1" bandwidth="1GBps" latency="0us"/>
        <link id="link2" bandwidth="1GBps" latency="0us"/>
        <link id="link3" bandwidth="1GBps" latency="0us"/>
        <link id="link4" bandwidth="1GBps" latency="0us"/>
        <route src="ComputeHost1" dst="ComputeHost2"><link_ctn id="link1"/></route>
        <route src="ComputeHost1" dst="ComputeHost3"><link_ctn id="link2"/></route>
        <route src="ComputeHost1" dst="ComputeHost4"><link_ctn id="link3"/></route>
    </zone>
</platform>"""
    
    platform_path = 'configs/platform.xml'
    if not os.path.exists(platform_path):
        os.makedirs('configs', exist_ok=True)
        with open(platform_path, 'w') as f:
            f.write(platform_xml)
        print("✅ 平台配置文件已创建")
    
    # 4. 创建实验配置文件
    experiment_config = {
        'experiment': {
            'name': 'wass_curated_experiment',
            'description': '净化后的WASS实验 - 仅HEFT vs WassHeuristic vs WASS-RAG',
            'schedulers': ['HEFT', 'WassHeuristic', 'WASS-RAG', 'WASS-DRL'],
            'workflows': {
                'count': 33,
                'patterns': ['montage', 'ligo', 'cybershake', 'sipht', 'genome']
            },
            'platforms': {
                'sizes': ['small', 'medium', 'large', 'xlarge']
            }
        },
        'rag': {
            'enabled': True,
            'knowledge_base': 'data/curated_kb_training_dataset.json',
            'reward_alpha': 0.8,
            'epsilon_decay': 0.995
        }
    }
    with open('configs/curated_experiment.yaml', 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    print("\n🎯 创建测试工作流...")
    
    # 5. 创建简单测试工作流
    test_workflow = {
        "workflow": {
            "name": "test_montage",
            "tasks": [
                {"id": "task1", "flops": 100, "input_files": 0, "output_files": 1},
                {"id": "task2", "flops": 200, "input_files": 1, "output_files": 1},
                {"id": "task3", "flops": 150, "input_files": 1, "output_files": 1}
            ],
            "dependencies": [
                {"from": "task1", "to": "task2"},
                {"from": "task2", "to": "task3"}
            ]
        }
    }
    
    os.makedirs('data/workflows', exist_ok=True)
    with open('data/workflows/test_workflow.json', 'w') as f:
        json.dump(test_workflow, f, indent=2)
    
    print("\n✅ 净化系统设置完成！")
    print("\n📋 生成的文件:")
    print("   - data/curated_kb_training_dataset.json (净化知识库)")
    print("   - configs/platform.xml (平台配置)")
    print("   - configs/curated_experiment.yaml (实验配置)")
    print("   - data/workflows/test_workflow.json (测试工作流)")
    
    print("\n🚀 下一步操作:")
    print("   1. 运行: python scripts/train_predictor_from_kb.py configs/curated_experiment.yaml")
    print("   2. 运行: python experiments/wrench_real_experiment.py")
    print("   3. 验证R_RAG动态奖励机制效果")
    
    return True

if __name__ == '__main__':
    success = setup_curated_system()
    if success:
        print("\n🎉 净化系统设置成功完成！")
    else:
        print("\n❌ 净化系统设置失败，请检查错误信息")
        sys.exit(1)