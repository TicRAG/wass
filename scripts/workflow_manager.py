#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WASS-RAG 工作流管理器
统一管理训练和实验中使用的工作流，确保生成方式一致
"""

import os
import sys
import json
import yaml
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.workflow_generator import WorkflowGenerator

class WorkflowManager:
    """工作流管理器，统一管理工作流的生成、配置和使用"""
    
    def __init__(self, config_path: str = "configs/workflow_config.yaml"):
        """初始化工作流管理器"""
        self.config_path = config_path
        self._load_config()
        
        # 初始化工作流生成器
        self.generator = WorkflowGenerator(
            output_dir=self.config.get('workflow_dir', 'workflows'),
            ccr=self.config.get('ccr', 1.0)
        )
        
        # 确保输出目录存在
        self.workflow_dir = Path(self.config.get('workflow_dir', 'workflows'))
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """加载工作流配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # 如果配置文件不存在，使用默认配置
            self.config = {
                'workflow_dir': 'workflows',
                'patterns': ['montage', 'highly_parallel'],
                'small_sizes': [5, 10, 15, 20],
                'medium_sizes': [50, 100],
                'large_sizes': [200, 500],
                'ccr': 1.0,
                'random_seed': 42
            }
            # 保存默认配置
            os.makedirs(Path(self.config_path).parent, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"📝 创建默认工作流配置文件: {self.config_path}")
    
    def generate_experiment_workflows(self) -> List[str]:
        """生成实验用工作流"""
        # 获取配置的工作流大小
        workflow_sizes = self.config.get('small_sizes', [5, 10, 15, 20])
        patterns = self.config.get('patterns', ['montage'])
        random_seed = self.config.get('random_seed', 42)
        
        generated_files = []
        
        for pattern in patterns:
            for size in workflow_sizes:
                # 生成用于实验的工作流，使用标准格式（支持WrenchExperimentRunner的搜索）
                # 格式1: {pattern}_{size}.json - 用于WrenchExperimentRunner搜索
                # 格式2: {pattern}_{size}_tasks.json - 保持与原生成器兼容
                
                # 先生成标准格式
                filename_std = f"{pattern}_{size}.json"
                file_path = self.generator.generate_single_workflow(
                    pattern=pattern,
                    task_count=size,
                    random_seed=random_seed,
                    filename=filename_std
                )
                generated_files.append(file_path)
                
                # 再生成兼容格式（如果不存在）
                filename_compat = f"{pattern}_{size}_tasks.json"
                if not (self.workflow_dir / filename_compat).exists():
                    # 创建符号链接以避免重复文件
                    try:
                        os.symlink(filename_std, self.workflow_dir / filename_compat)
                        print(f"🔗 创建兼容链接: {filename_compat} -> {filename_std}")
                    except OSError:
                        # 如果不支持符号链接，就复制文件
                        with open(self.workflow_dir / filename_std, 'r') as f:
                            content = json.load(f)
                        with open(self.workflow_dir / filename_compat, 'w') as f:
                            json.dump(content, f, indent=2)
                        print(f"📋 创建兼容文件: {filename_compat}")
        
        return generated_files
    
    def generate_training_workflows(self) -> List[str]:
        """生成训练用工作流"""
        # 训练工作流可以使用中等规模，增加训练数据多样性
        workflow_sizes = self.config.get('medium_sizes', [50, 100])
        patterns = self.config.get('patterns', ['montage'])
        random_seed = self.config.get('random_seed', 42)
        
        # 为了增加训练多样性，使用不同的随机种子
        generated_files = []
        
        for pattern in patterns:
            for size in workflow_sizes:
                for seed_offset in range(3):  # 为每种规模生成3个不同的随机变体
                    filename = f"{pattern}_{size}_seed{random_seed + seed_offset}_training.json"
                    file_path = self.generator.generate_single_workflow(
                        pattern=pattern,
                        task_count=size,
                        random_seed=random_seed + seed_offset,
                        filename=filename
                    )
                    generated_files.append(file_path)
        
        return generated_files
    
    def generate_all_workflows(self) -> Dict[str, List[str]]:
        """生成所有需要的工作流"""
        print("🚀 开始生成所有工作流...")
        
        # 生成实验用工作流
        exp_workflows = self.generate_experiment_workflows()
        print(f"✅ 实验工作流生成完成: {len(exp_workflows)} 个文件")
        
        # 生成训练用工作流
        train_workflows = self.generate_training_workflows()
        print(f"✅ 训练工作流生成完成: {len(train_workflows)} 个文件")
        
        # 创建工作流清单
        self._create_workflow_inventory(exp_workflows, train_workflows)
        
        return {
            'experiment': exp_workflows,
            'training': train_workflows
        }
    
    def _create_workflow_inventory(self, exp_workflows: List[str], train_workflows: List[str]):
        """创建工作流清单"""
        inventory = {
            'experiment_workflows': [Path(f).name for f in exp_workflows],
            'training_workflows': [Path(f).name for f in train_workflows],
            'config': self.config
        }
        
        inventory_path = self.workflow_dir / "workflow_inventory.json"
        with open(inventory_path, 'w', encoding='utf-8') as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)
        
        print(f"📋 工作流清单已保存: {inventory_path}")
    
    def get_workflow_paths(self, size: int) -> List[str]:
        """获取特定大小的工作流路径"""
        pattern = f"*_{size}.json"
        paths = list(self.workflow_dir.glob(pattern))
        return [str(p) for p in paths]
    
    def validate_workflows(self) -> bool:
        """验证所有工作流文件是否有效"""
        valid = True
        
        for file_path in self.workflow_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # 检查必要字段
                assert 'metadata' in data and 'workflow' in data
                assert 'tasks' in data['workflow'] and 'files' in data['workflow']
            except Exception as e:
                print(f"❌ 工作流文件无效: {file_path} - {e}")
                valid = False
        
        return valid

def update_experiment_config():
    """更新实验配置文件，确保使用统一的工作流设置"""
    experiment_config_path = "configs/real_heuristic_experiment.yaml"
    
    try:
        with open(experiment_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 确保工作流配置与工作流管理器一致
        workflow_manager = WorkflowManager()
        
        # 更新工作流目录配置
        config['workflow_dir'] = workflow_manager.config.get('workflow_dir', 'workflows')
        
        # 仅在配置中没有指定工作流大小时才设置默认值
        if 'workflow_sizes' not in config:
            config['workflow_sizes'] = workflow_manager.config.get('small_sizes', [5, 10, 15, 20])
        
        # 保存更新后的配置
        with open(experiment_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 实验配置已更新: {experiment_config_path}")
        return True
    except Exception as e:
        print(f"❌ 更新实验配置失败: {e}")
        return False


def update_drl_config():
    """更新DRL训练配置，确保与实验使用统一的工作流设置"""
    drl_config_path = "configs/drl.yaml"
    
    try:
        with open(drl_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 确保工作流配置与工作流管理器一致
        workflow_manager = WorkflowManager()
        
        # 更新平台文件路径，保持与实验一致
        experiment_config_path = "configs/real_heuristic_experiment.yaml"
        if os.path.exists(experiment_config_path):
            with open(experiment_config_path, 'r', encoding='utf-8') as f_exp:
                exp_config = yaml.safe_load(f_exp)
                if 'platform_file' in exp_config:
                    config['platform_file'] = exp_config['platform_file']
        
        # 获取工作流管理器中的小规模工作流大小
        small_sizes = workflow_manager.config.get('small_sizes', [5, 10, 15, 20])
        if small_sizes:
            # 设置任务范围为最小和最大的工作流大小
            config['task_range'] = [min(small_sizes), max(small_sizes)]
        
        # 保存更新后的配置
        with open(drl_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ DRL训练配置已更新: {drl_config_path}")
        return True
    except Exception as e:
        print(f"❌ 更新DRL训练配置失败: {e}")
        return False

def main():
    """主函数"""
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='WASS-RAG 工作流管理器')
    parser.add_argument('--action', choices=['generate', 'validate', 'update_config', 'update_all_configs'], 
                        default='generate',
                        help='执行的操作: 生成工作流、验证工作流、更新实验配置或更新所有配置')
    parser.add_argument('--config', default='configs/workflow_config.yaml',
                        help='工作流配置文件路径')
    
    args = parser.parse_args()
    
    if args.action == 'generate':
        workflow_manager = WorkflowManager(args.config)
        workflow_manager.generate_experiment_workflows()
        workflow_manager.generate_training_workflows()
    elif args.action == 'validate':
        workflow_manager = WorkflowManager(args.config)
        workflow_manager.validate_workflows()
    elif args.action == 'update_config':
        # 仅更新实验配置
        if update_experiment_config():
            print("✅ 实验配置更新完成!")
        else:
            sys.exit(1)
    elif args.action == 'update_all_configs':
        # 更新所有配置文件以确保一致性
        if update_experiment_config() and update_drl_config():
            print("✅ 所有配置更新完成!")
        else:
            sys.exit(1)
    else:
        print(f"未知的操作: {args.action}")

if __name__ == "__main__":
    main()