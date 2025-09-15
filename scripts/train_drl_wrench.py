#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRL智能体训练脚本包装器
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """加载配置文件"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练DRL智能体')
    parser.add_argument('config', help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        logger.info("开始训练DRL智能体...")
        
        # 导入并运行改进的DRL训练器
        from scripts.improved_drl_trainer import WRENCHDRLTrainer
        
        # 初始化训练器
        trainer = WRENCHDRLTrainer(args.config)
        
        # 运行训练
        trainer.train()
        
        logger.info("DRL智能体训练完成!")
        return 0
    except Exception as e:
        logger.error(f"训练DRL智能体时出错: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())