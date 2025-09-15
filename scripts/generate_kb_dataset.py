#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库数据集生成脚本
用于生成训练性能预测器所需的数据集
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from scripts.enhanced_rag_kb_generator import EnhancedRAGKnowledgeBaseGenerator
from src.knowledge_base.json_kb import KnowledgeCase

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

def generate_kb_dataset(config_path):
    """生成知识库数据集"""
    logger.info("开始生成知识库数据集...")
    
    # 加载配置
    config = load_config(config_path)
    
    # 初始化知识库生成器
    generator = EnhancedRAGKnowledgeBaseGenerator()
    
    # 获取配置参数
    output_dir = config.get('output_dir', 'data')
    kb_output_path = config.get('kb_output_path', 'data/wrench_rag_knowledge_base.json')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成增强知识库（直接生成5000个案例）
    logger.info("生成增强知识库...")
    enhanced_kb = generator.generate_enhanced_knowledge_base(num_cases=5000)
    
    # 从增强知识库中提取训练数据
    logger.info("提取训练数据...")
    training_data = []
    
    # 遍历知识库中的案例
    for i, case in enumerate(enhanced_kb.cases):
        try:
            # 提取特征和结果
            # 使用任务特征和平台特征作为输入特征
            task_features = case.task_features.tolist() if hasattr(case.task_features, 'tolist') else list(case.task_features)
            platform_features = list(case.node_features) if hasattr(case.node_features, '__iter__') else [case.node_features]
            
            # 合并特征
            features = task_features + platform_features
            
            # 使用实际执行时间作为标签
            outcome = case.task_execution_time
            
            # 构造训练样本
            sample = {
                'features': features,
                'outcome': outcome,
                'metadata': {
                    'workflow_id': case.workflow_id,
                    'task_id': case.task_id,
                    'scheduler_type': case.scheduler_type
                }
            }
            
            training_data.append(sample)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"已处理 {i + 1} 个案例")
                
        except Exception as e:
            logger.warning(f"处理案例 {i} 时出错: {e}")
            continue
    
    # 保存训练数据集
    training_dataset_path = os.path.join(output_dir, 'kb_training_dataset.json')
    logger.info(f"保存训练数据集到: {training_dataset_path}")
    
    with open(training_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"知识库数据集生成完成! 共生成 {len(training_data)} 个训练样本")
    return training_data

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成知识库数据集')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--num-cases', type=int, default=5000, help='生成案例数量')
    
    args = parser.parse_args()
    
    try:
        # 生成知识库数据集
        training_data = generate_kb_dataset(args.config)
        
        logger.info(f"训练数据集生成成功! 共 {len(training_data)} 个样本")
        return 0
    except Exception as e:
        logger.error(f"生成训练数据集时出错: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())