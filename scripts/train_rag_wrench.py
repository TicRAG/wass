#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG知识库训练脚本
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
    parser = argparse.ArgumentParser(description='训练RAG知识库')
    parser.add_argument('config', help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        logger.info("开始训练RAG知识库...")
        
        # 导入RAG知识库类
        from src.knowledge_base.wrench_full_kb import WRENCHKnowledgeCase, WRENCHRAGKnowledgeBase
        
        # 创建RAG知识库实例
        rag_kb = WRENCHRAGKnowledgeBase()
        
        # 加载增强知识库数据
        kb_file = "data/kb_training_dataset.json"
        if os.path.exists(kb_file):
            with open(kb_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            logger.info(f"加载了 {len(training_data)} 个训练样本")
            
            # 将训练数据转换为RAG知识库格式
            for i, sample in enumerate(training_data):
                import numpy as np

                # 创建WRENCHKnowledgeCase对象
                case = WRENCHKnowledgeCase(
                    workflow_id=sample.get('metadata', {}).get('workflow_id', f'workflow_{i}'),
                    task_count=10,  # 默认值
                    dependency_ratio=0.3,  # 默认值
                    critical_path_length=5,  # 默认值
                    workflow_embedding=np.random.rand(64),  # 随机嵌入向量
                    task_id=sample.get('metadata', {}).get('task_id', f'task_{i}'),
                    task_flops=100000000,  # 默认值
                    task_input_files=5,  # 默认值
                    task_output_files=3,  # 默认值
                    task_dependencies=2,  # 默认值
                    task_children=1,  # 默认值
                    task_features=np.array(sample.get('features', [0.0] * 15)),  # 特征向量
                    available_nodes=["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"],
                    node_capacities={"ComputeHost1": 2.0, "ComputeHost2": 3.0, "ComputeHost3": 2.5, "ComputeHost4": 4.0},
                    node_loads={"ComputeHost1": 0.1, "ComputeHost2": 0.2, "ComputeHost3": 0.15, "ComputeHost4": 0.25},
                    node_features=np.random.rand(4, 10),  # 节点特征
                    scheduler_type=sample.get('metadata', {}).get('scheduler_type', 'Unknown'),
                    chosen_node="ComputeHost1",  # 默认值
                    action_taken=0,  # 默认值
                    task_execution_time=sample.get('outcome', 0),
                    task_wait_time=1000000,  # 默认值
                    workflow_makespan=sample.get('outcome', 0),
                    node_utilization={"ComputeHost1": 0.5, "ComputeHost2": 0.6, "ComputeHost3": 0.4, "ComputeHost4": 0.7},
                    simulation_time=5000000,  # 默认值
                    platform_config="default_platform",  # 默认值
                    metadata=sample.get('metadata', {})  # 元数据
                )
                
                # 添加案例到RAG知识库
                rag_kb.add_case(case)
            
            logger.info(f"成功添加 {len(training_data)} 个案例到RAG知识库")
        else:
            logger.warning(f"训练数据文件不存在: {kb_file}")
            # 创建一个空的知识库
            pass
        
        # 保存RAG知识库
        output_file = "data/wrench_rag_knowledge_base.json"
        rag_kb.save_knowledge_base(output_file)
        logger.info(f"RAG知识库已保存到: {output_file}")
        
        logger.info("RAG知识库训练完成!")
        return 0
    except Exception as e:
        logger.error(f"训练RAG知识库时出错: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())