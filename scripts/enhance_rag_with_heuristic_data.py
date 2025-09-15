#!/usr/bin/env python3
"""
将HEFT和WASS-Heuristic的学习数据整合到RAG知识库中
创建混合知识库，结合启发式调度器的优秀决策
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_base.wrench_cases import WrenchKnowledgeCaseMinimal
from src.knowledge_base.wrench_full_kb import WRENCHKnowledgeCase

class EnhancedRAGBuilder:
    """构建增强版RAG知识库"""
    
    def __init__(self):
        self.heuristic_cases = []
        self.original_cases = []
        self.enhanced_cases = []
    
    def load_heuristic_data(self, filepath: str) -> List[Dict[str, Any]]:
        """加载启发式学习数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_original_kb(self, filepath: str) -> List[Dict[str, Any]]:
        """加载原始知识库"""
        if not os.path.exists(filepath):
            return []
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    
    def convert_to_rag_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """将学习案例转换为RAG格式"""
        # 节点映射
        node_map = {
            'ComputeHost1': 0,
            'ComputeHost2': 1,
            'ComputeHost3': 2,
            'ComputeHost4': 3
        }
        
        # 计算任务特征
        task_features = [
            case['task_flops'] / 1e9,  # 归一化计算量
            1.0,  # 任务优先级（启发式为1）
            case['task_execution_time'] / 10.0,  # 归一化执行时间
            len(case['task_id'].split('_')) * 0.1,  # 任务层级
            np.random.uniform(0.1, 1.0)  # 随机通信量
        ]
        
        # 计算节点特征
        chosen_node_idx = node_map.get(case['chosen_node'], 0)
        node_features = [0.0] * 4
        node_features[chosen_node_idx] = 1.0
        
        # 计算工作流嵌入
        workflow_embedding = [
            case['task_count'] * 0.1,
            case['workflow_makespan'] / 20.0,
            case['quality_score'],
            np.random.uniform(0.1, 0.9)
        ]
        
        # 计算奖励（基于makespan的倒数）
        reward = 1.0 / max(case['workflow_makespan'], 1.0)
        
        return {
            'workflow_id': case['workflow_id'],
            'task_id': case['task_id'],
            'scheduler_type': case['scheduler_type'],
            'chosen_node': case['chosen_node'],
            'chosen_node_idx': chosen_node_idx,
            'task_flops': case['task_flops'],
            'task_execution_time': case['task_execution_time'],
            'workflow_makespan': case['workflow_makespan'],
            'task_features': task_features,
            'node_features': node_features,
            'workflow_embedding': workflow_embedding,
            'reward': reward,
            'quality_score': case['quality_score']
        }
    
    def create_balanced_dataset(self, heuristic_cases: List[Dict[str, Any]], 
                              target_size: int = 5000) -> List[Dict[str, Any]]:
        """创建平衡的数据集"""
        enhanced_cases = []
        
        # 添加启发式案例（高权重）
        heuristic_weight = 3.0  # 启发式案例权重更高
        for case in heuristic_cases:
            converted = self.convert_to_rag_case(case)
            converted['weight'] = heuristic_weight
            enhanced_cases.append(converted)
        
        # 计算需要生成的合成案例数量
        heuristic_count = len(heuristic_cases)
        synthetic_needed = max(0, target_size - int(heuristic_count * heuristic_weight))
        
        print(f"启发式案例: {heuristic_count}")
        print(f"需要合成案例: {synthetic_needed}")
        
        # 生成合成案例（基于启发式数据分布）
        if synthetic_needed > 0:
            synthetic_cases = self.generate_synthetic_cases(
                heuristic_cases, synthetic_needed
            )
            enhanced_cases.extend(synthetic_cases)
        
        return enhanced_cases
    
    def generate_synthetic_cases(self, base_cases: List[Dict[str, Any]], 
                               count: int) -> List[Dict[str, Any]]:
        """基于启发式数据生成合成案例"""
        synthetic_cases = []
        
        if not base_cases:
            return synthetic_cases
        
        # 计算统计数据
        makespans = [c['workflow_makespan'] for c in base_cases]
        task_times = [c['task_execution_time'] for c in base_cases]
        nodes = [c['chosen_node'] for c in base_cases]
        
        avg_makespan = np.mean(makespans)
        std_makespan = np.std(makespans)
        avg_task_time = np.mean(task_times)
        std_task_time = np.std(task_times)
        
        # 节点分布
        node_counts = {}
        for node in nodes:
            node_counts[node] = node_counts.get(node, 0) + 1
        
        # 生成合成案例
        for i in range(count):
            # 从分布中采样
            makespan = max(1.0, np.random.normal(avg_makespan, std_makespan * 0.3))
            task_time = max(0.1, np.random.normal(avg_task_time, std_task_time * 0.3))
            
            # 根据节点分布选择节点
            nodes_list = list(node_counts.keys())
            weights = [node_counts[n] for n in nodes_list]
            chosen_node = np.random.choice(nodes_list, p=[w/sum(weights) for w in weights])
            
            # 生成合成特征
            task_flops = task_time * 1e9 * np.random.uniform(0.8, 1.2)
            
            synthetic_case = {
                'workflow_id': f'synthetic_{i}',
                'task_id': f'synthetic_task_{i}',
                'scheduler_type': 'Synthetic',
                'chosen_node': chosen_node,
                'task_flops': task_flops,
                'task_execution_time': task_time,
                'workflow_makespan': makespan,
                'task_count': int(np.random.normal(np.mean([c['task_count'] for c in base_cases]), 2)),
                'quality_score': 1.0 / max(makespan, 1.0),
                'weight': 1.0
            }
            
            synthetic_cases.append(self.convert_to_rag_case(synthetic_case))
        
        return synthetic_cases
    
    def save_enhanced_kb(self, cases: List[Dict[str, Any]], output_path: str):
        """保存增强知识库"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为标准格式
        kb_data = {
            'metadata': {
                'total_cases': len(cases),
                'heuristic_cases': len([c for c in cases if c['scheduler_type'] in ['HEFT', 'WASS-Heuristic']]),
                'synthetic_cases': len([c for c in cases if c['scheduler_type'] == 'Synthetic']),
                'created_at': '2025-09-15'
            },
            'cases': cases
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(kb_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存增强知识库到 {output_file}")
    
    def build_enhanced_kb(self, heuristic_data_path: str, output_path: str, 
                         target_size: int = 5000):
        """构建完整的增强知识库"""
        
        # 加载启发式数据
        print("正在加载启发式数据...")
        heuristic_cases = self.load_heuristic_data(heuristic_data_path)
        print(f"加载了 {len(heuristic_cases)} 个启发式案例")
        
        # 创建增强数据集
        print("正在创建增强数据集...")
        enhanced_cases = self.create_balanced_dataset(
            heuristic_cases, target_size
        )
        
        # 打乱数据顺序
        import random
        random.shuffle(enhanced_cases)
        
        # 保存增强知识库
        self.save_enhanced_kb(enhanced_cases, output_path)
        
        # 生成统计信息
        heuristic_count = len([c for c in enhanced_cases if c['scheduler_type'] in ['HEFT', 'WASS-Heuristic']])
        synthetic_count = len([c for c in enhanced_cases if c['scheduler_type'] == 'Synthetic'])
        
        print(f"\n增强知识库构建完成!")
        print(f"总案例数: {len(enhanced_cases)}")
        print(f"启发式案例: {heuristic_count}")
        print(f"合成案例: {synthetic_count}")

def main():
    """主函数"""
    builder = EnhancedRAGBuilder()
    
    try:
        builder.build_enhanced_kb(
            heuristic_data_path="data/heuristic_learning_data.json",
            output_path="data/enhanced_rag_knowledge_base.json",
            target_size=5000
        )
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())