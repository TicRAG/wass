#!/usr/bin/env python3
"""
测试增强的RAG知识库生成器
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.enhanced_rag_kb_generator import EnhancedRAGKnowledgeBaseGenerator, EnhancedWorkflowConfig
from src.knowledge_base.wrench_full_kb import WRENCHRAGKnowledgeBase

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_rag_kb_generator():
    """测试增强的RAG知识库生成器"""
    logger.info("Testing EnhancedRAGKnowledgeBaseGenerator...")
    
    # 创建生成器
    generator = EnhancedRAGKnowledgeBaseGenerator()
    
    # 测试工作流配置生成
    logger.info("Testing workflow configuration generation...")
    configs = generator.workflow_configs
    assert len(configs) > 0, "No workflow configurations generated"
    logger.info(f"Generated {len(configs)} workflow configurations")
    
    # 检查配置多样性
    workflow_types = set(config.workflow_type for config in configs)
    logger.info(f"Workflow types: {workflow_types}")
    assert len(workflow_types) > 1, "Insufficient workflow type diversity"
    
    # 测试小规模知识库生成
    logger.info("Testing small-scale knowledge base generation...")
    kb = generator.generate_enhanced_knowledge_base(num_cases=50)
    assert len(kb.cases) == 50, f"Expected 50 cases, got {len(kb.cases)}"
    logger.info(f"Successfully generated knowledge base with {len(kb.cases)} cases")
    
    # 测试知识库保存和加载
    logger.info("Testing knowledge base save and load...")
    generator.save_knowledge_base(kb, "test_enhanced_rag_kb.json")
    loaded_kb = generator.load_knowledge_base("test_enhanced_rag_kb.json")
    assert loaded_kb is not None, "Failed to load knowledge base"
    assert len(loaded_kb.cases) == len(kb.cases), "Loaded knowledge base has different number of cases"
    logger.info("Knowledge base save and load test passed")
    
    # 测试案例多样性
    logger.info("Testing case diversity...")
    workflow_types_in_kb = set(case.metadata.get('workflow_type', 'unknown') for case in kb.cases)
    scheduler_types_in_kb = set(case.scheduler_type for case in kb.cases)
    logger.info(f"Workflow types in KB: {workflow_types_in_kb}")
    logger.info(f"Scheduler types in KB: {scheduler_types_in_kb}")
    assert len(workflow_types_in_kb) > 1, "Insufficient workflow type diversity in knowledge base"
    assert len(scheduler_types_in_kb) > 1, "Insufficient scheduler type diversity in knowledge base"
    
    # 测试特征计算
    logger.info("Testing feature computation...")
    case = kb.cases[0]
    assert case.workflow_embedding is not None, "Workflow embedding is None"
    assert case.task_features is not None, "Task features are None"
    assert case.node_features is not None, "Node features are None"
    assert len(case.workflow_embedding) == 64, "Workflow embedding dimension is not 64"
    assert len(case.task_features) == 64, "Task features dimension is not 64"
    assert len(case.node_features) == 64, "Node features dimension is not 64"
    logger.info("Feature computation test passed")
    
    # 测试相似度计算
    logger.info("Testing similarity computation...")
    query_case = kb.cases[0]
    similar_cases = kb.find_similar_cases(query_case, top_k=5)
    assert len(similar_cases) == 5, f"Expected 5 similar cases, got {len(similar_cases)}"
    logger.info(f"Found {len(similar_cases)} similar cases")
    
    # 测试RAG建议生成
    logger.info("Testing RAG suggestion generation...")
    suggestions = kb.get_rag_suggestions(query_case, top_k=3)
    assert len(suggestions) == 3, f"Expected 3 suggestions, got {len(suggestions)}"
    logger.info(f"Generated {len(suggestions)} RAG suggestions")
    
    # 清理测试文件
    test_file = Path("data/test_enhanced_rag_kb.json")
    if test_file.exists():
        test_file.unlink()
        logger.info("Cleaned up test file")
    
    logger.info("All tests passed!")

def test_enhanced_workflow_config():
    """测试增强的工作流配置"""
    logger.info("Testing EnhancedWorkflowConfig...")
    
    # 创建测试配置
    config = EnhancedWorkflowConfig(
        workflow_size=10,
        ccr=1.0,
        dependency_probability=0.5,
        workflow_type="dag",
        computation_size_range=(1e9, 1e11),
        data_size_range=(1e6, 1e9)
    )
    
    # 验证配置属性
    assert config.workflow_size == 10, "Workflow size not set correctly"
    assert config.ccr == 1.0, "CCR not set correctly"
    assert config.dependency_probability == 0.5, "Dependency probability not set correctly"
    assert config.workflow_type == "dag", "Workflow type not set correctly"
    assert config.computation_size_range == (1e9, 1e11), "Computation size range not set correctly"
    assert config.data_size_range == (1e6, 1e9), "Data size range not set correctly"
    
    logger.info("EnhancedWorkflowConfig test passed")

def test_edge_cases():
    """测试边界情况"""
    logger.info("Testing edge cases...")
    
    # 创建生成器
    generator = EnhancedRAGKnowledgeBaseGenerator()
    
    # 测试极小知识库
    logger.info("Testing minimal knowledge base...")
    minimal_kb = generator.generate_enhanced_knowledge_base(num_cases=1)
    assert len(minimal_kb.cases) == 1, "Failed to generate minimal knowledge base"
    logger.info("Minimal knowledge base test passed")
    
    # 测试空知识库
    logger.info("Testing empty knowledge base...")
    empty_kb = WRENCHRAGKnowledgeBase(embedding_dim=64)
    assert len(empty_kb.cases) == 0, "Empty knowledge base is not empty"
    
    # 测试在空知识库上查找相似案例
    query_case = minimal_kb.cases[0]
    similar_cases = empty_kb.find_similar_cases(query_case, top_k=5)
    assert len(similar_cases) == 0, "Empty knowledge base returned similar cases"
    logger.info("Empty knowledge base test passed")
    
    logger.info("Edge cases test passed")

def main():
    """主函数：运行所有测试"""
    logger.info("Starting enhanced RAG knowledge base generator tests...")
    
    try:
        test_enhanced_workflow_config()
        test_enhanced_rag_kb_generator()
        test_edge_cases()
        
        logger.info("All tests completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)