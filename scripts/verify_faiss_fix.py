#!/usr/bin/env python3
"""
测试修复后的AI模型初始化脚本
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))

def main():
    """运行简化的初始化测试"""
    
    print("=== FAISS修复验证 ===")
    
    try:
        # 导入我们的模块
        from src.ai_schedulers import RAGKnowledgeBase
        import numpy as np
        
        print("1. 创建知识库...")
        kb = RAGKnowledgeBase(embedding_dim=32)
        
        print("2. 添加测试案例...")
        # 创建测试embedding
        test_embedding = np.random.random(32).astype(np.float32)
        
        # 添加案例 - 这里应该不再报错
        kb.add_case(
            embedding=test_embedding,
            workflow_info={"name": "test_workflow"},
            actions=[{"action": "test"}],
            makespan=10.0
        )
        
        print("3. 测试查询...")
        query_embedding = np.random.random(32).astype(np.float32)
        results = kb.retrieve_similar_cases(query_embedding, top_k=1)
        
        print(f"   ✓ 查询成功! 找到 {len(results['similar_cases'])} 个相似案例")
        
        print("\n🎉 FAISS修复验证成功!")
        print("现在可以安全运行完整的初始化脚本")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
