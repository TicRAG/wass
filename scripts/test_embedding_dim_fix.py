#!/usr/bin/env python3
"""
验证embedding_dim修复的测试脚本
"""

import sys
import os
import numpy as np

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_embedding_dim_fix():
    """测试embedding_dim属性修复"""
    
    try:
        from src.ai_schedulers import RAGKnowledgeBase
        
        print("=== 测试embedding_dim修复 ===")
        
        # 1. 测试默认初始化
        print("1. 测试默认初始化...")
        kb1 = RAGKnowledgeBase()
        print(f"   默认embedding_dim: {kb1.embedding_dim}")
        print(f"   FAISS索引维度: {kb1.index.d}")
        
        # 2. 测试指定维度初始化
        print("2. 测试指定维度初始化...")
        kb2 = RAGKnowledgeBase(embedding_dim=64)
        print(f"   指定embedding_dim: {kb2.embedding_dim}")
        print(f"   FAISS索引维度: {kb2.index.d}")
        
        # 3. 测试添加案例（32维）
        print("3. 测试添加32维案例...")
        test_embedding_32 = np.random.random(32).astype(np.float32)
        
        kb1.add_case(
            embedding=test_embedding_32,
            workflow_info={"test": "data"},
            actions=["action1"],
            makespan=10.0
        )
        print(f"   ✓ 成功添加32维案例! 知识库现有 {len(kb1.cases)} 个案例")
        
        # 4. 测试添加案例（64维）
        print("4. 测试添加64维案例...")
        test_embedding_64 = np.random.random(64).astype(np.float32)
        
        kb2.add_case(
            embedding=test_embedding_64,
            workflow_info={"test": "data"},
            actions=["action1"],
            makespan=10.0
        )
        print(f"   ✓ 成功添加64维案例! 知识库现有 {len(kb2.cases)} 个案例")
        
        # 5. 测试维度不匹配（应该报错）
        print("5. 测试维度不匹配...")
        try:
            kb1.add_case(
                embedding=test_embedding_64,  # 64维embedding加到32维知识库
                workflow_info={"test": "data"},
                actions=["action1"],
                makespan=10.0
            )
            print("   ❌ 应该报错但没有报错")
            return False
        except ValueError as e:
            print(f"   ✓ 正确捕获维度不匹配错误: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding_dim_fix()
    
    if success:
        print("\n🎉 embedding_dim修复验证成功!")
        print("现在可以运行完整的初始化脚本")
    else:
        print("\n❌ 仍有问题需要解决")
