#!/usr/bin/env python3
"""
测试FAISS aggressive fix
"""

import sys
import os
import numpy as np
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_single_case():
    """测试单个案例添加"""
    try:
        from src.ai_schedulers import RAGKnowledgeBase
        
        print("=== FAISS Aggressive Fix Test ===")
        
        # 创建知识库
        kb = RAGKnowledgeBase(embedding_dim=32)
        print(f"Knowledge base created with embedding_dim={kb.embedding_dim}")
        
        # 模拟从JSON读取的数据（这是原始问题来源）
        original_data = [0.05, 0.3, 0.25, 0.4, 0.6, 0.5, 0.7] + [np.random.randn() for _ in range(25)]
        json_serialized = json.dumps(original_data)  # 模拟JSON序列化
        json_loaded = json.loads(json_serialized)    # 模拟JSON读取
        
        # 转换为numpy数组（就像initialize_ai_models.py中的做法）
        embedding = np.array(json_loaded, dtype=np.float32)
        
        print(f"Test embedding: type={type(embedding)}, shape={embedding.shape}, dtype={embedding.dtype}")
        print(f"Contiguous: {embedding.flags.c_contiguous}")
        
        # 添加案例
        kb.add_case(
            embedding=embedding,
            workflow_info={"task_count": 5, "type": "test"},
            actions=["node_0", "node_1"],
            makespan=10.0
        )
        
        print(f"✓ Successfully added case! KB now has {len(kb.cases)} cases")
        print(f"✓ FAISS index now has {kb.index.ntotal} vectors")
        
        # 测试查询
        query = np.array([0.1] * 32, dtype=np.float32)
        results = kb.retrieve_similar_cases(query, top_k=1)
        print(f"✓ Query successful! Found {len(results['similar_cases'])} similar cases")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_case()
    
    if success:
        print("\n🎉 Aggressive fix test passed!")
        print("The FAISS issue should now be resolved.")
    else:
        print("\n❌ Still encountering issues")
