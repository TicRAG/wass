#!/usr/bin/env python3
"""
独立测试FAISS操作，模拟initialize_ai_models.py的具体情况
"""

import sys
import os
import numpy as np
import json

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_faiss_with_real_data():
    """使用真实数据测试FAISS操作"""
    
    try:
        from src.ai_schedulers import RAGKnowledgeBase
        
        print("=== FAISS实际数据测试 ===")
        
        # 1. 模拟从JSON读取的数据（这是问题的根源）
        print("1. 模拟从JSON读取embedding数据...")
        
        # 创建模拟的state_embedding（就像initialize_ai_models.py中的数据）
        original_numpy = np.array([
            0.05,  # task_count / 100.0
            0.3,   # avg_flops / 10e9
            0.25,  # avg_memory / 4e9
            0.4,   # dependency_ratio
            0.6,   # data_intensity
            0.5,   # node_count / 16.0
            0.7,   # avg_load
        ] + [np.random.randn() for _ in range(25)])  # 填充到32维
        
        # 模拟JSON序列化/反序列化过程
        json_data = original_numpy.tolist()  # 这是保存到JSON的过程
        print(f"   JSON数据类型: {type(json_data)}")
        print(f"   JSON数据长度: {len(json_data)}")
        
        # 模拟从JSON读取回来的过程
        loaded_embedding = np.array(json_data, dtype=np.float32)  # 这是从JSON读取的过程
        print(f"   加载后类型: {type(loaded_embedding)}")
        print(f"   加载后形状: {loaded_embedding.shape}")
        print(f"   加载后dtype: {loaded_embedding.dtype}")
        
        # 2. 创建知识库并测试添加
        print("\n2. 创建知识库...")
        kb = RAGKnowledgeBase(embedding_dim=32)
        
        print("3. 测试添加案例...")
        
        # 这就是initialize_ai_models.py中调用的方式
        kb.add_case(
            embedding=loaded_embedding,  # 从JSON加载的数据
            workflow_info={"task_count": 5, "complexity": "medium", "type": "synthetic"},
            actions=["node_0", "node_1", "node_2"],
            makespan=10.5
        )
        
        print("   ✓ 成功添加第一个案例!")
        
        # 添加更多案例
        for i in range(5):
            test_data = [0.1 * (i+1)] * 7 + [np.random.randn() for _ in range(25)]
            json_data = test_data  # 模拟JSON数据
            embedding = np.array(json_data, dtype=np.float32)
            
            kb.add_case(
                embedding=embedding,
                workflow_info={"task_count": i+1, "complexity": "test", "type": "synthetic"},
                actions=[f"node_{j}" for j in range(i+1)],
                makespan=5.0 + i
            )
        
        print(f"   ✓ 成功添加了 {len(kb.cases)} 个案例!")
        
        # 4. 测试查询
        print("\n4. 测试查询...")
        query_embedding = np.array([0.05] * 7 + [0.1] * 25, dtype=np.float32)
        results = kb.retrieve_similar_cases(query_embedding, top_k=3)
        
        print(f"   ✓ 查询成功! 找到 {len(results['similar_cases'])} 个相似案例")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_faiss_with_real_data()
    
    if success:
        print("\n🎉 FAISS实际数据测试成功!")
        print("问题已解决，可以运行完整的初始化脚本")
    else:
        print("\n❌ 仍有问题需要解决")
