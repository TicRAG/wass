#!/usr/bin/env python3
"""
测试FAISS修复的简单脚本
"""

def test_faiss_fix():
    """测试FAISS修复"""
    try:
        import numpy as np
        import faiss
        
        print("1. 创建测试向量...")
        # 模拟我们的embedding数据
        embedding_list = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 27  # 32维
        
        # 使用和我们代码相同的处理方式
        embedding_array = np.asarray(embedding_list, dtype=np.float32)
        if len(embedding_array.shape) == 1:
            embedding_vector = embedding_array.reshape(1, -1)
        else:
            embedding_vector = embedding_array
        
        # 关键修复：确保连续内存布局
        embedding_vector = np.ascontiguousarray(embedding_vector, dtype=np.float32)
        
        print(f"   向量形状: {embedding_vector.shape}")
        print(f"   向量类型: {embedding_vector.dtype}")
        print(f"   连续内存: {embedding_vector.flags.c_contiguous}")
        
        print("2. 创建FAISS索引...")
        index = faiss.IndexFlatIP(32)
        
        print("3. 添加向量到索引...")
        index.add(embedding_vector)  # 这里应该不会报错
        print(f"   ✓ 成功添加! 索引现在有 {index.ntotal} 个向量")
        
        print("4. 测试查询...")
        # 测试查询
        query_vector = np.ascontiguousarray(
            embedding_array.reshape(1, -1), 
            dtype=np.float32
        )
        similarities, indices = index.search(query_vector, 1)
        print(f"   ✓ 查询成功! 相似度: {similarities[0][0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== FAISS修复测试 ===")
    
    success = test_faiss_fix()
    
    if success:
        print("\n🎉 FAISS修复测试通过!")
        print("现在可以运行 initialize_ai_models.py")
    else:
        print("\n❌ 还有问题需要解决")
