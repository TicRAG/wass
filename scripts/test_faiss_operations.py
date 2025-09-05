#!/usr/bin/env python3
"""
测试FAISS操作的简单脚本
"""

def test_faiss_operations():
    """测试FAISS向量操作"""
    try:
        import numpy as np
        import faiss
        
        print("✓ 成功导入numpy和faiss")
        
        # 创建测试数据
        embedding_list = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 27  # 32维向量
        
        # 测试不同的数组创建方式
        test_cases = [
            ("list转numpy", np.array(embedding_list, dtype=np.float32)),
            ("直接asarray", np.asarray(embedding_list, dtype=np.float32)),
            ("显式reshape", np.array(embedding_list, dtype=np.float32).reshape(1, -1)),
        ]
        
        # 创建FAISS索引
        index = faiss.IndexFlatIP(32)
        print("✓ 成功创建FAISS索引")
        
        for name, embedding in test_cases:
            try:
                # 确保是2D数组
                if len(embedding.shape) == 1:
                    embedding_2d = embedding.reshape(1, -1)
                else:
                    embedding_2d = embedding
                
                print(f"  测试 {name}: shape={embedding_2d.shape}, dtype={embedding_2d.dtype}")
                
                # 添加到索引
                index.add(embedding_2d)
                print(f"  ✓ {name} 成功添加到FAISS索引")
                
            except Exception as e:
                print(f"  ❌ {name} 失败: {e}")
        
        print(f"✓ FAISS索引现在有 {index.ntotal} 个向量")
        
        # 测试搜索
        query = np.array(embedding_list, dtype=np.float32).reshape(1, -1)
        similarities, indices = index.search(query, 2)
        print(f"✓ 搜索成功: 相似度={similarities[0]}, 索引={indices[0]}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ FAISS操作失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== FAISS操作测试 ===")
    
    success = test_faiss_operations()
    
    if success:
        print("\n🎉 FAISS操作测试通过!")
        print("现在可以安全运行 initialize_ai_models.py")
    else:
        print("\n❌ FAISS操作测试失败")
