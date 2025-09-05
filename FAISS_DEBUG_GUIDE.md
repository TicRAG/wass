# FAISS错误调试指南

## 当前状况
用户在服务器上运行初始化脚本时，仍然遇到FAISS的"input not a numpy array"错误，尽管我们已经添加了`np.ascontiguousarray()`修复。

## 增强的调试版本
我已经在`src/ai_schedulers.py`的`add_case`方法中添加了详细的调试信息：

1. **类型检查**: 检查输入embedding的原始类型
2. **强制转换**: 对列表和数组分别处理
3. **属性验证**: 验证numpy数组的所有关键属性
4. **错误详情**: 提供具体的错误信息

## 运行调试版本

在服务器上运行：
```bash
cd /mnt/home/wass
python scripts/initialize_ai_models.py
```

## 预期调试输出
如果仍有错误，现在会显示：
```
Error adding case to knowledge base: [具体错误]
  Original embedding type: [类型]
  Original embedding shape: [形状]
  Original embedding length: [长度]
```

## 可能的问题和解决方案

### 问题1: embedding_dim不匹配
```python
# 如果看到维度错误，检查RAGKnowledgeBase初始化
kb = RAGKnowledgeBase(embedding_dim=32)  # 确保是32维
```

### 问题2: 数据类型问题  
```python
# 在initialize_ai_models.py中强制转换
embedding = np.array(data["state_embedding"], dtype=np.float32)
# 改为
embedding = np.ascontiguousarray(data["state_embedding"], dtype=np.float32)
```

### 问题3: FAISS版本兼容性
某些FAISS版本可能有特殊要求，可能需要：
```python
# 更激进的转换
embedding_vector = np.copy(embedding_vector)
```

## 下一步
运行调试版本，根据具体的错误信息进行针对性修复。
