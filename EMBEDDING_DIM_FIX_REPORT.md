# RAGKnowledgeBase embedding_dim 修复报告

## 问题诊断
用户报告错误：
```
AttributeError: 'RAGKnowledgeBase' object has no attribute 'embedding_dim'
```

## 根本原因
在`RAGKnowledgeBase`类的`__init__`方法中，`embedding_dim`只是一个局部变量，没有保存为实例属性：

**问题代码:**
```python
def __init__(self, knowledge_base_path: Optional[str] = None):
    # ...
    
def _initialize_empty_kb(self):
    embedding_dim = 32  # 局部变量，不是实例属性
    self.index = faiss.IndexFlatIP(embedding_dim)
```

## 修复内容

### 1. 修复RAGKnowledgeBase初始化
**文件:** `src/ai_schedulers.py`

**修复前:**
```python
def __init__(self, knowledge_base_path: Optional[str] = None):
    self.knowledge_base_path = knowledge_base_path
    self.index = None
    self.cases = []
```

**修复后:**
```python
def __init__(self, knowledge_base_path: Optional[str] = None, embedding_dim: int = 32):
    self.knowledge_base_path = knowledge_base_path
    self.embedding_dim = embedding_dim  # 保存为实例属性
    self.index = None
    self.cases = []
```

### 2. 修复_initialize_empty_kb方法
**修复前:**
```python
def _initialize_empty_kb(self):
    embedding_dim = 32
    self.index = faiss.IndexFlatIP(embedding_dim)
```

**修复后:**
```python
def _initialize_empty_kb(self):
    self.index = faiss.IndexFlatIP(self.embedding_dim)
```

### 3. 修复initialize_ai_models.py中的调用
**文件:** `scripts/initialize_ai_models.py`

**修复前:**
```python
kb = RAGKnowledgeBase()
```

**修复后:**
```python
kb = RAGKnowledgeBase(embedding_dim=32)
```

## 技术改进

1. **类型安全**: 添加了embedding_dim参数的类型注解
2. **默认值**: 保持32维作为默认值，向后兼容
3. **灵活性**: 现在可以创建不同维度的知识库
4. **一致性**: FAISS索引维度与类属性保持一致

## 验证步骤

运行修复后的初始化脚本：
```bash
cd /mnt/home/wass
python scripts/initialize_ai_models.py
```

预期输出：
```
=== WASS-RAG AI Model and Knowledge Base Initialization ===

1. Generating synthetic training data...
   Saved training data to: data/synthetic_training_data.json

2. Creating pre-trained models...
   Saved models to: models/wass_models.pth

3. Creating knowledge base...
Initialized empty knowledge base
Adding 2000 cases to knowledge base...
✓ 成功添加案例到知识库 (不再有AttributeError)
```

## 完整修复状态
- ✅ RAGKnowledgeBase初始化修复
- ✅ embedding_dim属性添加
- ✅ FAISS索引创建修复
- ✅ 调用代码更新
- ✅ 向后兼容性保持
