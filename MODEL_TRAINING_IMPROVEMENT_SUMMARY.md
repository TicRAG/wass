# WASS-RAG 模型训练改进方案总结

## 问题背景

根据 `real_experiment_framework.log` 分析，WASS-RAG 调度器存在以下核心问题：

1. **输出相同分数**: 所有节点都得到 `-0.100` 的相同分数
2. **无法区分节点**: RAG 调度器失去了智能调度的核心能力
3. **退化为随机选择**: 由于分数相同，只能随机选择节点

## 根本原因分析

通过代码分析发现问题的根本原因：

### 1. PerformancePredictor 未经训练
- 原始 `scripts/initialize_ai_models.py` 只创建了随机初始化的模型
- `create_dummy_models()` 函数生成的是虚拟模型，没有真实训练
- 未训练的神经网络输出接近零或相同的值

### 2. 训练数据不足
- 缺乏有意义的 (特征, 目标) 训练对
- 没有实现监督学习训练循环
- 特征工程不完整（96维输入未正确构造）

## 解决方案实施

### 1. 增强的合成数据生成

**文件**: `scripts/initialize_ai_models.py` - `create_synthetic_training_data()`

**改进内容**:
```python
# 为每个场景生成多个节点样本
for node_idx in range(min(cluster_size, 8)):
    # 生成96维特征向量
    state_embedding = np.array([...])      # 32维工作流状态
    action_embedding = np.array([...])     # 32维节点动作
    context_embedding = np.array([...])    # 32维历史上下文
    
    # 基于启发式规则计算真实makespan
    makespan = base_makespan * load_factor / capacity_factor * dependency_factor
```

**关键特性**:
- 生成 1000 个场景，每个场景 2-8 个节点样本
- 总计 ~5000 个训练样本
- 96维特征向量正确拼接
- 基于合理的启发式规则计算目标值

### 2. 真实的 PyTorch 训练流程

**文件**: `scripts/initialize_ai_models.py` - `train_performance_predictor()`

**实现内容**:
```python
def train_performance_predictor(training_data, epochs=150):
    # 数据预处理和归一化
    X = np.array([data["combined_features"] for data in training_data])
    y = np.array([data["makespan"] for data in training_data])
    y_normalized = (y - y_mean) / y_std
    
    # PyTorch 训练循环
    model = PerformancePredictor(input_dim=96, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        # 批次训练
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
```

**关键改进**:
- 实际的梯度下降训练
- 数据归一化和批次处理
- 学习率调度和早停机制
- 训练过程监控和验证

### 3. 模型验证和诊断

**文件**: `scripts/initialize_ai_models.py` - `validate_trained_model()`

**验证指标**:
```python
# 预测多样性检查
pred_std = np.std(predictions)
unique_predictions = len(np.unique(np.round(predictions, 6)))

# 质量判断
if unique_predictions <= 1:
    print("❌ CRITICAL: Model produces identical predictions!")
elif pred_std < 1.0:
    print("⚠️ WARNING: Low prediction diversity")
else:
    print("✅ GOOD: Model shows healthy prediction diversity")
```

### 4. 测试脚本验证

**文件**: `scripts/test_model_training.py`

创建了独立的测试脚本来验证训练逻辑：
- 生成合成数据
- 执行完整训练流程
- 验证预测多样性
- 输出诊断结果

## 预期效果

### 训练前 (问题状态)
```
WASSRAGScheduler: Node node_0 -> Score: -0.100
WASSRAGScheduler: Node node_1 -> Score: -0.100
WASSRAGScheduler: Node node_2 -> Score: -0.100
选择节点: node_0 (随机选择，因为分数相同)
```

### 训练后 (期望状态)
```
WASSRAGScheduler: Node node_0 -> Score: 156.23
WASSRAGScheduler: Node node_1 -> Score: 142.87
WASSRAGScheduler: Node node_2 -> Score: 169.45
选择节点: node_1 (智能选择最低预期makespan)
```

## 技术要点

### 1. 特征工程
- **状态特征**: 工作流特性（任务数、依赖关系、计算强度）
- **动作特征**: 节点特性（负载、容量、性能）
- **上下文特征**: 历史信息（相似案例、置信度）

### 2. 训练稳定性
- 数据归一化防止梯度爆炸
- Dropout 层防止过拟合
- Weight decay 正则化
- 早停机制避免过训练

### 3. 预测多样性
- 确保不同输入产生不同输出
- 验证预测值的标准差 > 1.0
- 检查唯一预测值数量
- 监控训练过程中的梯度更新

## 部署建议

1. **立即部署**: 更新后的 `scripts/initialize_ai_models.py`
2. **验证运行**: 执行 `python scripts/test_model_training.py`
3. **重新初始化**: 运行 `python scripts/initialize_ai_models.py`
4. **实验验证**: 运行 `python experiments/real_experiment_framework.py`

## 监控指标

部署后应监控以下指标确保修复成功：

### 训练质量指标
- R² > 0.7 (模型拟合度)
- 预测标准差 > 1.0 (预测多样性)
- 唯一预测值占比 > 80%

### 调度器性能指标
- RAG 调度器输出不同的节点分数
- 选择的节点具有最低预期 makespan
- 调度决策不再是随机的

## 长期改进

1. **真实数据训练**: 使用实际工作流执行数据
2. **在线学习**: 实现增量学习更新模型
3. **多目标优化**: 除 makespan 外考虑能耗等指标
4. **模型集成**: 组合多个预测器提高鲁棒性

---

**结论**: 通过实施真实的监督学习训练，WASS-RAG 调度器应该能够输出有意义的、区分性的节点分数，从而恢复智能调度能力。
