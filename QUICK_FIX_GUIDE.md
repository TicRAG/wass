# WASS-RAG 模型训练修复 - 使用指南

## 快速开始

### 1. 验证修复方案 (可选)
```bash
# 运行测试脚本验证训练逻辑
cd d:\Workspace\sjtu\wass
python scripts/test_model_training.py
```

期望输出:
```
✅ SUCCESS: Model shows good prediction diversity
🎉 CONCLUSION: Model training approach is VALIDATED!
```

### 2. 重新训练模型
```bash
# 使用改进的训练脚本重新初始化模型
python scripts/initialize_ai_models.py
```

期望输出:
```
=== Training PerformancePredictor ===
Training samples: 5000+
Epochs: 150
✓ Good prediction diversity achieved
PerformancePredictor R²: 0.85+
```

### 3. 验证修复效果
```bash
# 运行实验框架检查RAG调度器
python experiments/real_experiment_framework.py
```

期望输出:
```
WASSRAGScheduler: Node node_0 -> Score: 156.23  # 不再是 -0.100
WASSRAGScheduler: Node node_1 -> Score: 142.87  # 不同的分数
WASSRAGScheduler: Node node_2 -> Score: 169.45  # 有区分度
选择节点: node_1  # 智能选择，不是随机
```

## 修复前后对比

### 修复前 (问题症状)
```
RAG调度器输出: 所有节点分数 = -0.100
调度决策: 随机选择 (因为分数相同)
根本原因: PerformancePredictor 未训练
```

### 修复后 (期望结果)
```
RAG调度器输出: 不同节点不同分数
调度决策: 选择预期makespan最低的节点
根本原因: 已解决 - 模型经过真实训练
```

## 文件变更说明

### 主要修改文件
1. **scripts/initialize_ai_models.py** - 主要修改
   - 替换 `create_dummy_models()` 为 `create_trained_models()`
   - 实现真实的 PyTorch 训练流程
   - 添加模型验证和诊断

2. **scripts/test_model_training.py** - 新增
   - 独立测试脚本验证训练逻辑
   - 无需依赖复杂的项目模块

### 关键技术改进
- **合成数据生成**: 5000+ 个有意义的训练样本
- **特征工程**: 正确的96维特征向量 (32+32+32)
- **监督学习**: 真实的梯度下降训练
- **模型验证**: 预测多样性检查

## 故障排除

### 如果训练失败
1. 检查PyTorch安装: `pip install torch`
2. 检查模块导入: 确保 `src/ai_schedulers.py` 可访问
3. 查看训练日志: 寻找具体错误信息

### 如果预测多样性低
1. 增加训练样本: 修改 `num_samples` 参数
2. 调整学习率: 修改 `learning_rate` 参数
3. 增加训练轮数: 修改 `epochs` 参数

### 如果RAG调度器仍输出相同分数
1. 验证模型文件: 检查 `models/wass_models.pth` 是否更新
2. 重启实验: 确保加载了新训练的模型
3. 检查知识库: 验证 `data/knowledge_base.pkl` 正常

## 性能监控

运行实验时注意以下指标:

### 训练质量指标 ✅
- R² > 0.7 (模型拟合良好)
- 预测标准差 > 1.0 (输出有差异)
- 训练损失持续下降

### 调度性能指标 ✅
- RAG分数不再全部相同
- 选择的节点makespan确实较低
- 调度决策不是随机的

## 技术支持

如果遇到问题，请检查:
1. `MODEL_TRAINING_IMPROVEMENT_SUMMARY.md` - 详细技术文档
2. `scripts/test_model_training.py` - 测试训练逻辑
3. 实验日志输出 - 查看具体错误信息

---
**最终目标**: RAG调度器恢复智能调度能力，输出有区分度的节点分数。
