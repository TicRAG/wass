# 背景
阅读doc/wass_paper.md,这是正在进行编写的论文，其中的实验部分是Fake的。
当前是开发环境，无法运行python、wrench等，不过在另一个环境已经部署好了wrench python api。

# 目标
1. 完成实验所需的所有代码，记录开发过程与开发目标到markdown。
2. 如果用到wrench，请阅读https://wrench-python-api.readthedocs.io/en/latest/

# 当前进展 (2025-09-05)
- 已添加目录: `src/`, `experiments/`, `notes/`
- 建立 `notes/dev_log.md` 记录开发过程
- 定义初始接口与架构草案: `src/interfaces.py`, `src/architecture.md`
- 创建最小流水线脚本: `experiments/run_pipeline.py`
- 添加示例整体配置: `configs_example.yaml`

# 开发待办 (滚动维护)
- [ ] 数据适配器实现 (simple_jsonl)
- [ ] 关键词型 Label Function 模板 + 注册机制
- [ ] Label Matrix 构建器 (对接 Wrench API 占位包装)
- [ ] 多种 Label Model (MajorityVote, Generative)
- [ ] 图构建器 (共现/相似度) + GNN 占位 (GCN/GAT)
- [ ] RAG 模块 (BM25 + 向量检索占位) + 融合策略
- [ ] DRL 环境与策略 (Active Learning / Sampling)
- [ ] 训练 & 评估脚本 (指标: accuracy, f1, coverage, abstain_rate)
- [ ] 结果输出与复现 (results/<exp_name>)
- [ ] 报告生成脚本（汇总指标 + 表格markdown）

# 运行占位流水线
在可运行 Python 的环境中:
```
python experiments/run_pipeline.py
```
输出为各阶段的占位执行打印。

# 配置扩展示例
参考 `configs_example.yaml` 拆分为：
```
configs/
  data.yaml
  labeling.yaml
  label_model.yaml
  graph.yaml
  rag.yaml
  drl.yaml
  experiment.yaml
```

# 与 Wrench 集成设计要点
- 提供一个 `WrenchLabelModelWrapper` 封装其训练与预测接口，内部惰性导入 (避免无环境时报错)
- Label Function -> Wrench 期望的格式：构造 label matrix (稀疏) 并传入
- 需要记录：LF 覆盖率、冲突率、精度估计

# 后续
详见 `notes/dev_log.md`，逐步实现功能并更新论文实验部分。
