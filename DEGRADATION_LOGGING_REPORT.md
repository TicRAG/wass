# 降级日志增强实现报告

## 增强目标

在WASS-RAG实验中，如果任何AI组件发生降级（fallback），都要有明显的日志标记，帮助识别实验数据的真实质量。

## 降级场景识别

### 1. WASSSmartScheduler降级场景
- **DRL决策失败** → 降级到随机选择
- **GNN编码器不可用** → 降级到简单特征提取  
- **图数据构建失败** → 降级到简单特征提取

### 2. WASSRAGScheduler降级场景
- **RAG决策失败** → 降级到基础DRL方法
- **知识库检索失败** → 降级到基础DRL方法
- **性能预测失败** → 降级到基础DRL方法

## 日志增强实现

### 统一降级标记格式
```
⚠️  [DEGRADATION] <具体原因>
⚠️  [DEGRADATION] <调度器名称> falling back to <降级方法>
⚠️  [DEGRADATION] Task: <任务ID>, <额外上下文>
⚠️  [DEGRADATION] <降级结果描述>
```

### 推理信息标记
降级决策的reasoning字段都以`🔴 DEGRADED:`开头，便于后续分析：
```python
reasoning=f"🔴 DEGRADED: <降级类型> fallback due to <具体错误>"
```

## 具体修改

### 1. DRL决策失败降级
**位置**: WASSSmartScheduler.make_decision()
```python
except Exception as e:
    print(f"⚠️  [DEGRADATION] DRL decision making failed: {e}")
    print(f"⚠️  [DEGRADATION] WASSSmartScheduler falling back to RANDOM selection")
    print(f"⚠️  [DEGRADATION] Task: {state.current_task}, Available nodes: {state.available_nodes}")
    
    fallback_node = np.random.choice(state.available_nodes)
    print(f"⚠️  [DEGRADATION] Random fallback selected: {fallback_node}")
    
    return SchedulingAction(
        reasoning=f"🔴 DEGRADED: Random fallback due to DRL error: {e}"
    )
```

### 2. GNN编码器不可用降级
**位置**: WASSSmartScheduler._encode_state_graph()
```python
if self.gnn_encoder is None:
    print(f"⚠️  [DEGRADATION] GNN encoder not available, using simple features for {state.current_task}")
    return self._extract_simple_features(state)
```

### 3. 图数据构建失败降级
**位置**: WASSSmartScheduler._build_graph_data()
```python
if not HAS_TORCH_GEOMETRIC:
    print(f"⚠️  [DEGRADATION] torch_geometric not available, graph data will be None for {state.current_task}")
    return None

# 异常处理
except Exception as e:
    print(f"⚠️  [DEGRADATION] Graph data construction failed: {e}")
    print(f"⚠️  [DEGRADATION] Falling back to None (will use simple features)")
    return None
```

### 4. RAG决策失败降级
**位置**: WASSRAGScheduler.make_decision()
```python
except Exception as e:
    print(f"⚠️  [DEGRADATION] RAG decision making failed: {e}")
    print(f"⚠️  [DEGRADATION] WASSRAGScheduler falling back to base DRL method")
    print(f"⚠️  [DEGRADATION] Task: {state.current_task}, Attempting base scheduler...")
    
    fallback_action = self.base_scheduler.make_decision(state)
    print(f"⚠️  [DEGRADATION] Base DRL fallback result: {fallback_action.target_node}")
    
    fallback_action.reasoning = f"🔴 DEGRADED: RAG->DRL fallback due to error: {e}"
    return fallback_action
```

## 使用方法

### 实验执行
```bash
cd /mnt/home/wass
python experiments/real_experiment_framework.py
```

### 日志分析
实验后可以通过以下方式分析降级情况：

**1. 统计降级次数**
```bash
grep "⚠️.*DEGRADATION" experiments/real_experiment_framework.log | wc -l
```

**2. 查看降级原因**
```bash
grep "⚠️.*DEGRADATION.*failed" experiments/real_experiment_framework.log
```

**3. 查找降级决策**
```bash
grep "🔴 DEGRADED" experiments/real_experiment_framework.log
```

### 实验结果验证
- **0% 降级率**: 所有AI组件正常工作，实验数据完全可靠
- **<10% 降级率**: 轻微降级，数据基本可靠，需要说明
- **>10% 降级率**: 需要检查AI组件配置，可能影响结论

## 预期输出示例

### 正常情况
```
Running: WASS-DRL (w/o RAG), 10 tasks, 4 nodes, rep 0
  WASS-DRL (w/o RAG): task_0 -> node_2 (confidence: 0.85, time: 2.1ms)
```

### 降级情况
```
Running: WASS-DRL (w/o RAG), 10 tasks, 4 nodes, rep 0
⚠️  [DEGRADATION] DRL decision making failed: CUDA out of memory
⚠️  [DEGRADATION] WASSSmartScheduler falling back to RANDOM selection
⚠️  [DEGRADATION] Task: task_0, Available nodes: ['node_0', 'node_1', 'node_2', 'node_3']
⚠️  [DEGRADATION] Random fallback selected: node_1
  WASS-DRL (w/o RAG): task_0 -> node_1 (confidence: 0.10, time: 0.1ms)
```

## 实验数据质量保证

通过这套降级日志系统，你可以：
1. **实时监控**: 实验过程中立即发现降级
2. **质量评估**: 根据降级率评估数据可靠性
3. **问题诊断**: 快速定位AI组件问题
4. **结果解释**: 在论文中准确说明实验条件
