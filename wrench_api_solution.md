# Wrench API 调用问题解决方案

## 问题描述
在使用wrench库时，发现通过`simulation.create_workflow_from_json()`方法调用API成功，但手动调用相同的API却失败，返回错误：
```
failure_cause: "[json.exception.type_error.302] type must be string, but is null"
wrench_api_request_success: false
```

## 根本原因
通过深入分析wrench库的源码和实际请求，发现了关键差异：

### 1. 字段名差异
- **wrench库使用**: `json_string` 字段
- **手动调用使用**: `workflow_json` 字段

### 2. 数据格式差异
- **wrench库发送**: JSON字符串（字符串格式）
- **手动调用发送**: JSON对象（字典格式）

## 正确调用方式
```python
import json
import requests
import wrench

# 创建模拟
simulation = wrench.Simulation()
simulation.start(platform_xml, 'UserHost')

# 加载工作流数据
with open('workflow.json', 'r') as f:
    workflow_data = json.load(f)

# 正确的API调用方式
url = f"{simulation.daemon_url}/{simulation.simid}/createWorkflowFromJSON"
data = {
    'json_string': json.dumps(workflow_data),  # 关键：使用json_string且为字符串格式
    'reference_flop_rate': '100Mf',
    'ignore_machine_specs': True,
    'redundant_dependencies': False,
    'ignore_cycle_creating_dependencies': False,
    'min_cores_per_task': 1,
    'max_cores_per_task': 1,
    'enforce_num_cores': True,
    'ignore_avg_cpu': True,
    'show_warnings': True
}

response = requests.post(url, json=data)
```

## 验证结果
使用正确的字段名和格式后：
- 响应内容长度：204397字节（之前119字节）
- 响应包含完整的workflow信息
- 包含569个任务和1025个文件的详细信息
- 成功创建工作流

## 技术细节
1. **wrench库内部处理**: 在`create_workflow_from_json`方法中，将工作流数据转换为JSON字符串
2. **API端点期望**: 接收`json_string`字段，内容为序列化的JSON字符串
3. **错误处理**: 当接收到错误的字段名或格式时，API返回类型错误

## 最佳实践
1. 始终使用`json.dumps()`将JSON对象转换为字符串
2. 使用`json_string`作为字段名，而不是`workflow_json`
3. 可以参考wrench库的源码实现：`wrench/simulation.py`中的`create_workflow_from_json`方法