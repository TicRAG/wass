#!/usr/bin/env python3

import requests
import json

# 重写requests.post来捕获createWorkflowFromJSON请求
original_post = requests.post

def debug_post(url, data=None, **kwargs):
    """修复后的debug_post函数，接受data参数"""
    print(f'DEBUG POST: {url}')
    if data:
        print(f'DATA: {data}')
    # 调用原始的requests.post
    response = original_post(url, data=data, **kwargs)
    print(f'响应状态码: {response.status_code}')
    try:
        resp_json = response.json()
        print(f'响应: {resp_json}')
    except:
        print(f'响应文本: {response.text[:200]}')
    return response

requests.post = debug_post

# 导入并测试wrench
import wrench
from pathlib import Path

# 创建仿真
sim = wrench.simulation.Simulation()
platform_xml = '''<?xml version=\"1.0\"?>\n<!DOCTYPE platform SYSTEM \"https://simgrid.org/simgrid.dtd\">\n<platform version=\"4.1\">\n  <zone id=\"AS0\" routing=\"Full\">\n    <host id=\"host1\" speed=\"1Gf\" core=\"1\"/>\n    <host id=\"localhost\" speed=\"1Gf\" core=\"1\"/>\n  </zone>\n</platform>'''

print('启动仿真...')
sim.start(platform_xml, 'localhost')
print(f'仿真端口: {sim.daemon_port}')
print(f'仿真ID: {sim.simid}')

# 测试工作流数据 - 使用示例WfCommons文件
workflow_file = 'wrenchtest/examples/json_workflow_simulator/sample_wfcommons_workflow.json'
with open(workflow_file, 'r') as f:
    workflow_data = json.load(f)

print(f'\\n原始工作流数据键: {list(workflow_data.keys())}')
print(f'workflow键: {list(workflow_data["workflow"].keys())}')
print(f'specification任务数: {len(workflow_data["workflow"]["specification"]["tasks"])}')
print(f'execution任务数: {len(workflow_data["workflow"]["execution"]["tasks"])}')
print(f'execution机器数: {len(workflow_data["workflow"]["execution"]["machines"])}')

# 尝试创建工作流
print('\n尝试创建工作流...')
try:
    workflow = sim.create_workflow_from_json(workflow_data,
                                            reference_flop_rate="100Mf",
                                            ignore_machine_specs=True,
                                            redundant_dependencies=False,
                                            ignore_cycle_creating_dependencies=False,
                                            min_cores_per_task=1,
                                            max_cores_per_task=1,
                                            enforce_num_cores=True,
                                            ignore_avg_cpu=True,
                                            show_warnings=True)
    print('工作流创建成功！')
    print(f'工作流任务数: {len(workflow.get_tasks())}')
except Exception as e:
    print(f'工作流创建失败: {e}')
finally:
    # 清理
    sim.terminate()