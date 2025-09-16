#!/usr/bin/env python3
import json
import pathlib
import wrench
import requests

def main():
    try:
        # 创建模拟
        print("创建WRENCH模拟...")
        simulation = wrench.Simulation()
        
        # 加载平台文件
        print("加载平台文件...")
        platform_file_path = pathlib.Path("test_platform_wrench.xml")
        with open(platform_file_path, "r") as platform_file:
            xml_string = platform_file.read()
        
        # 启动模拟
        print("启动模拟...")
        simulation.start(xml_string, "UserHost")
        print(f"模拟启动成功，simid: {simulation.simid}")
        print(f"守护进程URL: {simulation.daemon_url}")
        
        # 创建工作流文件（转换格式）
        print("转换工作流格式...")
        with open('workflows/montage_5.json', 'r') as f:
            json_doc = json.load(f)
        
        # 转换为wrench格式
        wrench_format = {
            'name': json_doc['metadata']['name'],
            'description': json_doc['metadata']['description'],
            'workflow': {
                'specification': {
                    'tasks': [],
                    'files': []
                }
            }
        }
        
        # 构建task_children映射
        task_children = {}
        for task in json_doc['workflow']['tasks']:
            task_children[task['id']] = []
        for task in json_doc['workflow']['tasks']:
            for parent_id in task.get('dependencies', []):
                if parent_id in task_children:
                    task_children[parent_id].append(task['id'])
        
        # 转换所有任务
        for task in json_doc['workflow']['tasks']:
            wrench_task = {
                'name': task['name'],
                'id': task['id'],
                'children': task_children[task['id']],
                'parents': task.get('dependencies', []),
                'inputFiles': task.get('input_files', []),
                'outputFiles': task.get('output_files', []),
                'flops': task.get('flops', 0),
                'memory': task.get('memory', 0),
                'min_cores': 1,
                'max_cores': 1
            }
            wrench_format['workflow']['specification']['tasks'].append(wrench_task)
        
        # 转换所有文件
        for file in json_doc['workflow']['files']:
            wrench_file = {
                'name': file['name'],
                'size': file.get('size', 0)
            }
            wrench_format['workflow']['specification']['files'].append(wrench_file)
        
        print(f"转换完成，任务数量: {len(wrench_format['workflow']['specification']['tasks'])}")
        
        # 手动调用API来查看响应
        print("手动调用API查看响应...")
        route = f"{simulation.daemon_url}/{simulation.simid}/createWorkflowFromJSON"
        print(f"API端点: {route}")
        
        data = {
            'workflow_json': wrench_format,
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
        
        response = requests.post(route, json=data)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text[:1000]}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"响应JSON键: {list(result.keys())}")
                if 'workflow_name' in result:
                    print(f"找到workflow_name: {result['workflow_name']}")
                else:
                    print("响应中没有workflow_name键")
                    print("完整响应:", json.dumps(result, indent=2))
            except Exception as e:
                print(f"JSON解析失败: {e}")
                print(f"原始响应: {response.text}")
        else:
            print(f"非200响应，完整内容: {response.text}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()