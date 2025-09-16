#!/usr/bin/env python3

import requests
import json

# 重写requests.post来捕获响应
original_post = requests.post

def debug_post(url, **kwargs):
    print(f'API调用: {url}')
    if 'json' in kwargs:
        print(f'JSON数据中的关键字段...')
        data = kwargs['json']
        if 'platform_xml' in data:
            print(f'  platform_xml长度: {len(data["platform_xml"])}')
        if 'controller_hostname' in data:
            print(f'  controller_hostname: {data["controller_hostname"]}')
    
    response = original_post(url, **kwargs)
    print(f'响应状态码: {response.status_code}')
    try:
        resp_json = response.json()
        print(f'响应JSON: {resp_json}')
        # 保存端口号
        if 'port_number' in resp_json:
            print(f'捕获到端口号: {resp_json["port_number"]}')
            return resp_json['port_number']
    except:
        print(f'响应文本: {response.text[:200]}')
    
    return response

requests.post = debug_post

# 导入并测试wrench
import wrench

# 创建仿真
sim = wrench.simulation.Simulation()
print('Simulation对象创建完成')

# 使用简单的平台XML
platform_xml = '''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="host1" speed="1Gf" core="1"/>
  </zone>
</platform>'''

print('启动仿真...')
try:
    sim.start(platform_xml, 'localhost')
    print('仿真启动成功')
    print(f'Simulation对象属性: {dir(sim)}')
except Exception as e:
    print(f'启动失败: {e}')
    print(f'Simulation对象属性: {dir(sim)}')