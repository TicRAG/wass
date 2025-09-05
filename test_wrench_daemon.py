#!/usr/bin/env python3
"""
WRENCH daemon问题绕过测试

直接向WRENCH daemon发送请求，排查问题
"""

def test_wrench_daemon_direct():
    """直接与WRENCH daemon通信"""
    print("🔌 直接测试WRENCH daemon...")
    
    try:
        import wrench
        import requests
        import json
        
        # 创建仿真对象以获取daemon URL
        simulation = wrench.Simulation()
        daemon_url = simulation.daemon_url
        print(f"✅ WRENCH daemon URL: {daemon_url}")
        
        # 测试daemon是否响应
        try:
            response = requests.get(f"{daemon_url}/ping")
            print(f"✅ Daemon ping响应: {response.status_code}")
        except Exception as e:
            print(f"❌ Daemon ping失败: {e}")
            return False
        
        # 尝试不同的平台XML传递方式
        platform_variants = {
            "简单字符串": "<platform><host id='h1' speed='1Gf'/></platform>",
            "完整XML": '''<?xml version="1.0"?>
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1Gf"/>
  </zone>
</platform>''',
            "转义XML": '''<?xml version=\\"1.0\\"?>
<platform version=\\"4.1\\">
  <zone id=\\"AS0\\" routing=\\"Full\\">
    <host id=\\"controller_host\\" speed=\\"1Gf\\"/>
  </zone>
</platform>''',
            "单行XML": '''<?xml version="1.0"?><platform version="4.1"><zone id="AS0" routing="Full"><host id="controller_host" speed="1Gf"/></zone></platform>'''
        }
        
        for variant_name, platform_xml in platform_variants.items():
            print(f"\n🧪 测试 {variant_name}...")
            
            spec = {
                "platform_xml": platform_xml,
                "controller_hostname": "controller_host"
            }
            
            try:
                print(f"发送请求，XML长度: {len(platform_xml)} 字符")
                response = requests.post(f"{daemon_url}/startSimulation", json=spec)
                print(f"响应状态: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"✅ {variant_name} 成功!")
                    result = response.json()
                    print(f"响应内容: {result}")
                    
                    # 尝试获取主机信息
                    try:
                        hosts_response = requests.get(f"{daemon_url}/getHostnames")
                        if hosts_response.status_code == 200:
                            hosts = hosts_response.json()
                            print(f"✅ 主机列表: {hosts}")
                    except Exception as e:
                        print(f"获取主机列表失败: {e}")
                    
                    return True
                else:
                    error_info = response.text
                    print(f"❌ {variant_name} 失败: {error_info}")
                    
            except Exception as e:
                print(f"❌ {variant_name} 请求异常: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ daemon测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wrench_daemon_endpoints():
    """探索WRENCH daemon的所有端点"""
    print("\n🔍 探索WRENCH daemon端点...")
    
    try:
        import wrench
        import requests
        
        simulation = wrench.Simulation()
        daemon_url = simulation.daemon_url
        
        # 常见的可能端点
        endpoints = [
            "/ping", "/status", "/health", 
            "/simulation", "/platform", "/hosts",
            "/getHostnames", "/getSimulatedTime",
            "/listServices", "/help", "/api"
        ]
        
        working_endpoints = []
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{daemon_url}{endpoint}", timeout=2)
                if response.status_code == 200:
                    working_endpoints.append(endpoint)
                    print(f"✅ {endpoint}: {response.status_code}")
                    if len(response.text) < 200:
                        print(f"   内容: {response.text}")
                elif response.status_code == 405:  # Method not allowed
                    print(f"📝 {endpoint}: 存在但需要POST")
                elif response.status_code != 404:
                    print(f"⚠️  {endpoint}: {response.status_code}")
            except requests.exceptions.Timeout:
                print(f"⏰ {endpoint}: 超时")
            except Exception as e:
                print(f"❌ {endpoint}: {e}")
        
        print(f"\n✅ 发现 {len(working_endpoints)} 个工作端点: {working_endpoints}")
        return working_endpoints
        
    except Exception as e:
        print(f"❌ 端点探索失败: {e}")
        return []

def try_file_based_platform():
    """尝试基于文件的平台传递"""
    print("\n📁 尝试基于文件的平台传递...")
    
    try:
        import wrench
        import requests
        import os
        
        simulation = wrench.Simulation()
        daemon_url = simulation.daemon_url
        
        # 创建平台文件
        platform_content = '''<?xml version="1.0"?>
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1Gf"/>
  </zone>
</platform>'''
        
        platform_file = "/tmp/file_based_platform.xml"
        with open(platform_file, 'w') as f:
            f.write(platform_content)
        
        # 尝试传递文件路径而不是内容
        spec_variants = [
            {"platform_xml_file": platform_file, "controller_hostname": "controller_host"},
            {"platform_file": platform_file, "controller_hostname": "controller_host"},
            {"platform_path": platform_file, "controller_hostname": "controller_host"},
            {"platform": platform_file, "controller_hostname": "controller_host"}
        ]
        
        for i, spec in enumerate(spec_variants):
            print(f"🧪 测试文件传递方式 {i+1}: {list(spec.keys())}")
            
            try:
                response = requests.post(f"{daemon_url}/startSimulation", json=spec)
                if response.status_code == 200:
                    print(f"✅ 文件方式 {i+1} 成功!")
                    return True
                else:
                    print(f"❌ 状态码: {response.status_code}")
            except Exception as e:
                print(f"❌ 异常: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ 文件传递测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 WRENCH daemon绕过测试")
    print("=" * 60)
    
    # 探索端点
    endpoints = test_wrench_daemon_endpoints()
    
    # 直接daemon通信
    if test_wrench_daemon_direct():
        print("\n🎉 找到工作的daemon通信方式!")
    else:
        print("\n继续尝试文件方式...")
        if try_file_based_platform():
            print("\n🎉 文件方式成功!")
        else:
            print("\n⚠️  需要查看WRENCH daemon日志或源码")
