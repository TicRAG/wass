#!/usr/bin/env python3
"""
WRENCH XML格式修复测试

基于daemon错误信息找出正确的XML格式
"""

def test_xml_encoding_formats():
    """测试不同的XML编码格式"""
    print("🧪 测试XML编码格式...")
    
    try:
        import wrench
        import base64
        import json
        
        # 测试不同的编码方式
        platform_xml = '''<?xml version="1.0"?>
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1Gf"/>
  </zone>
</platform>'''
        
        encoding_tests = {
            "原始XML": platform_xml,
            "Base64编码": base64.b64encode(platform_xml.encode()).decode(),
            "URL编码": platform_xml.replace('<', '%3C').replace('>', '%3E'),
            "转义XML": platform_xml.replace('<', '&lt;').replace('>', '&gt;'),
            "JSON转义": json.dumps(platform_xml),
            "双重转义": json.dumps(platform_xml)[1:-1],  # 去掉外层引号
        }
        
        for encoding_name, encoded_xml in encoding_tests.items():
            print(f"\n🔄 测试 {encoding_name}...")
            print(f"   格式预览: {str(encoded_xml)[:50]}...")
            
            try:
                simulation = wrench.Simulation()
                simulation.start(encoded_xml, "controller_host")
                print(f"✅ {encoding_name} 成功!")
                
                # 获取主机信息验证
                hostnames = simulation.get_all_hostnames()
                print(f"✅ 主机列表: {hostnames}")
                
                simulation.shutdown()
                return encoding_name, encoded_xml
                
            except Exception as e:
                error_msg = str(e)
                if "Unexpected character" in error_msg:
                    # 提取出错的字符
                    char_info = error_msg.split("Unexpected character")[1].split("in prolog")[0].strip()
                    print(f"❌ {encoding_name} 失败: 意外字符 {char_info}")
                else:
                    print(f"❌ {encoding_name} 失败: {error_msg[:60]}...")
        
        return None, None
        
    except Exception as e:
        print(f"❌ 编码测试失败: {e}")
        return None, None

def test_file_content_inspection():
    """检查WRENCH创建的临时文件内容"""
    print("\n🔍 检查临时文件内容...")
    
    try:
        import wrench
        import os
        import glob
        import time
        import threading
        
        # 监控临时文件的线程
        temp_files_found = []
        
        def monitor_temp_files():
            while True:
                files = glob.glob("/tmp/wrench_daemon_platform_file_*.xml")
                for f in files:
                    if f not in temp_files_found:
                        temp_files_found.append(f)
                        try:
                            with open(f, 'rb') as file:
                                content = file.read()
                            print(f"\n📄 发现临时文件: {f}")
                            print(f"   文件大小: {len(content)} 字节")
                            print(f"   前20字节: {content[:20]}")
                            print(f"   前20字节(十六进制): {content[:20].hex()}")
                            if len(content) < 100:
                                print(f"   完整内容: {content}")
                        except Exception as e:
                            print(f"   读取失败: {e}")
                time.sleep(0.1)
        
        # 启动监控
        monitor_thread = threading.Thread(target=monitor_temp_files, daemon=True)
        monitor_thread.start()
        
        # 触发WRENCH创建临时文件
        print("🚀 触发WRENCH创建临时文件...")
        simulation = wrench.Simulation()
        
        try:
            simulation.start("<platform><host id='h' speed='1Gf'/></platform>", "h")
        except:
            pass  # 我们只是想查看临时文件
        
        # 等待文件监控
        time.sleep(2)
        
        if temp_files_found:
            print(f"\n✅ 检查了 {len(temp_files_found)} 个临时文件")
            return True
        else:
            print("❌ 没有发现临时文件")
            return False
            
    except Exception as e:
        print(f"❌ 文件检查失败: {e}")
        return False

def test_different_xml_structures():
    """测试不同的XML结构"""
    print("\n🏗️ 测试不同XML结构...")
    
    xml_structures = {
        "无声明": "<platform><host id='controller_host' speed='1Gf'/></platform>",
        "简化声明": "<?xml version='1.0'?><platform><host id='controller_host' speed='1Gf'/></platform>",
        "无DOCTYPE": "<?xml version='1.0'?><platform version='4.1'><host id='controller_host' speed='1Gf'/></platform>",
        "单行完整": "<?xml version='1.0'?><!DOCTYPE platform SYSTEM 'https://simgrid.org/simgrid.dtd'><platform version='4.1'><zone id='AS0' routing='Full'><host id='controller_host' speed='1Gf'/></zone></platform>",
        "紧凑格式": "<platform version='4.1'><host id='controller_host' speed='1Gf'/></platform>",
    }
    
    import wrench
    
    for structure_name, xml_content in xml_structures.items():
        print(f"\n🔄 测试 {structure_name}...")
        print(f"   内容: {xml_content[:60]}...")
        
        try:
            simulation = wrench.Simulation()
            simulation.start(xml_content, "controller_host")
            print(f"✅ {structure_name} 成功!")
            
            hostnames = simulation.get_all_hostnames()
            print(f"✅ 主机: {hostnames}")
            
            simulation.shutdown()
            return structure_name, xml_content
            
        except Exception as e:
            print(f"❌ {structure_name} 失败: {str(e)[:80]}...")
    
    return None, None

if __name__ == "__main__":
    print("🚀 WRENCH XML格式修复测试")
    print("=" * 60)
    
    # 检查临时文件内容
    test_file_content_inspection()
    
    # 测试编码格式
    print("\n" + "=" * 60)
    success_encoding = test_xml_encoding_formats()
    
    if success_encoding[0]:
        print(f"\n🎉 找到工作的编码: {success_encoding[0]}")
    else:
        # 测试XML结构
        print("\n" + "=" * 60)
        success_structure = test_different_xml_structures()
        
        if success_structure[0]:
            print(f"\n🎉 找到工作的结构: {success_structure[0]}")
        else:
            print("\n⚠️  需要进一步调试WRENCH daemon")
