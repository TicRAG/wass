#!/usr/bin/env python3
"""
深度调试WRENCH内部平台文件处理

检查WRENCH如何处理我们的平台文件
"""

import os
import time
import threading
import glob

def monitor_temp_files():
    """监控/tmp目录中WRENCH创建的文件"""
    print("🔍 开始监控临时文件...")
    while True:
        # 查找WRENCH临时文件
        wrench_files = glob.glob("/tmp/wrench_daemon_platform_file_*.xml")
        if wrench_files:
            for file_path in wrench_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    print(f"📄 发现WRENCH临时文件: {file_path}")
                    print(f"文件大小: {len(content)} 字节")
                    print(f"前100字符: {repr(content[:100])}")
                    if len(content) < 200:
                        print(f"完整内容: {repr(content)}")
                    print("-" * 40)
                except Exception as e:
                    print(f"❌ 读取临时文件失败: {e}")
        time.sleep(0.1)

def test_wrench_with_monitoring():
    """在监控下测试WRENCH"""
    print("🧪 在文件监控下测试WRENCH...")
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_temp_files, daemon=True)
    monitor_thread.start()
    
    try:
        import wrench
        print(f"✅ WRENCH {wrench.__version__} 导入成功")
        
        # 创建最简单的平台文件
        platform_content = '''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1Gf"/>
  </zone>
</platform>'''
        
        platform_file = "/tmp/debug_platform.xml"
        with open(platform_file, 'w', encoding='utf-8') as f:
            f.write(platform_content)
        
        print(f"✅ 平台文件创建: {platform_file}")
        
        # 验证我们的文件
        with open(platform_file, 'rb') as f:
            raw_content = f.read()
        print(f"📄 原始文件字节: {raw_content[:50]}")
        
        # 尝试启动WRENCH
        simulation = wrench.Simulation()
        print("🚀 启动WRENCH仿真...")
        
        try:
            simulation.start(platform_file, "controller_host")
            print("✅ 启动成功！")
            simulation.shutdown()
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            print("等待一下查看临时文件...")
            time.sleep(1)
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_simgrid_directly():
    """尝试直接使用SimGrid验证XML"""
    print("\n🔬 尝试直接验证SimGrid XML...")
    
    try:
        # 检查是否可以导入simgrid
        try:
            import simgrid
            print("✅ SimGrid可用")
            
            # 尝试加载平台
            platform_file = "/tmp/debug_platform.xml"
            if os.path.exists(platform_file):
                print(f"🔍 直接用SimGrid验证: {platform_file}")
                # 这里可能需要根据SimGrid API调整
                
        except ImportError:
            print("⚠️  SimGrid不可直接导入，通过WRENCH间接使用")
            
    except Exception as e:
        print(f"❌ SimGrid测试失败: {e}")

def inspect_wrench_source():
    """检查WRENCH源码位置和配置"""
    print("\n🔍 检查WRENCH安装信息...")
    
    try:
        import wrench
        print(f"WRENCH版本: {wrench.__version__}")
        print(f"WRENCH路径: {wrench.__file__}")
        
        # 检查simulation.py源码
        import inspect
        sim_source = inspect.getsource(wrench.Simulation.start)
        print("🔬 Simulation.start方法源码片段:")
        lines = sim_source.split('\n')[:20]  # 前20行
        for i, line in enumerate(lines):
            print(f"  {i+1:2d}: {line}")
            
    except Exception as e:
        print(f"❌ 源码检查失败: {e}")

def try_alternative_xml_formats():
    """尝试不同的XML格式"""
    print("\n🧪 尝试替代XML格式...")
    
    # 格式1: 不同的DTD
    xml_formats = {
        "local_dtd": '''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1Gf"/>
  </zone>
</platform>''',
        
        "no_dtd": '''<?xml version="1.0"?>
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1Gf"/>
  </zone>
</platform>''',
        
        "minimal": '''<?xml version="1.0"?>
<platform version="4.1">
  <host id="controller_host" speed="1Gf"/>
</platform>'''
    }
    
    import wrench
    
    for format_name, xml_content in xml_formats.items():
        print(f"🔄 测试格式: {format_name}")
        
        test_file = f"/tmp/test_{format_name}.xml"
        with open(test_file, 'w') as f:
            f.write(xml_content)
        
        try:
            simulation = wrench.Simulation()
            simulation.start(test_file, "controller_host")
            print(f"✅ {format_name} 格式成功！")
            simulation.shutdown()
            return True
        except Exception as e:
            print(f"❌ {format_name} 格式失败: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 WRENCH深度调试")
    print("=" * 60)
    
    # 先检查WRENCH信息
    inspect_wrench_source()
    
    # 测试替代格式
    if try_alternative_xml_formats():
        print("\n🎉 找到工作的XML格式!")
    else:
        print("\n继续深度调试...")
        # 监控调试
        test_wrench_with_monitoring()
        
        # SimGrid直接测试
        test_simgrid_directly()
