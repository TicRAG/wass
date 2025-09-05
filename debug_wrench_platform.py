#!/usr/bin/env python3
"""
WRENCH 0.3-dev 调试平台文件问题

让我们尝试不同的方法来解决平台文件问题
"""

def test_wrench_debug_platform():
    """调试WRENCH平台文件处理"""
    print("🔍 调试WRENCH平台文件处理...")
    
    try:
        import wrench
        print(f"✅ WRENCH {wrench.__version__} 导入成功")
        
        # 创建仿真对象
        simulation = wrench.Simulation()
        print("✅ Simulation对象创建成功")
        
        # 尝试1: 最简单的平台文件
        simple_platform = '''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1Gf"/>
    <host id="compute_host" speed="1Gf"/>
  </zone>
</platform>'''
        
        platform_file = "/tmp/simple_platform.xml"
        with open(platform_file, 'w') as f:
            f.write(simple_platform)
        print(f"✅ 简单平台文件创建: {platform_file}")
        
        # 验证文件内容
        with open(platform_file, 'r') as f:
            content = f.read()
            print(f"📄 文件内容前50字符: {repr(content[:50])}")
        
        # 尝试启动
        try:
            print("🚀 尝试简单平台启动...")
            simulation.start(platform_file, "controller_host")
            print("✅ 简单平台启动成功！")
            
            # 获取主机信息
            hostnames = simulation.get_all_hostnames()
            print(f"✅ 主机列表: {hostnames}")
            
            simulation.shutdown()
            print("✅ 仿真关闭")
            return True
            
        except Exception as e:
            print(f"❌ 简单平台失败: {e}")
        
        # 尝试2: 使用绝对路径
        import os
        abs_platform_file = os.path.abspath(platform_file)
        print(f"🔄 尝试绝对路径: {abs_platform_file}")
        
        simulation2 = wrench.Simulation()
        try:
            simulation2.start(abs_platform_file, "controller_host")
            print("✅ 绝对路径成功！")
            simulation2.shutdown()
            return True
        except Exception as e:
            print(f"❌ 绝对路径失败: {e}")
        
        # 尝试3: 检查WRENCH源码示例格式
        wrench_example_platform = '''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "http://simgrid.gforge.inria.fr/simgrid/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1000000000f">
      <disk id="large_disk" read_bw="100000000Bps" write_bw="100000000Bps">
        <prop id="size" value="5000000000000B"/>
        <prop id="mount" value="/"/>
      </disk>
    </host>
    <host id="compute_host" speed="1000000000f">
      <disk id="large_disk" read_bw="100000000Bps" write_bw="100000000Bps">
        <prop id="size" value="5000000000000B"/>
        <prop id="mount" value="/"/>
      </disk>
    </host>
  </zone>
</platform>'''
        
        platform_file3 = "/tmp/wrench_example_platform.xml"
        with open(platform_file3, 'w') as f:
            f.write(wrench_example_platform)
        print(f"✅ WRENCH示例格式文件创建: {platform_file3}")
        
        simulation3 = wrench.Simulation()
        try:
            simulation3.start(platform_file3, "controller_host")
            print("✅ WRENCH示例格式成功！")
            simulation3.shutdown()
            return True
        except Exception as e:
            print(f"❌ WRENCH示例格式失败: {e}")
        
        # 尝试4: 检查WRENCH是否有内置平台
        print("🔍 检查WRENCH API中是否有平台创建方法...")
        simulation4 = wrench.Simulation()
        
        # 列出所有可用方法
        methods = [m for m in dir(simulation4) if not m.startswith('_')]
        platform_methods = [m for m in methods if 'platform' in m.lower()]
        print(f"平台相关方法: {platform_methods}")
        
        create_methods = [m for m in methods if 'create' in m.lower()]
        print(f"创建相关方法: {create_methods}")
        
        # 尝试5: 看看是否能不用平台文件直接启动
        try:
            print("🔄 尝试无参数启动...")
            simulation5 = wrench.Simulation()
            simulation5.start()
            print("✅ 无参数启动成功！")
            simulation5.shutdown()
            return True
        except Exception as e:
            print(f"❌ 无参数启动失败: {e}")
        
        return False
            
    except ImportError as e:
        print(f"❌ WRENCH导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_platform_file_encoding():
    """检查平台文件编码问题"""
    print("\n🔍 检查文件编码...")
    
    # 创建不同编码的文件
    platform_content = '''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1Gf"/>
  </zone>
</platform>'''
    
    # UTF-8 编码
    utf8_file = "/tmp/platform_utf8.xml"
    with open(utf8_file, 'w', encoding='utf-8') as f:
        f.write(platform_content)
    print(f"✅ UTF-8 文件创建: {utf8_file}")
    
    # ASCII 编码
    ascii_file = "/tmp/platform_ascii.xml"
    with open(ascii_file, 'w', encoding='ascii') as f:
        f.write(platform_content)
    print(f"✅ ASCII 文件创建: {ascii_file}")
    
    # 二进制模式
    binary_file = "/tmp/platform_binary.xml"
    with open(binary_file, 'wb') as f:
        f.write(platform_content.encode('utf-8'))
    print(f"✅ 二进制文件创建: {binary_file}")
    
    # 测试每个文件
    import wrench
    for file_path, encoding in [(utf8_file, "UTF-8"), (ascii_file, "ASCII"), (binary_file, "Binary")]:
        try:
            simulation = wrench.Simulation()
            simulation.start(file_path, "controller_host")
            print(f"✅ {encoding} 编码成功！")
            simulation.shutdown()
            return True
        except Exception as e:
            print(f"❌ {encoding} 编码失败: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 WRENCH 平台文件调试")
    print("=" * 50)
    
    success = test_wrench_debug_platform()
    
    if not success:
        print("\n" + "=" * 50)
        success = check_platform_file_encoding()
    
    if success:
        print("\n🎉 找到了工作的解决方案!")
    else:
        print("\n⚠️  需要进一步调试")
