#!/usr/bin/env python3
"""
测试正确的WRENCH平台XML格式
"""

def create_correct_platform_xml():
    """
    创建符合SimGrid DTD的正确平台XML
    """
    # 基于SimGrid官方文档的正确格式
    platform_xml = '''<?xml version='1.0'?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <!-- Controller host -->
    <host id="controller_host" speed="1Gf" core="1">
      <disk id="controller_disk" read_bw="100MBps" write_bw="80MBps">
        <prop id="size" value="1000000000"/>
        <prop id="mount" value="/"/>
      </disk>
    </host>
    
    <!-- Compute hosts -->
    <host id="compute_host_1" speed="2Gf" core="4">
      <disk id="compute_disk_1" read_bw="100MBps" write_bw="80MBps">
        <prop id="size" value="1000000000"/>
        <prop id="mount" value="/"/>
      </disk>
    </host>
    
    <host id="compute_host_2" speed="1.5Gf" core="2">
      <disk id="compute_disk_2" read_bw="100MBps" write_bw="80MBps">
        <prop id="size" value="1000000000"/>
        <prop id="mount" value="/"/>
      </disk>
    </host>
    
    <!-- Storage host -->
    <host id="storage_host" speed="1Gf" core="1">
      <disk id="storage_disk" read_bw="200MBps" write_bw="150MBps">
        <prop id="size" value="10000000000"/>
        <prop id="mount" value="/storage"/>
      </disk>
    </host>
    
    <!-- Network links -->
    <link id="network_link" bandwidth="1GBps" latency="1ms"/>
    
    <!-- Routes between hosts -->
    <route src="controller_host" dst="compute_host_1">
      <link_ctn id="network_link"/>
    </route>
    <route src="controller_host" dst="compute_host_2">
      <link_ctn id="network_link"/>
    </route>
    <route src="controller_host" dst="storage_host">
      <link_ctn id="network_link"/>
    </route>
    <route src="compute_host_1" dst="compute_host_2">
      <link_ctn id="network_link"/>
    </route>
    <route src="compute_host_1" dst="storage_host">
      <link_ctn id="network_link"/>
    </route>
    <route src="compute_host_2" dst="storage_host">
      <link_ctn id="network_link"/>
    </route>
  </zone>
</platform>'''
    
    return platform_xml

def test_xml_platform():
    """
    测试XML平台文件的有效性
    """
    print("🧪 测试XML平台文件...")
    
    # 创建平台文件
    platform_xml = create_correct_platform_xml()
    platform_file = "/tmp/correct_platform.xml"
    
    with open(platform_file, 'w') as f:
        f.write(platform_xml)
    
    print(f"✅ 平台文件创建: {platform_file}")
    
    # 验证XML语法
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(platform_file)
        root = tree.getroot()
        print(f"✅ XML解析成功, 根元素: {root.tag}")
        
        # 显示主机信息
        hosts = root.findall(".//host")
        print(f"✅ 找到 {len(hosts)} 个主机:")
        for host in hosts:
            host_id = host.get('id')
            speed = host.get('speed')
            cores = host.get('core')
            print(f"   - {host_id}: {speed}, {cores} cores")
            
    except Exception as e:
        print(f"❌ XML解析失败: {e}")
        return False
    
    # 测试WRENCH
    try:
        import wrench
        simulation = wrench.Simulation()
        controller_hostname = "controller_host"
        
        print(f"🚀 启动WRENCH仿真...")
        simulation.start(platform_file, controller_hostname)
        print("✅ WRENCH仿真启动成功!")
        
        # 获取主机列表
        hostnames = simulation.get_all_hostnames()
        print(f"✅ 可用主机: {hostnames}")
        
        # 清理
        simulation.shutdown()
        print("✅ 仿真关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ WRENCH测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 XML平台文件测试")
    print("=" * 50)
    success = test_xml_platform()
    if success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️  测试失败")
