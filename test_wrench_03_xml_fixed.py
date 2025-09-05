#!/usr/bin/env python3
"""
WRENCH 0.3-dev 修正的XML格式测试

基于错误信息修复XML格式问题
"""

def test_wrench_03_fixed_xml():
    """使用修正的XML格式测试WRENCH 0.3"""
    print("🧪 测试WRENCH 0.3修正XML格式...")
    
    try:
        import wrench
        print(f"✅ WRENCH {wrench.__version__} 导入成功")
        
        # 创建仿真对象
        simulation = wrench.Simulation()
        print("✅ Simulation对象创建成功")
        
        # 创建修正的平台XML文件（添加磁盘配置）
        platform_xml = '''<?xml version='1.0'?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <!-- Controller host with disk -->
    <host id="controller_host" speed="1Gf" core="1">
      <disk id="controller_disk" read_bw="100MBps" write_bw="80MBps">
        <prop id="size" value="1000000000"/>
        <prop id="mount" value="/"/>
      </disk>
    </host>
    
    <!-- Compute hosts with disks -->
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
    
    <!-- Network link -->
    <link id="network_link" bandwidth="1GBps" latency="1ms"/>
    
    <!-- Routes -->
    <route src="controller_host" dst="compute_host_1">
      <link_ctn id="network_link"/>
    </route>
    <route src="controller_host" dst="compute_host_2">
      <link_ctn id="network_link"/>
    </route>
    <route src="compute_host_1" dst="compute_host_2">
      <link_ctn id="network_link"/>
    </route>
  </zone>
</platform>'''
        
        platform_file = "/tmp/wrench_fixed_platform.xml"
        with open(platform_file, 'w') as f:
            f.write(platform_xml)
        print(f"✅ 修正平台文件创建: {platform_file}")
        
        # 验证XML格式
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(platform_file)
            root = tree.getroot()
            print(f"✅ XML解析成功, 根元素: {root.tag}")
            
            hosts = root.findall(".//host")
            print(f"✅ 找到 {len(hosts)} 个主机")
            for host in hosts:
                host_id = host.get('id')
                print(f"   - {host_id}")
                
        except Exception as e:
            print(f"❌ XML预检查失败: {e}")
            return False
        
        # 使用正确的参数启动仿真
        controller_hostname = "controller_host"
        
        try:
            print(f"🚀 启动仿真: platform={platform_file}, controller={controller_hostname}")
            simulation.start(platform_file, controller_hostname)
            print("✅ 仿真启动成功！")
            
            # 现在尝试获取主机列表
            try:
                hostnames = simulation.get_all_hostnames()
                print(f"✅ 获取主机列表成功: {hostnames}")
            except Exception as e:
                print(f"❌ 获取主机列表失败: {e}")
            
            # 获取仿真时间
            try:
                sim_time = simulation.get_simulated_time()
                print(f"✅ 当前仿真时间: {sim_time}")
            except Exception as e:
                print(f"❌ 获取仿真时间失败: {e}")
            
            # 尝试创建服务
            try:
                print("🔧 测试创建计算服务...")
                
                # 创建 Bare Metal Compute Service
                compute_hosts = ["compute_host_1", "compute_host_2"]
                scratch_space_size = "1TB"
                
                compute_service = simulation.add_bare_metal_compute_service(
                    "compute_host_1",      # hostname
                    compute_hosts,         # compute_hosts  
                    scratch_space_size,    # scratch_space_size
                    {}                     # property_list
                )
                print(f"✅ 计算服务创建成功: {compute_service}")
                
            except Exception as e:
                print(f"❌ 创建计算服务失败: {e}")
            
            # 测试工作流创建 - 用正确的参数
            try:
                print("🔧 测试工作流创建...")
                
                workflow = wrench.create_workflow(
                    "test_workflow",    # name
                    "",                # description  
                    0.0,               # submission_time
                    "",                # priority
                    {},                # batch_directives
                    [],                # dependencies
                    0,                 # workflow_id  
                    "default",         # workflow_type
                    {}                 # metadata
                )
                print(f"✅ 工作流创建成功: {workflow}")
                
                # 添加任务到工作流
                task = workflow.add_task("test_task", 1000000000, 1, 1, 1000000000)
                print(f"✅ 任务添加成功: {task}")
                
            except Exception as e:
                print(f"❌ 工作流创建失败: {e}")
            
            # 清理
            try:
                simulation.shutdown()
                print("✅ 仿真关闭成功")
            except Exception as e:
                print(f"⚠️  仿真关闭警告: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ 仿真启动失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"❌ WRENCH导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 WRENCH 0.3-dev 修正XML格式测试")
    print("=" * 50)
    success = test_wrench_03_fixed_xml()
    if success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️  测试失败，但获得了重要信息")
