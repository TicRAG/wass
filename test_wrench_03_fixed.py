#!/usr/bin/env python3
"""
WRENCH 0.3-dev 正确参数测试

基于错误信息修复API调用
"""

def test_wrench_03_correct_api():
    """使用正确的参数测试WRENCH 0.3"""
    print("🧪 测试WRENCH 0.3正确API...")
    
    try:
        import wrench
        print(f"✅ WRENCH {wrench.__version__} 导入成功")
        
        # 创建仿真对象
        simulation = wrench.Simulation()
        print("✅ Simulation对象创建成功")
        
        # 创建平台XML文件
        platform_xml = '''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller_host" speed="1Gf" core="1"/>
    <host id="compute_host_1" speed="2Gf" core="4"/>
    <host id="compute_host_2" speed="1.5Gf" core="2"/>
    <link id="link1" bandwidth="1GBps" latency="0.001s"/>
    <route src="controller_host" dst="compute_host_1">
      <link_ctn id="link1"/>
    </route>
    <route src="controller_host" dst="compute_host_2">
      <link_ctn id="link1"/>
    </route>
  </zone>
</platform>'''
        
        platform_file = "/tmp/wrench_03_platform.xml"
        with open(platform_file, 'w') as f:
            f.write(platform_xml)
        print(f"✅ 平台文件创建: {platform_file}")
        
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
            
            # 测试工作流创建（需要完整参数）
            print("\n🔬 测试工作流创建...")
            
            # 方法1: create_workflow()
            try:
                workflow = simulation.create_workflow()
                print(f"✅ create_workflow() 成功: {type(workflow)}")
                
                # 查看工作流方法
                workflow_methods = [attr for attr in dir(workflow) if not attr.startswith('_')]
                print(f"📋 工作流方法数量: {len(workflow_methods)}")
                
                # 查找任务相关方法
                task_methods = [method for method in workflow_methods if 'task' in method.lower()]
                print(f"📋 任务相关方法: {task_methods}")
                
            except Exception as e:
                print(f"❌ create_workflow() 失败: {e}")
            
            # 方法2: create_workflow_from_json() 带完整参数
            try:
                workflow_json = '{"name": "test", "tasks": []}'
                
                # 根据错误信息提供所有必需参数
                workflow2 = simulation.create_workflow_from_json(
                    workflow_json,
                    reference_flop_rate="1Gf",  # 参考FLOP速率
                    ignore_machine_specs=False,  # 是否忽略机器规格
                    redundant_dependencies=True,  # 冗余依赖
                    ignore_cycle_creating_dependencies=False,  # 忽略循环依赖
                    min_cores_per_task=1,  # 每任务最小核心数
                    max_cores_per_task=4,  # 每任务最大核心数
                    enforce_num_cores=False,  # 强制核心数
                    ignore_avg_cpu=False,  # 忽略平均CPU
                    show_warnings=True  # 显示警告
                )
                print(f"✅ create_workflow_from_json() 成功: {type(workflow2)}")
                
            except Exception as e:
                print(f"❌ create_workflow_from_json() 失败: {e}")
            
            # 测试服务创建
            print("\n🛠️  测试服务创建...")
            
            if hostnames:
                # 计算服务
                try:
                    compute_hosts = [h for h in hostnames if 'compute' in h]
                    if compute_hosts:
                        compute_service = simulation.create_bare_metal_compute_service(
                            hostname=compute_hosts[0],
                            compute_hosts=compute_hosts,
                            scratch_space_size="100MB"
                        )
                        print(f"✅ 计算服务创建成功")
                except Exception as e:
                    print(f"❌ 计算服务创建失败: {e}")
                
                # 存储服务
                try:
                    storage_service = simulation.create_simple_storage_service(
                        hostname=hostnames[0]
                    )
                    print(f"✅ 存储服务创建成功")
                except Exception as e:
                    print(f"❌ 存储服务创建失败: {e}")
            
            print("🎉 WRENCH 0.3 API测试完成！")
            return True
            
        except Exception as e:
            print(f"❌ 仿真启动失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_creation_detailed():
    """详细测试工作流创建"""
    print("\n🔬 详细测试工作流创建...")
    
    try:
        import wrench
        simulation = wrench.Simulation()
        
        # 创建平台
        platform_xml = '''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="controller" speed="1Gf" core="1"/>
    <host id="worker1" speed="2Gf" core="4"/>
  </zone>
</platform>'''
        
        platform_file = "/tmp/test_platform.xml"
        with open(platform_file, 'w') as f:
            f.write(platform_xml)
        
        # 启动仿真
        simulation.start(platform_file, "controller")
        print("✅ 仿真已启动")
        
        # 尝试不同的工作流JSON格式
        workflow_formats = [
            # 格式1: 最简单
            '{"name": "simple", "tasks": []}',
            
            # 格式2: 带任务
            '''{"name": "with_task", "tasks": [
                {"name": "task1", "type": "compute", "flops": 1000000000}
            ]}''',
            
            # 格式3: 更完整
            '''{"name": "complete", "tasks": [
                {
                    "name": "task1",
                    "type": "compute", 
                    "flops": 1000000000,
                    "bytes_read": 1000000,
                    "bytes_written": 1000000,
                    "dependencies": []
                }
            ]}'''
        ]
        
        for i, wf_json in enumerate(workflow_formats, 1):
            print(f"\n📝 测试工作流格式 {i}:")
            try:
                workflow = simulation.create_workflow_from_json(
                    wf_json,
                    reference_flop_rate="1Gf",
                    ignore_machine_specs=False,
                    redundant_dependencies=True,
                    ignore_cycle_creating_dependencies=False,
                    min_cores_per_task=1,
                    max_cores_per_task=4,
                    enforce_num_cores=False,
                    ignore_avg_cpu=False,
                    show_warnings=True
                )
                print(f"✅ 格式 {i} 成功: {type(workflow)}")
                
                # 查看工作流内容
                if hasattr(workflow, 'get_tasks'):
                    try:
                        tasks = workflow.get_tasks()
                        print(f"   📋 任务数量: {len(tasks)}")
                    except:
                        pass
                
            except Exception as e:
                print(f"❌ 格式 {i} 失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 详细工作流测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 WRENCH 0.3-dev 正确API测试")
    print("="*50)
    
    success1 = test_wrench_03_correct_api()
    success2 = test_workflow_creation_detailed()
    
    if success1 and success2:
        print("\n🎉 WRENCH 0.3 API测试成功！现在我们知道如何正确使用了。")
    else:
        print("\n⚠️  部分测试失败，但获得了重要信息")
