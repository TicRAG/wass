#!/usr/bin/env python3
"""
WRENCH 0.3-dev 适配测试

基于探测到的API重新实现WRENCH集成
"""

def test_wrench_03_workflow():
    """测试WRENCH 0.3的工作流功能"""
    print("🧪 测试WRENCH 0.3工作流功能...")
    
    try:
        import wrench
        print(f"✅ WRENCH {wrench.__version__} 导入成功")
        
        # 创建仿真对象
        simulation = wrench.Simulation()
        print("✅ Simulation对象创建成功")
        
        # 获取所有主机名（这是0.3版本获取平台信息的方法）
        try:
            hostnames = simulation.get_all_hostnames()
            print(f"✅ 获取主机列表成功: {hostnames}")
        except Exception as e:
            print(f"❌ 获取主机列表失败: {e}")
            print("🔧 可能需要先启动仿真或配置平台")
            
            # 尝试启动仿真看看会发生什么
            try:
                print("🚀 尝试启动空仿真...")
                simulation.start()
                print("✅ 空仿真启动成功")
                
                # 再次尝试获取主机
                hostnames = simulation.get_all_hostnames()
                print(f"✅ 启动后获取主机列表: {hostnames}")
                
            except Exception as e2:
                print(f"❌ 启动空仿真失败: {e2}")
                return False
        
        # 尝试创建工作流
        try:
            workflow = simulation.create_workflow()
            print(f"✅ 工作流创建成功: {type(workflow)}")
            
            # 检查工作流对象的方法
            workflow_methods = [attr for attr in dir(workflow) if not attr.startswith('_')]
            print(f"📋 工作流可用方法: {len(workflow_methods)}")
            for method in workflow_methods[:10]:  # 只显示前10个
                print(f"   - {method}")
            if len(workflow_methods) > 10:
                print(f"   ... 还有{len(workflow_methods)-10}个方法")
                
        except Exception as e:
            print(f"❌ 工作流创建失败: {e}")
            return False
        
        # 尝试创建任务
        try:
            # 检查是否有Task类
            if hasattr(wrench, 'Task'):
                task = wrench.Task()
                print(f"✅ 任务对象创建成功: {type(task)}")
            else:
                print("ℹ️  没有独立的Task类")
                
        except Exception as e:
            print(f"❌ 任务创建失败: {e}")
        
        # 尝试创建服务
        print("\n🛠️  测试计算服务...")
        try:
            # 这个版本可能需要先有主机才能创建服务
            if hostnames:
                # 尝试创建裸机计算服务
                try:
                    compute_service = simulation.create_bare_metal_compute_service(
                        hostname=hostnames[0],
                        compute_hosts=hostnames,
                        scratch_space_size="100MB"
                    )
                    print(f"✅ 裸机计算服务创建成功")
                except Exception as e:
                    print(f"❌ 裸机计算服务创建失败: {e}")
                
                # 尝试创建存储服务
                try:
                    storage_service = simulation.create_simple_storage_service(
                        hostname=hostnames[0]
                    )
                    print(f"✅ 存储服务创建成功")
                except Exception as e:
                    print(f"❌ 存储服务创建失败: {e}")
            else:
                print("⚠️  没有可用主机，跳过服务创建")
        
        except Exception as e:
            print(f"❌ 服务测试失败: {e}")
        
        # 测试事件系统
        print("\n📡 测试事件系统...")
        try:
            events = simulation.get_events()
            print(f"✅ 获取事件成功: {len(events)} 个事件")
        except Exception as e:
            print(f"❌ 获取事件失败: {e}")
        
        # 测试时间
        try:
            sim_time = simulation.get_simulated_time()
            print(f"✅ 当前仿真时间: {sim_time}")
        except Exception as e:
            print(f"❌ 获取仿真时间失败: {e}")
        
        print("🎉 WRENCH 0.3基础功能探测完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def explore_workflow_creation():
    """深入探索工作流创建"""
    print("\n🔬 深入探索工作流创建...")
    
    try:
        import wrench
        simulation = wrench.Simulation()
        
        # 方法1: create_workflow()
        try:
            workflow1 = simulation.create_workflow()
            print(f"✅ create_workflow() 成功: {type(workflow1)}")
            
            # 查看工作流的详细方法
            workflow_methods = [attr for attr in dir(workflow1) if not attr.startswith('_')]
            task_methods = [method for method in workflow_methods if 'task' in method.lower()]
            print(f"📋 任务相关方法: {task_methods}")
            
        except Exception as e:
            print(f"❌ create_workflow() 失败: {e}")
        
        # 方法2: create_workflow_from_json()
        try:
            # 创建简单的JSON工作流描述
            workflow_json = {
                "name": "test_workflow",
                "tasks": [
                    {
                        "name": "task1",
                        "type": "compute",
                        "flops": 1000000000,
                        "dependencies": []
                    }
                ]
            }
            
            import json
            json_str = json.dumps(workflow_json)
            workflow2 = simulation.create_workflow_from_json(json_str)
            print(f"✅ create_workflow_from_json() 成功: {type(workflow2)}")
            
        except Exception as e:
            print(f"❌ create_workflow_from_json() 失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工作流探索失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 WRENCH 0.3-dev 深度测试")
    print("="*50)
    
    success1 = test_wrench_03_workflow()
    success2 = explore_workflow_creation()
    
    if success1 and success2:
        print("\n🎉 所有测试完成！我们现在了解了WRENCH 0.3的工作方式。")
        print("📝 下一步: 基于这些信息重写我们的WRENCHSimulator类")
    else:
        print("\n⚠️  部分测试失败，但我们已经获得了有用的信息")
