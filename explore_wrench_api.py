#!/usr/bin/env python3
"""
WRENCH API探测工具

用于检查WRENCH版本和可用的API方法，帮助我们适配不同版本的WRENCH。
"""

def explore_wrench_api():
    """探测WRENCH API"""
    print("🔍 探测WRENCH API...")
    
    try:
        import wrench
        print(f"✅ WRENCH版本: {wrench.__version__}")
        
        # 检查Simulation类的方法
        print("\n📋 检查Simulation类...")
        simulation = wrench.Simulation()
        
        # 获取所有方法
        all_methods = [attr for attr in dir(simulation) if not attr.startswith('_')]
        print(f"   可用方法总数: {len(all_methods)}")
        
        # 检查平台相关方法
        platform_methods = [attr for attr in all_methods if 'platform' in attr.lower()]
        print(f"\n🏗️  平台相关方法:")
        for method in platform_methods:
            print(f"   - {method}")
        
        # 检查仿真控制方法
        sim_control_methods = [attr for attr in all_methods 
                              if any(keyword in attr.lower() 
                                   for keyword in ['start', 'run', 'launch', 'execute'])]
        print(f"\n⚙️  仿真控制方法:")
        for method in sim_control_methods:
            print(f"   - {method}")
        
        # 检查工作流相关方法
        workflow_methods = [attr for attr in all_methods if 'workflow' in attr.lower()]
        print(f"\n📊 工作流相关方法:")
        for method in workflow_methods:
            print(f"   - {method}")
        
        # 检查文件/数据相关方法
        file_methods = [attr for attr in all_methods 
                       if any(keyword in attr.lower() 
                            for keyword in ['file', 'data', 'load', 'add'])]
        print(f"\n📄 文件/数据方法:")
        for method in file_methods:
            print(f"   - {method}")
        
        # 尝试检查方法签名
        print(f"\n🔧 方法详情:")
        key_methods = ['instantiatePlatform', 'add_platform', 'loadPlatform', 'start', 'launch', 'run']
        
        for method_name in key_methods:
            if hasattr(simulation, method_name):
                method = getattr(simulation, method_name)
                print(f"   ✅ {method_name}: {type(method)}")
                
                # 尝试获取文档字符串
                if hasattr(method, '__doc__') and method.__doc__:
                    doc = method.__doc__.strip().split('\n')[0]  # 只取第一行
                    print(f"      📝 {doc}")
            else:
                print(f"   ❌ {method_name}: 不存在")
        
        # 检查是否有其他重要的类
        print(f"\n📦 其他WRENCH类:")
        wrench_attrs = [attr for attr in dir(wrench) if not attr.startswith('_')]
        important_classes = ['Workflow', 'Task', 'Job', 'Platform', 'Service']
        
        for class_name in important_classes:
            if hasattr(wrench, class_name):
                print(f"   ✅ wrench.{class_name}")
            else:
                print(f"   ❌ wrench.{class_name}: 不存在")
        
        print(f"\n🎯 完整方法列表:")
        for i, method in enumerate(sorted(all_methods), 1):
            print(f"   {i:2d}. {method}")
            if i % 20 == 0:  # 每20个方法暂停一下
                print("      ...")
        
        return True
        
    except Exception as e:
        print(f"❌ 探测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    explore_wrench_api()
