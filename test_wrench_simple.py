#!/usr/bin/env python3
"""
简化的WRENCH测试脚本

专门用于在有WRENCH环境的测试机器上验证我们的代码。
只测试WRENCH集成的核心功能，不依赖其他复杂的ML库。
"""

import sys
import os

def test_wrench_basic():
    """测试WRENCH基础功能"""
    print("🔍 测试1: WRENCH基础功能")
    
    try:
        import wrench
        print(f"✅ WRENCH {wrench.__version__} 导入成功")
        
        # 创建仿真对象
        simulation = wrench.Simulation()
        print("✅ WRENCH仿真对象创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ WRENCH基础测试失败: {e}")
        return False

def test_our_simulator():
    """测试我们的WRENCHSimulator封装"""
    print("\n🔍 测试2: WRENCHSimulator封装")
    
    try:
        sys.path.append('.')
        from wrench_integration.simulator import WRENCHSimulator
        
        # 创建模拟器
        simulator = WRENCHSimulator()
        print("✅ WRENCHSimulator创建成功")
        
        # 测试平台创建
        platform_config = {
            'hosts': [
                {'id': 'test_node', 'speed': '1Gf', 'cores': 2}
            ],
            'links': [],
            'routes': []
        }
        
        platform_file = simulator.create_platform(platform_config)
        print(f"✅ 平台文件创建: {platform_file}")
        
        # 测试工作流创建
        workflow_spec = {
            'name': 'test_workflow',
            'tasks': [
                {
                    'id': 'task1',
                    'flops': 1e9,
                    'bytes_read': 1e6,
                    'bytes_written': 1e6,
                    'dependencies': []
                }
            ]
        }
        
        workflow_id = simulator.create_workflow(workflow_spec)
        print(f"✅ 工作流创建: {workflow_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ WRENCHSimulator测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulation_run():
    """测试完整仿真运行"""
    print("\n🔍 测试3: 完整仿真运行")
    
    try:
        from wrench_integration.simulator import test_wrench_integration
        
        result = test_wrench_integration()
        if result:
            print("✅ 完整仿真测试成功")
        else:
            print("❌ 完整仿真测试失败")
        
        return result
        
    except Exception as e:
        print(f"❌ 完整仿真测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 WASS-RAG WRENCH集成测试")
    print("="*50)
    
    # 运行测试
    results = []
    
    results.append(test_wrench_basic())
    results.append(test_our_simulator())
    results.append(test_simulation_run())
    
    # 总结
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 测试总结")
    print("="*30)
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 所有测试通过！WRENCH集成工作正常。")
        return 0
    else:
        print("⚠️  部分测试失败，需要修复问题。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
