#!/usr/bin/env python3
"""
最小化WRENCH测试

尝试最基本的WRENCH功能
"""

def minimal_wrench_test():
    """最小化的WRENCH测试"""
    print("🧪 最小化WRENCH测试...")
    
    try:
        import wrench
        print(f"✅ WRENCH导入成功: {wrench.__version__}")
        
        # 只创建对象，不启动仿真
        simulation = wrench.Simulation()
        print("✅ Simulation对象创建成功")
        
        # 检查对象属性
        attrs = [attr for attr in dir(simulation) if not attr.startswith('_')]
        print(f"📋 Simulation对象方法数: {len(attrs)}")
        
        # 检查daemon相关属性
        if hasattr(simulation, 'daemon_url'):
            print(f"🔗 Daemon URL: {simulation.daemon_url}")
        if hasattr(simulation, 'started'):
            print(f"📊 Started: {simulation.started}")
        if hasattr(simulation, 'terminated'):
            print(f"📊 Terminated: {simulation.terminated}")
            
        # 尝试最简单的平台
        minimal_platform = "<platform><host id='h' speed='1Gf'/></platform>"
        
        print("🚀 尝试启动最简单平台...")
        try:
            simulation.start(minimal_platform, "h")
            print("✅ 最简单平台启动成功！")
            simulation.shutdown()
            return True
        except Exception as e:
            print(f"❌ 最简单平台失败: {e}")
            
        # 尝试空平台
        print("🚀 尝试更简单的方式...")
        try:
            # 也许可以传递空字符串或特殊值？
            for test_platform in ["", "<platform/>", "<platform><host id='controller_host' speed='1Gf'/></platform>"]:
                simulation2 = wrench.Simulation()
                try:
                    simulation2.start(test_platform, "controller_host")
                    print(f"✅ 平台 '{test_platform[:20]}...' 成功!")
                    simulation2.shutdown()
                    return True
                except Exception as e:
                    print(f"❌ 平台 '{test_platform[:20]}...' 失败: {str(e)[:50]}...")
        except Exception as e:
            print(f"❌ 替代测试失败: {e}")
            
        return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 最小化WRENCH测试")
    print("=" * 40)
    
    success = minimal_wrench_test()
    if success:
        print("\n🎉 找到了可用的方法!")
    else:
        print("\n💡 建议：可能需要检查WRENCH daemon配置或重启服务")
