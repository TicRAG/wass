#!/usr/bin/env python3
# 快速测试WRENCH可用性
try:
    import wrench
    print(f"✅ WRENCH {wrench.__version__} 可用")
    
    # 简单测试
    sim = wrench.Simulation()
    print("✅ WRENCH仿真对象创建成功")
    
    print("🎉 WRENCH环境检查通过！")
except ImportError as e:
    print(f"❌ WRENCH不可用: {e}")
    print("请检查WRENCH安装")
except Exception as e:
    print(f"❌ WRENCH测试失败: {e}")
