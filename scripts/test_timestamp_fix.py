#!/usr/bin/env python3
"""
测试时间戳修复的简单脚本
"""

def test_timestamp_fix():
    """测试numpy datetime64的字符串转换"""
    try:
        import numpy as np
        
        # 测试原来有问题的代码
        try:
            # 这个会失败
            ts_fail = np.datetime64('now').isoformat()
            print("❌ 意外成功: isoformat() 应该会失败")
        except AttributeError as e:
            print("✓ 确认问题: np.datetime64('now').isoformat() 确实失败")
            print(f"  错误: {e}")
        
        # 测试修复后的代码
        ts_fixed = str(np.datetime64('now'))
        print(f"✓ 修复成功: str(np.datetime64('now')) = {ts_fixed}")
        
        # 验证格式
        if len(ts_fixed) >= 10 and 'T' in ts_fixed:
            print("✓ 时间戳格式正确 (ISO 8601格式)")
        else:
            print(f"⚠ 时间戳格式可能有问题: {ts_fixed}")
        
        return True
        
    except ImportError:
        print("⚠ numpy 不可用，无法测试时间戳修复")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 时间戳修复测试 ===")
    
    success = test_timestamp_fix()
    
    if success:
        print("\n🎉 时间戳修复测试通过!")
        print("现在您可以在服务器上运行:")
        print("  python scripts/initialize_ai_models.py")
    else:
        print("\n❌ 时间戳修复测试失败")
