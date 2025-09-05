#!/usr/bin/env python3
"""
测试包准备脚本

这个脚本用于创建WRENCH测试包，包含所有必要的文件。
"""

import os
import shutil
import sys
from pathlib import Path

def create_test_package():
    """创建WRENCH测试包"""
    
    # 测试包目录
    test_dir = "wass_wrench_test"
    
    # 清理已存在的测试包
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # 创建测试包目录
    os.makedirs(test_dir)
    
    # 需要拷贝的目录和文件
    items_to_copy = [
        # 目录
        "wrench_integration/",
        "experiments/", 
        "src/",
        "configs/",
        
        # 文件
        "requirements_wrench.txt",
        "run_wrench_tests.py",
        "WRENCH_TEST_GUIDE.md",
        "README_WRENCH_TEST.md"
    ]
    
    print("📦 创建WRENCH测试包...")
    
    for item in items_to_copy:
        src = item
        dst = os.path.join(test_dir, item)
        
        if os.path.isdir(src):
            print(f"  📁 拷贝目录: {src}")
            shutil.copytree(src, dst)
            
            # 创建__init__.py文件
            init_file = os.path.join(dst, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# WASS-RAG module\n")
                    
        elif os.path.isfile(src):
            print(f"  📄 拷贝文件: {src}")
            shutil.copy2(src, dst)
        else:
            print(f"  ⚠️  跳过不存在的项目: {src}")
    
    # 创建打包脚本
    pack_script = os.path.join(test_dir, "pack_for_upload.sh")
    with open(pack_script, 'w', encoding='utf-8') as f:
        f.write("""#!/bin/bash
# 打包测试包
echo "📦 打包WRENCH测试包..."
cd ..
tar -czf wass_wrench_test.tar.gz wass_wrench_test/
echo "✅ 测试包已创建: wass_wrench_test.tar.gz"
echo "现在可以上传到测试机器了："
echo "scp wass_wrench_test.tar.gz user@test-machine:~/"
""")
    os.chmod(pack_script, 0o755)
    
    # 创建快速测试脚本
    quick_test = os.path.join(test_dir, "quick_test.py")
    with open(quick_test, 'w', encoding='utf-8') as f:
        f.write("""#!/usr/bin/env python3
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
""")
    os.chmod(quick_test, 0o755)
    
    print(f"\n✅ 测试包创建完成: {test_dir}/")
    print("\n📋 测试包内容:")
    
    # 列出测试包内容
    for root, dirs, files in os.walk(test_dir):
        level = root.replace(test_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # 只显示前几个文件，避免输出太长
        subindent = ' ' * 2 * (level + 1)
        for i, file in enumerate(files):
            if i < 5:  # 只显示前5个文件
                print(f"{subindent}{file}")
            elif i == 5:
                print(f"{subindent}... ({len(files)-5} more files)")
                break
    
    print(f"\n🚀 下一步:")
    print(f"1. 运行: cd {test_dir} && ./pack_for_upload.sh")
    print(f"2. 上传: scp wass_wrench_test.tar.gz user@test-machine:~/")
    print(f"3. 在测试机器上解压并运行测试")

if __name__ == "__main__":
    create_test_package()
