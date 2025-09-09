#!/usr/bin/env python3
"""
测试路径修复
"""

import os

def test_file_paths():
    """测试各种可能的文件路径"""
    
    print("🔍 测试实验数据文件路径")
    print("=" * 50)
    
    # 所有可能的路径
    test_paths = [
        "results/real_experiments/experiment_results.json",
        "experiments/results/real_experiments/experiment_results.json", 
        "../experiments/results/real_experiments/experiment_results.json",
        "./experiments/results/real_experiments/experiment_results.json"
    ]
    
    found_files = []
    
    for path in test_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"✅ 找到: {path}")
            print(f"   绝对路径: {abs_path}")
            print(f"   文件大小: {size:.1f} KB")
            found_files.append(path)
        else:
            print(f"❌ 不存在: {path}")
    
    if found_files:
        print(f"\n✅ 总共找到 {len(found_files)} 个数据文件")
        return found_files[0]  # 返回第一个找到的文件
    else:
        print("\n❌ 未找到任何实验数据文件")
        return None

def test_data_loading():
    """测试数据加载"""
    
    data_file = test_file_paths()
    
    if not data_file:
        return False
    
    print(f"\n📊 测试数据加载: {data_file}")
    print("-" * 50)
    
    try:
        import json
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 数据加载成功")
        print(f"📊 数据类型: {type(data)}")
        
        if isinstance(data, list):
            print(f"📊 实验数量: {len(data)}")
            if data:
                print(f"📊 第一个实验字段: {list(data[0].keys())}")
        elif isinstance(data, dict):
            print(f"📊 字典键: {list(data.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_chart_generator():
    """测试图表生成器"""
    
    print(f"\n🎨 测试图表生成器")
    print("-" * 50)
    
    try:
        from paper_charts import PaperChartGenerator
        
        # 尝试不同的results_dir设置
        for results_dir in ["results", ".", ".."]:
            print(f"\n尝试 results_dir: {results_dir}")
            try:
                generator = PaperChartGenerator(results_dir=results_dir)
                results = generator.load_experimental_results()
                print(f"✅ 成功加载数据，results_dir: {results_dir}")
                
                if 'experiments' in results:
                    print(f"📊 实验数量: {len(results['experiments'])}")
                
                return True
                
            except Exception as e:
                print(f"❌ 失败，results_dir: {results_dir}, 错误: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"❌ 图表生成器导入失败: {e}")
        return False

if __name__ == "__main__":
    
    print("🧪 WASS-RAG 路径修复测试")
    print("=" * 80)
    
    # 当前工作目录
    print(f"📁 当前工作目录: {os.getcwd()}")
    
    # 测试文件路径
    data_file = test_file_paths()
    
    if data_file:
        # 测试数据加载
        if test_data_loading():
            # 测试图表生成器
            if test_chart_generator():
                print("\n🎉 所有测试通过!")
                print("💡 现在可以运行 python paper_charts.py")
            else:
                print("\n⚠️ 图表生成器测试失败")
        else:
            print("\n⚠️ 数据加载测试失败")
    else:
        print("\n❌ 未找到实验数据文件")
        print("💡 请确保已运行实验: python experiments/real_experiment_framework.py")
