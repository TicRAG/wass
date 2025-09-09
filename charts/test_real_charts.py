#!/usr/bin/env python3
"""
测试真实实验数据的图表生成
"""

import os
import sys
import json

def test_real_data_loading():
    """测试真实数据加载"""
    
    print("🧪 测试真实实验数据图表生成")
    print("=" * 60)
    
    # 检查数据文件
    data_file = "../results/real_experiments/experiment_results.json"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    # 检查数据格式
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 成功加载数据文件")
        print(f"📊 数据条目数量: {len(data)}")
        
        # 检查数据结构
        if data:
            first_item = data[0]
            print(f"📝 数据字段: {list(first_item.keys())}")
            
            # 检查关键字段
            required_fields = ['scheduling_method', 'makespan', 'cpu_utilization', 'cluster_size']
            missing_fields = [field for field in required_fields if field not in first_item]
            
            if missing_fields:
                print(f"⚠️ 缺少字段: {missing_fields}")
            else:
                print("✅ 数据格式验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_chart_generation():
    """测试图表生成"""
    
    print("\n📊 测试图表生成...")
    
    try:
        from paper_charts import PaperChartGenerator
        
        # 创建生成器，指向真实数据
        generator = PaperChartGenerator(results_dir="../results")
        
        print("✅ 生成器创建成功")
        
        # 尝试加载数据
        results = generator.load_experimental_results()
        
        if not results:
            print("❌ 没有加载到数据")
            return False
        
        print("✅ 数据加载成功")
        
        # 生成单个图表测试
        print("\n🔥 生成性能热力图...")
        heatmap_path = generator.generate_performance_heatmap(results)
        print(f"✅ 热力图生成成功: {heatmap_path}")
        
        print("\n📡 生成雷达图...")
        radar_path = generator.generate_radar_chart(results)
        print(f"✅ 雷达图生成成功: {radar_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 图表生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_generation():
    """测试完整图表生成"""
    
    print("\n🎨 测试完整图表生成...")
    
    try:
        from paper_charts import PaperChartGenerator
        
        generator = PaperChartGenerator(results_dir="../results")
        
        # 生成所有图表
        chart_paths = generator.generate_all_charts()
        
        print("✅ 所有图表生成成功!")
        print("\n📁 生成的图表:")
        for chart_type, path in chart_paths.items():
            print(f"   • {chart_type}: {os.path.basename(path)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    
    print("🎯 WASS-RAG 真实数据图表生成测试")
    print("=" * 80)
    
    # 步骤1: 测试数据加载
    if not test_real_data_loading():
        print("\n❌ 数据加载测试失败，无法继续")
        sys.exit(1)
    
    # 步骤2: 测试单个图表生成
    if not test_chart_generation():
        print("\n❌ 图表生成测试失败")
        sys.exit(1)
    
    # 步骤3: 测试完整生成
    if not test_full_generation():
        print("\n❌ 完整生成测试失败")
        sys.exit(1)
    
    print("\n🎉 所有测试通过！")
    print("📊 真实实验数据图表生成系统工作正常")
    print("🎯 图表已准备好用于ACM论文提交")
