#!/usr/bin/env python3
"""
测试综合摘要图表修复
"""

def test_summary_chart():
    """测试综合摘要图表生成"""
    
    print("🧪 测试综合摘要图表修复")
    print("=" * 50)
    
    try:
        from paper_charts import PaperChartGenerator
        
        # 创建生成器
        generator = PaperChartGenerator(results_dir=".")
        print("✅ 生成器创建成功")
        
        # 加载数据
        results = generator.load_experimental_results()
        print("✅ 数据加载成功")
        
        # 测试综合摘要生成
        print("\n📈 生成综合摘要图表...")
        summary_path = generator.generate_combined_summary(results)
        print(f"✅ 综合摘要生成成功: {summary_path}")
        
        # 检查文件
        import os
        if os.path.exists(summary_path):
            file_size = os.path.getsize(summary_path) / 1024
            print(f"✅ 文件已保存，大小: {file_size:.1f} KB")
        else:
            print("❌ 文件未生成")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_generation():
    """测试完整图表生成"""
    
    print("\n🎨 测试完整图表生成")
    print("=" * 50)
    
    try:
        from paper_charts import PaperChartGenerator
        
        generator = PaperChartGenerator(results_dir=".")
        
        # 生成所有图表
        chart_paths = generator.generate_all_charts()
        
        print("✅ 所有图表生成成功!")
        print("\n📁 生成的图表:")
        
        import os
        for chart_type, path in chart_paths.items():
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024
                print(f"   • {chart_type}: {os.path.basename(path)} ({size:.1f} KB)")
            else:
                print(f"   ❌ {chart_type}: 文件未生成")
                
        return True
        
    except Exception as e:
        print(f"❌ 完整生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    
    print("🎯 WASS-RAG 综合摘要图表修复测试")
    print("=" * 80)
    
    # 先测试单个图表
    if test_summary_chart():
        print("\n✅ 综合摘要图表测试通过!")
        
        # 再测试完整生成
        if test_full_generation():
            print("\n🎉 所有测试通过!")
            print("💡 现在可以安全运行 python paper_charts.py")
        else:
            print("\n⚠️ 完整生成测试失败")
    else:
        print("\n❌ 综合摘要图表测试失败")
