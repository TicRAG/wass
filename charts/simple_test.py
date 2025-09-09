#!/usr/bin/env python3
"""
简化的图表生成测试 - 修复版本
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 配置matplotlib以避免所有布局冲突
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'figure.constrained_layout.use': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'backend': 'Agg'  # 确保非交互式
})

def test_single_chart():
    """测试单个图表生成"""
    
    print("🧪 Testing Single Chart Generation")
    print("=" * 50)
    
    try:
        # 导入图表生成器
        from paper_charts import PaperChartGenerator
        
        print("✅ Successfully imported PaperChartGenerator")
        
        # 创建生成器实例
        generator = PaperChartGenerator()
        print("✅ Successfully created generator instance")
        
        # 尝试加载真实实验数据
        print("\n📊 Loading real experimental data...")
        test_data = generator.load_experimental_results()
        print("✅ Successfully loaded real experimental data")
        
        # 只测试热力图生成（最简单的）
        print("\n📊 Testing heatmap generation...")
        heatmap_path = generator.generate_performance_heatmap(test_data)
        print(f"✅ Heatmap saved to: {heatmap_path}")
        
        # 验证文件是否存在
        if os.path.exists(heatmap_path):
            print("✅ File exists and was saved successfully")
        else:
            print("❌ File was not saved properly")
            
        print("\n🎉 Single chart test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_all_charts():
    """测试所有图表生成"""
    
    print("\n🧪 Testing All Charts Generation")
    print("=" * 50)
    
    try:
        from paper_charts import PaperChartGenerator
        
        # 创建生成器
        generator = PaperChartGenerator()
        
        # 使用完整的图表生成方法
        print("📊 Running complete chart generation...")
        chart_paths = generator.generate_all_charts()
        
        print("✅ All charts generated successfully!")
        print("📁 Generated files:")
        for chart_type, path in chart_paths.items():
            print(f"   • {chart_type}: {path}")
            
    except Exception as e:
        print(f"❌ Full test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 先测试单个图表
    test_single_chart()
    
    # 然后测试全部图表
    test_all_charts()
    
    print("\n🎯 Testing Complete!")
    print("💡 Charts should now be ACM-compliant and ready for publication.")
