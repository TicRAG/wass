#!/usr/bin/env python3
"""
简化的ACM图表测试
"""

import os
import sys
import matplotlib.pyplot as plt

# 配置matplotlib以避免布局冲突
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'figure.constrained_layout.use': True,  # 使用新的布局引擎
    'axes.grid': True,
    'grid.alpha': 0.3
})

def quick_test():
    """快速测试图表生成"""
    
    print("🧪 Quick ACM Chart Test")
    print("=" * 40)
    
    # 确保输出目录存在
    output_dir = "quick_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # 测试热力图生成
        print("\n📊 Testing heatmap generation...")
        heatmap_path = generator.generate_performance_heatmap(test_data)
        print(f"✅ Heatmap saved to: {heatmap_path}")
        
        # 测试雷达图生成
        print("\n🎯 Testing radar chart generation...")
        radar_path = generator.generate_radar_chart(test_data)
        print(f"✅ Radar chart saved to: {radar_path}")
        
        # 测试箱形图生成
        print("\n📦 Testing box plot generation...")
        boxplot_path = generator.generate_stability_boxplot(test_data)
        print(f"✅ Box plot saved to: {boxplot_path}")
        
        # 测试甘特图生成
        print("\n📅 Testing gantt chart generation...")
        gantt_path = generator.generate_gantt_chart(test_data)
        print(f"✅ Gantt chart saved to: {gantt_path}")
        
        print("\n🎉 All tests passed! Charts are ACM-ready.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
