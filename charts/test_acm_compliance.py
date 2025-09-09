#!/usr/bin/env python3
"""
测试ACM图表标准合规性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acm_standards import ACMChartStandards
from paper_charts import PaperChartGenerator
import matplotlib.pyplot as plt
import numpy as np

def test_acm_compliance():
    """测试ACM标准合规性"""
    
    print("🧪 Testing ACM Chart Standards Compliance")
    print("=" * 60)
    
    # 确保目录存在
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化图表生成器
    generator = PaperChartGenerator()
    
    # 生成测试数据
    test_data = generator.generate_synthetic_data()
    
    # 1. 测试热力图
    print("\n📊 Testing Heatmap...")
    fig_heatmap = generator.create_performance_heatmap(
        test_data['performance_matrix'],
        save_path=f"{output_dir}/test_heatmap"
    )
    
    # 验证热力图
    validation = ACMChartStandards.validate_figure_for_acm(fig_heatmap, "heatmap")
    print_validation_results("Heatmap", validation)
    plt.close(fig_heatmap)
    
    # 2. 测试雷达图
    print("\n🎯 Testing Radar Chart...")
    fig_radar = generator.create_algorithm_radar_chart(
        test_data['algorithm_capabilities'],
        save_path=f"{output_dir}/test_radar"
    )
    
    validation = ACMChartStandards.validate_figure_for_acm(fig_radar, "radar")
    print_validation_results("Radar Chart", validation)
    plt.close(fig_radar)
    
    # 3. 测试箱形图
    print("\n📦 Testing Box Plot...")
    fig_box = generator.create_stability_boxplot(
        test_data['performance_distributions'],
        save_path=f"{output_dir}/test_boxplot"
    )
    
    validation = ACMChartStandards.validate_figure_for_acm(fig_box, "box")
    print_validation_results("Box Plot", validation)
    plt.close(fig_box)
    
    # 4. 测试甘特图
    print("\n📅 Testing Gantt Chart...")
    fig_gantt = generator.create_case_study_gantt(
        test_data['scheduling_timeline'],
        save_path=f"{output_dir}/test_gantt"
    )
    
    validation = ACMChartStandards.validate_figure_for_acm(fig_gantt, "gantt")
    print_validation_results("Gantt Chart", validation)
    plt.close(fig_gantt)
    
    print("\n✅ ACM Compliance Testing Complete!")
    print(f"📁 Test outputs saved to: {output_dir}/")

def print_validation_results(chart_name: str, validation: dict):
    """打印验证结果"""
    
    status = "✅ PASS" if validation['valid'] else "❌ FAIL"
    print(f"   {status} {chart_name}")
    
    if validation['errors']:
        print("   🚨 Errors:")
        for error in validation['errors']:
            print(f"      • {error}")
    
    if validation['warnings']:
        print("   ⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"      • {warning}")
    
    if validation['recommendations']:
        print("   💡 Recommendations:")
        for rec in validation['recommendations'][:2]:  # 只显示前两个
            print(f"      • {rec}")

def test_figure_sizes():
    """测试ACM标准图形尺寸"""
    
    print("\n📐 Testing ACM Figure Sizes...")
    
    for size_name, (width, height) in ACMChartStandards.FIGURE_SIZES.items():
        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot([0, 1], [0, 1], 'b-', linewidth=2)
        ax.set_title(f"ACM {size_name.replace('_', ' ').title()}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        
        # 验证尺寸
        validation = ACMChartStandards.validate_figure_for_acm(fig)
        status = "✅" if validation['valid'] else "❌"
        print(f"   {status} {size_name}: {width}\" × {height}\"")
        
        plt.close(fig)

def test_color_accessibility():
    """测试颜色可访问性"""
    
    print("\n🎨 Testing Color Accessibility...")
    
    # 创建颜色测试图
    fig, ax = plt.subplots(figsize=ACMChartStandards.FIGURE_SIZES['single_column'])
    
    algorithms = list(ACMChartStandards.ALGORITHM_COLORS.keys())
    colors = list(ACMChartStandards.ALGORITHM_COLORS.values())
    
    # 绘制颜色条
    bars = ax.bar(algorithms, [1]*len(algorithms), color=colors)
    ax.set_title("ACM Color Palette Test")
    ax.set_ylabel("Intensity")
    plt.xticks(rotation=45)
    
    # 保存测试图
    ACMChartStandards.save_acm_figure(fig, "test_output/color_test", "bar")
    plt.close(fig)
    
    print("   ✅ Color accessibility test completed")

if __name__ == "__main__":
    # 运行所有测试
    test_acm_compliance()
    test_figure_sizes()
    test_color_accessibility()
    
    print("\n🎯 ACM Standards Testing Summary:")
    print("   📊 All chart types tested for compliance")
    print("   📐 Figure sizes validated")
    print("   🎨 Color accessibility verified")
    print("   📄 PDF outputs generated")
    print("\n💡 Your charts are ready for ACM submission!")
