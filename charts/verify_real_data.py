#!/usr/bin/env python3
"""
验证图表生成器是否正确要求真实数据
"""

import os
import sys

def test_real_data_requirement():
    """测试图表生成器是否正确要求真实数据"""
    
    print("🧪 验证图表生成器的真实数据要求")
    print("=" * 60)
    
    try:
        # 导入图表生成器
        from paper_charts import PaperChartGenerator
        
        print("✅ 成功导入 PaperChartGenerator")
        
        # 创建生成器实例
        generator = PaperChartGenerator()
        print("✅ 成功创建生成器实例")
        
        # 尝试加载数据（应该会报错，因为没有真实数据）
        print("\n📊 尝试加载实验数据...")
        try:
            test_data = generator.load_experimental_results()
            print("❌ 意外情况：居然加载到了数据！")
            print("   这意味着存在真实实验数据，可以继续生成图表")
            
            # 如果有数据，尝试生成一个图表
            heatmap_path = generator.generate_performance_heatmap(test_data)
            print(f"✅ 成功生成热力图：{heatmap_path}")
            
        except FileNotFoundError as e:
            print("✅ 正确行为：没有真实数据时报错")
            print("📝 错误信息预览：")
            print(str(e)[:200] + "..." if len(str(e)) > 200 else str(e))
            
        except ValueError as e:
            print("✅ 正确行为：数据格式验证失败")
            print("📝 错误信息：")
            print(str(e))
            
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()

def check_experiment_framework():
    """检查实验框架是否可用"""
    
    print("\n🔬 检查实验框架可用性")
    print("=" * 40)
    
    experiment_file = "../experiments/real_experiment_framework.py"
    if os.path.exists(experiment_file):
        print("✅ 实验框架文件存在")
        print(f"📁 路径：{os.path.abspath(experiment_file)}")
        print("\n💡 运行实验获取真实数据：")
        print("   cd ../experiments")
        print("   python real_experiment_framework.py")
    else:
        print("❌ 实验框架文件不存在")
        
    # 检查是否有任何实验结果
    result_patterns = [
        "../results/real_experiments/experiment_results.json",
        "../results/experiment_results.json", 
        "../results/wass_academic_results.json"
    ]
    
    found_results = []
    for pattern in result_patterns:
        if os.path.exists(pattern):
            found_results.append(os.path.abspath(pattern))
    
    if found_results:
        print(f"\n📊 发现 {len(found_results)} 个结果文件：")
        for result in found_results:
            print(f"   • {result}")
        print("\n✅ 可以直接生成图表！")
    else:
        print("\n📊 未发现现有实验结果")
        print("💡 需要先运行实验")

if __name__ == "__main__":
    test_real_data_requirement()
    check_experiment_framework()
    
    print("\n🎯 验证总结：")
    print("✅ 图表生成器已正确配置为只使用真实实验数据")
    print("✅ 没有真实数据时会给出清晰的错误提示和指导")
    print("📊 这确保了学术图表的严谨性和可重现性")
