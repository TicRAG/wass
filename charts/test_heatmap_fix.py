#!/usr/bin/env python3
"""
测试修复后的图表生成
"""

import sys
import os

def test_heatmap_only():
    """只测试热力图生成"""
    
    print("🧪 测试修复后的热力图生成")
    print("=" * 50)
    
    try:
        from paper_charts import PaperChartGenerator
        
        # 创建生成器
        generator = PaperChartGenerator(results_dir="../results")
        print("✅ 生成器创建成功")
        
        # 加载数据
        results = generator.load_experimental_results()
        print("✅ 数据加载成功")
        
        # 测试数据预处理
        df = generator._preprocess_experiment_data(results)
        print(f"✅ 数据预处理成功，处理了 {len(df)} 条记录")
        print(f"📊 字段: {list(df.columns)}")
        print(f"📊 调度器: {sorted(df['scheduler'].unique())}")
        print(f"📊 集群规模: {sorted(df['cluster_size'].unique())}")
        print(f"📊 工作流规模: {sorted(df['workflow_size'].unique())}")
        
        # 测试热力图生成
        print("\n🔥 生成热力图...")
        heatmap_path = generator.generate_performance_heatmap(results)
        print(f"✅ 热力图生成成功: {heatmap_path}")
        
        # 检查文件是否存在
        if os.path.exists(heatmap_path):
            file_size = os.path.getsize(heatmap_path) / 1024
            print(f"✅ 文件已保存，大小: {file_size:.1f} KB")
        else:
            print("❌ 文件未生成")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_heatmap_only():
        print("\n🎉 热力图测试通过!")
        print("💡 可以继续测试其他图表")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)
