#!/usr/bin/env python3
"""
修复paper_charts.py以适应当前实验数据
"""

def fix_paper_charts():
    """修复图表生成代码中的问题"""
    
    fixes = {
        'scheduler_list': "将 ['HEFT', 'WASS-DRL', 'WASS-RAG'] 改为 ['HEFT', 'SJF', 'WASS-RAG']",
        'data_format': "修复数据加载格式（直接列表而非{experiments: [...]}）",
        'field_mapping': "正确映射workflow_size字段",
        'error_handling': "添加缺失数据的错误处理"
    }
    
    print("🔧 建议的代码修复：")
    for fix_type, description in fixes.items():
        print(f"  • {fix_type}: {description}")
    
    print("\n⚠️  警告：即使修复代码，图表质量仍会受限于数据问题")
    print("   - 性能改善热力图将显示相同的值")
    print("   - 箱型图将没有变异性")
    print("   - 数据局部性是人工设定值")

if __name__ == "__main__":
    fix_paper_charts()
