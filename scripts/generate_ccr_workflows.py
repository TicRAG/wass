#!/usr/bin/env python3
"""
生成不同CCR值的工作流用于实验验证
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.workflow_generator import WorkflowGenerator

def generate_ccr_workflows():
    """生成不同CCR值的工作流"""
    
    # 定义CCR测试值
    ccr_values = [0.1, 1.0, 5.0, 10.0]  # 计算密集型、均衡型、通信密集型
    
    # 定义测试规模
    test_sizes = [50, 100, 200]
    
    # 定义工作流模式
    patterns = ['montage', 'ligo', 'cybershake']
    
    output_base = "data/workflows_ccr"
    
    for ccr in ccr_values:
        print(f"\n🎯 生成CCR={ccr}的工作流...")
        
        # 为每个CCR值创建子目录
        ccr_dir = f"{output_base}/ccr_{ccr}"
        generator = WorkflowGenerator(ccr_dir, ccr)
        
        for pattern in patterns:
            print(f"  📊 生成{pattern}模式...")
            
            # 生成测试规模的工作流
            files = generator.generate_workflow_set(pattern, test_sizes)
            
            for file_path in files:
                # 重命名文件以包含CCR信息
                old_name = Path(file_path).name
                new_name = old_name.replace('.json', f'_ccr{ccr}.json')
                new_path = Path(file_path).parent / new_name
                
                if Path(file_path).exists():
                    Path(file_path).rename(new_path)
                    print(f"    ✅ {new_name}")
    
    print(f"\n🎉 所有CCR工作流已生成在: {output_base}")
    print("📝 使用方式:")
    print("  python scripts/generate_ccr_workflows.py")
    print("  python run_complete_experiment.sh --workflow-dir data/workflows_ccr")

if __name__ == "__main__":
    generate_ccr_workflows()