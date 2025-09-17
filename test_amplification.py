#!/usr/bin/env python3
"""
测试时间差异放大效果 - 方案A: 缩小算力 + 增加负载
"""

import sys
from pathlib import Path
import yaml
import json
import subprocess
import re

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

def create_amplified_config():
    """创建放大实验配置"""
    
    config = {
        'drl_model_path': 'models/improved_wass_drl.pth',
        'enabled_schedulers': ['FIFO', 'HEFT'],
        'output_dir': 'results/amplification_test',
        'platform_file': 'test_platform_reduced.xml',  # 低算力平台
        'rag_config_path': 'configs/rag.yaml',
        'repetitions': 1,
        'workflow_dir': 'workflows',
        'workflow_sizes': [5]  # 使用放大工作流
    }
    
    config_path = "/data/workspace/traespace/wass_trae/configs/amplification_test.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"创建放大实验配置: {config_path}")
    return config_path

def extract_makespan_from_output(output_text):
    """从输出中提取makespan信息"""
    
    # 查找makespan相关的输出
    makespan_pattern = r'makespan.*?([\d.]+)'
    time_pattern = r'时间.*?([\d.]+)'
    
    matches = re.findall(makespan_pattern, output_text, re.IGNORECASE)
    if matches:
        return float(matches[-1])  # 取最后一个匹配
    
    # 尝试其他模式
    matches = re.findall(time_pattern, output_text, re.IGNORECASE)
    if matches:
        return float(matches[-1])
    
    return None

def run_amplification_test():
    """运行放大测试"""
    
    print("=" * 80)
    print("时间差异放大测试 - 方案A")
    print("=" * 80)
    
    # 创建配置
    config_path = create_amplified_config()
    
    print(f"\n测试配置:")
    print(f"- 平台: test_platform_reduced.xml (降低算力)")
    print(f"- 工作流: compute_intensive_amplified.json (增加负载)")
    print(f"- 调度器: FIFO vs HEFT")
    
    # 运行实验
    print(f"\n运行放大实验...")
    
    try:
        result = subprocess.run([
            sys.executable, "experiments/wrench_real_experiment.py"
        ], capture_output=True, text=True, cwd="/data/workspace/traespace/wass_trae", timeout=300)
        
        print(f"实验完成，返回码: {result.returncode}")
        
        if result.stdout:
            print(f"\n标准输出前1000字符:")
            print(result.stdout[:1000])
            
            # 提取makespan信息
            makespan = extract_makespan_from_output(result.stdout)
            if makespan:
                print(f"\n提取到的makespan: {makespan}")
        
        if result.stderr:
            print(f"\n错误输出前500字符:")
            print(result.stderr[:500])
        
        # 分析调度器输出
        print(f"\n调度器行为分析:")
        
        # 提取FIFO调度信息
        fifo_matches = re.findall(r'FIFO调度任务.*?(?:固定分配到|ComputeHost\d+)', result.stdout)
        if fifo_matches:
            print(f"FIFO分配了 {len(fifo_matches)} 个任务")
        
        # 提取HEFT调度信息  
        heft_matches = re.findall(r'HEFT调度任务.*?(?:选择主机|ComputeHost\d+)', result.stdout)
        if heft_matches:
            print(f"HEFT分配了 {len(heft_matches)} 个任务")
        
        # 提取主机选择信息
        host_assignments = re.findall(r'(ComputeHost\d+)', result.stdout)
        if host_assignments:
            from collections import Counter
            host_counts = Counter(host_assignments)
            print(f"\n主机分配统计:")
            for host, count in host_counts.items():
                print(f"  {host}: {count} 次")
        
        return result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print("实验超时 (5分钟)")
        return None, "实验超时"
    except Exception as e:
        print(f"运行实验时出错: {e}")
        return None, str(e)

def analyze_platform_differences():
    """分析平台配置差异"""
    
    print(f"\n" + "=" * 80)
    print("平台配置对比分析")
    print("=" * 80)
    
    # 读取原始平台配置
    original_file = "/data/workspace/traespace/wass_trae/test_platform.xml"
    reduced_file = "/data/workspace/traespace/wass_trae/test_platform_reduced.xml"
    
    try:
        with open(original_file, 'r') as f:
            original_content = f.read()
        
        with open(reduced_file, 'r') as f:
            reduced_content = f.read()
        
        # 提取速度信息
        import re
        original_speeds = re.findall(r'<host id="(ComputeHost\d+)" speed="([\d.]+Gf)"', original_content)
        reduced_speeds = re.findall(r'<host id="(ComputeHost\d+)" speed="([\d.]+Gf)"', reduced_content)
        
        print(f"\n主机速度对比:")
        print(f"{'主机':<15} {'原始速度':<10} {'降低后速度':<12} {'降低倍数':<10}")
        print("-" * 50)
        
        for (host1, orig_speed), (host2, red_speed) in zip(original_speeds, reduced_speeds):
            if host1 == host2:
                orig_val = float(orig_speed.replace('Gf', ''))
                red_val = float(red_speed.replace('Gf', ''))
                reduction_factor = orig_val / red_val
                print(f"{host1:<15} {orig_speed:<10} {red_speed:<12} {reduction_factor:.1f}x")
        
        print(f"\n预期效果:")
        print(f"- 任务执行时间增加 2-4 倍")
        print(f"- 主机间性能差异更明显")
        print(f"- HEFT的智能选择优势更突出")
        
    except Exception as e:
        print(f"分析平台配置时出错: {e}")

def analyze_workflow_differences():
    """分析工作流负载差异"""
    
    print(f"\n" + "=" * 80)
    print("工作流负载对比分析") 
    print("=" * 80)
    
    # 这里我们假设原始工作流的runtime较小
    print(f"\n工作流负载放大策略:")
    print(f"- 生成任务 runtime: 1000 (大幅放大)")
    print(f"- 处理任务 runtime: 5000 (大幅放大)")
    print(f"- 聚合任务 runtime: 3000 (大幅放大)")
    
    print(f"\n预期效果:")
    print(f"- 总执行时间增加 10-50 倍")
    print(f"- 调度策略的影响更显著")
    print(f"- 更容易观察到性能差异")

if __name__ == "__main__":
    print("时间差异放大测试")
    print("测试目标: 通过降低平台算力和增加工作负载来放大FIFO和HEFT的时间差异")
    
    # 运行放大测试
    stdout, stderr = run_amplification_test()
    
    # 分析配置差异
    analyze_platform_differences()
    analyze_workflow_differences()
    
    print(f"\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if stdout:
        print("✓ 实验运行成功")
        print("✓ 需要检查输出目录中的详细结果")
        print("✓ 对比makespan差异是否明显放大")
    else:
        print("✗ 实验运行失败")
        print("✗ 需要检查错误信息和配置")
    
    print(f"\n后续步骤:")
    print("1. 检查 results/amplification_test/ 目录中的结果")
    print("2. 对比原始配置和放大配置的makespan")
    print("3. 如果差异仍不够明显，尝试方案C（高方差+重负载）")