#!/usr/bin/env python3
"""
WRENCH测试运行脚本

这个脚本用于在有WRENCH环境的测试机器上运行所有测试。

使用方法:
    python run_wrench_tests.py --all
    python run_wrench_tests.py --basic
    python run_wrench_tests.py --integration
    python run_wrench_tests.py --performance

Author: WASS-RAG Team  
Date: 2024-12
"""

import argparse
import sys
import os
import traceback
from pathlib import Path
from datetime import datetime
import json

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_wrench_environment():
    """检查WRENCH环境是否正确设置"""
    print("🔍 检查WRENCH环境...")
    
    # 检查WRENCH导入
    try:
        import wrench
        print(f"✅ WRENCH {wrench.__version__} 可用")
        return True
    except ImportError as e:
        print(f"❌ WRENCH导入失败: {e}")
        print("请确保WRENCH已正确安装并且Python绑定可用")
        return False

def run_basic_tests():
    """运行基础测试"""
    print("\n" + "="*50)
    print("🧪 运行基础测试")
    print("="*50)
    
    success_count = 0
    total_count = 0
    
    # 测试1: WRENCH模块测试
    total_count += 1
    print(f"\n📋 测试 {total_count}: WRENCH模块基础功能")
    try:
        from wrench_integration.simulator import test_wrench_integration
        if test_wrench_integration():
            success_count += 1
            print("✅ WRENCH模块测试通过")
        else:
            print("❌ WRENCH模块测试失败")
    except Exception as e:
        print(f"❌ WRENCH模块测试异常: {e}")
        traceback.print_exc()
    
    # 测试2: 平台创建测试
    total_count += 1
    print(f"\n📋 测试 {total_count}: 平台创建功能")
    try:
        from wrench_integration.simulator import WRENCHSimulator
        
        simulator = WRENCHSimulator()
        platform_config = {
            'hosts': [
                {'id': 'test_node', 'speed': '1Gf', 'cores': 2}
            ],
            'links': [],
            'routes': []
        }
        
        platform_file = simulator.create_platform(platform_config)
        if os.path.exists(platform_file):
            success_count += 1
            print(f"✅ 平台文件创建成功: {platform_file}")
        else:
            print("❌ 平台文件创建失败")
            
    except Exception as e:
        print(f"❌ 平台创建测试异常: {e}")
        traceback.print_exc()
    
    # 测试3: 工作流创建测试
    total_count += 1
    print(f"\n📋 测试 {total_count}: 工作流创建功能")
    try:
        workflow_spec = {
            'name': 'test_workflow',
            'tasks': [
                {
                    'id': 'task1',
                    'flops': 1e9,
                    'bytes_read': 1e6,
                    'bytes_written': 1e6,
                    'dependencies': []
                }
            ]
        }
        
        workflow_id = simulator.create_workflow(workflow_spec)
        if workflow_id == 'test_workflow':
            success_count += 1
            print(f"✅ 工作流创建成功: {workflow_id}")
        else:
            print("❌ 工作流创建失败")
            
    except Exception as e:
        print(f"❌ 工作流创建测试异常: {e}")
        traceback.print_exc()
    
    print(f"\n📊 基础测试结果: {success_count}/{total_count} 通过")
    return success_count, total_count

def run_integration_tests():
    """运行集成测试"""
    print("\n" + "="*50)
    print("🔧 运行集成测试")
    print("="*50)
    
    success_count = 0
    total_count = 0
    
    # 测试1: 基础仿真实验
    total_count += 1
    print(f"\n📋 测试 {total_count}: 基础仿真实验")
    try:
        from experiments.basic_simulation import run_basic_simulation, get_default_config
        
        config = get_default_config()
        # 简化配置以加快测试
        config['workflows'] = config['workflows'][:1]  # 只测试第一个工作流
        
        output_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        results = run_basic_simulation(config, output_dir)
        
        if results and 'workflows' in results:
            success_count += 1
            print(f"✅ 基础仿真实验成功，结果保存到: {output_dir}")
        else:
            print("❌ 基础仿真实验失败")
            
    except Exception as e:
        print(f"❌ 基础仿真实验异常: {e}")
        traceback.print_exc()
    
    # 测试2: WRENCH直接接口测试
    total_count += 1
    print(f"\n📋 测试 {total_count}: WRENCH直接接口")
    try:
        import wrench
        
        # 创建仿真
        simulation = wrench.Simulation()
        print("✅ WRENCH仿真对象创建成功")
        
        # 测试平台加载（需要实际的平台文件）
        test_platform = create_test_platform_file()
        simulation.add_platform(test_platform)
        print("✅ 平台加载成功")
        
        success_count += 1
        
    except Exception as e:
        print(f"❌ WRENCH直接接口测试异常: {e}")
        traceback.print_exc()
    
    print(f"\n📊 集成测试结果: {success_count}/{total_count} 通过")
    return success_count, total_count

def create_test_platform_file():
    """创建测试用的平台文件"""
    platform_xml = '''<?xml version="1.0"?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="test_host" speed="1Gf" core="1"/>
  </zone>
</platform>'''
    
    platform_file = "test_platform.xml"
    with open(platform_file, 'w') as f:
        f.write(platform_xml)
    
    return platform_file

def run_performance_tests():
    """运行性能测试"""
    print("\n" + "="*50)
    print("⚡ 运行性能测试")
    print("="*50)
    
    success_count = 0
    total_count = 0
    
    # 测试1: 大工作流仿真
    total_count += 1
    print(f"\n📋 测试 {total_count}: 大规模工作流仿真")
    try:
        from wrench_integration.simulator import WRENCHSimulator
        
        # 创建大工作流
        large_workflow = {
            'name': 'large_test_workflow',
            'tasks': []
        }
        
        # 生成100个任务的工作流
        for i in range(100):
            task = {
                'id': f'task_{i}',
                'flops': 1e8,  # 100 MFlops
                'bytes_read': 1e5,  # 100 KB
                'bytes_written': 1e5,  # 100 KB
                'dependencies': [f'task_{i-1}'] if i > 0 else []
            }
            large_workflow['tasks'].append(task)
        
        simulator = WRENCHSimulator()
        workflow_id = simulator.create_workflow(large_workflow)
        
        print(f"✅ 大工作流创建成功: {len(large_workflow['tasks'])} 个任务")
        success_count += 1
        
    except Exception as e:
        print(f"❌ 大工作流测试异常: {e}")
        traceback.print_exc()
    
    print(f"\n📊 性能测试结果: {success_count}/{total_count} 通过")
    return success_count, total_count

def generate_test_report(basic_results, integration_results, performance_results):
    """生成测试报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'environment': {
            'python_version': sys.version,
            'platform': sys.platform,
        },
        'test_results': {
            'basic_tests': {
                'passed': basic_results[0],
                'total': basic_results[1],
                'success_rate': basic_results[0] / basic_results[1] if basic_results[1] > 0 else 0
            },
            'integration_tests': {
                'passed': integration_results[0],
                'total': integration_results[1],
                'success_rate': integration_results[0] / integration_results[1] if integration_results[1] > 0 else 0
            },
            'performance_tests': {
                'passed': performance_results[0],
                'total': performance_results[1],
                'success_rate': performance_results[0] / performance_results[1] if performance_results[1] > 0 else 0
            }
        }
    }
    
    # 添加WRENCH版本信息
    try:
        import wrench
        report['environment']['wrench_version'] = wrench.__version__
    except:
        report['environment']['wrench_version'] = 'Unknown'
    
    total_passed = basic_results[0] + integration_results[0] + performance_results[0]
    total_tests = basic_results[1] + integration_results[1] + performance_results[1]
    report['overall'] = {
        'passed': total_passed,
        'total': total_tests,
        'success_rate': total_passed / total_tests if total_tests > 0 else 0
    }
    
    # 保存报告
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 测试报告已保存: {report_file}")
    return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WRENCH测试运行器")
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--basic', action='store_true', help='运行基础测试')
    parser.add_argument('--integration', action='store_true', help='运行集成测试')
    parser.add_argument('--performance', action='store_true', help='运行性能测试')
    
    args = parser.parse_args()
    
    if not any([args.all, args.basic, args.integration, args.performance]):
        print("请指定要运行的测试类型")
        parser.print_help()
        return 1
    
    print("🚀 WASS-RAG WRENCH测试开始")
    print("="*60)
    
    # 检查环境
    if not check_wrench_environment():
        print("❌ WRENCH环境检查失败，无法继续测试")
        return 1
    
    basic_results = (0, 0)
    integration_results = (0, 0)
    performance_results = (0, 0)
    
    # 运行测试
    if args.all or args.basic:
        basic_results = run_basic_tests()
    
    if args.all or args.integration:
        integration_results = run_integration_tests()
    
    if args.all or args.performance:
        performance_results = run_performance_tests()
    
    # 生成报告
    report = generate_test_report(basic_results, integration_results, performance_results)
    
    # 打印总结
    print("\n" + "="*60)
    print("📋 测试总结")
    print("="*60)
    
    total_passed = report['overall']['passed']
    total_tests = report['overall']['total']
    success_rate = report['overall']['success_rate']
    
    print(f"总体结果: {total_passed}/{total_tests} 通过 ({success_rate:.1%})")
    
    if args.all or args.basic:
        basic = report['test_results']['basic_tests']
        print(f"基础测试: {basic['passed']}/{basic['total']} 通过 ({basic['success_rate']:.1%})")
    
    if args.all or args.integration:
        integration = report['test_results']['integration_tests']
        print(f"集成测试: {integration['passed']}/{integration['total']} 通过 ({integration['success_rate']:.1%})")
    
    if args.all or args.performance:
        performance = report['test_results']['performance_tests']
        print(f"性能测试: {performance['passed']}/{performance['total']} 通过 ({performance['success_rate']:.1%})")
    
    if success_rate >= 0.8:
        print("\n🎉 测试结果良好！")
        return 0
    elif success_rate >= 0.5:
        print("\n⚠️  测试结果一般，需要修复一些问题")
        return 1
    else:
        print("\n❌ 测试结果较差，需要重点修复")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
