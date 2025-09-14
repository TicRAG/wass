#!/usr/bin/env python3
"""
评估指标测试脚本

演示如何使用新的评估指标系统进行系统性能评估。
"""
import sys
import os
import logging
import json
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation_metrics import ComprehensiveEvaluator
from src.evaluation_visualizer import EvaluationVisualizer
from src.evaluation_report_generator import EvaluationReportGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    生成示例数据用于测试评估指标系统
    
    Returns:
        包含各类指标历史记录的字典
    """
    np.random.seed(42)  # 确保结果可复现
    
    # 生成系统性能指标历史
    system_metrics_history = []
    for i in range(10):
        # 模拟性能改进趋势
        makespan = 100 - i * 3 + np.random.normal(0, 5)
        load_balance = 0.5 + i * 0.04 + np.random.normal(0, 0.05)
        resource_utilization = {
            'CPU': 0.6 + i * 0.03 + np.random.normal(0, 0.05),
            'Memory': 0.7 + i * 0.02 + np.random.normal(0, 0.05),
            'GPU': 0.5 + i * 0.04 + np.random.normal(0, 0.05),
            'Network': 0.4 + i * 0.03 + np.random.normal(0, 0.05)
        }
        throughput = 0.8 + i * 0.05 + np.random.normal(0, 0.05)
        energy_efficiency = 0.6 + i * 0.03 + np.random.normal(0, 0.05)
        cost_efficiency = 0.7 + i * 0.02 + np.random.normal(0, 0.05)
        fairness = 0.65 + i * 0.03 + np.random.normal(0, 0.05)
        reliability = 0.8 + i * 0.02 + np.random.normal(0, 0.05)
        
        system_metrics_history.append({
            'makespan': max(10, makespan),
            'load_balance': min(1, max(0, load_balance)),
            'resource_utilization': resource_utilization,
            'throughput': max(0, throughput),
            'energy_efficiency': min(1, max(0, energy_efficiency)),
            'cost_efficiency': min(1, max(0, cost_efficiency)),
            'fairness': min(1, max(0, fairness)),
            'reliability': min(1, max(0, reliability))
        })
    
    # 生成工作流调度质量指标历史
    workflow_metrics_history = []
    for i in range(10):
        scheduling_length = 80 - i * 2.5 + np.random.normal(0, 4)
        critical_path_ratio = 0.6 + i * 0.03 + np.random.normal(0, 0.05)
        parallel_efficiency = 0.5 + i * 0.04 + np.random.normal(0, 0.05)
        data_locality_score = 0.6 + i * 0.03 + np.random.normal(0, 0.05)
        communication_overhead = 20 - i * 1.5 + np.random.normal(0, 3)
        deadline_satisfaction_rate = 0.7 + i * 0.03 + np.random.normal(0, 0.05)
        priority_violation_rate = 0.3 - i * 0.02 + np.random.normal(0, 0.05)
        workflow_completion_rate = 0.75 + i * 0.02 + np.random.normal(0, 0.05)
        
        workflow_metrics_history.append({
            'scheduling_length': max(10, scheduling_length),
            'critical_path_ratio': min(1, max(0, critical_path_ratio)),
            'parallel_efficiency': min(1, max(0, parallel_efficiency)),
            'data_locality_score': min(1, max(0, data_locality_score)),
            'communication_overhead': max(0, communication_overhead),
            'deadline_satisfaction_rate': min(1, max(0, deadline_satisfaction_rate)),
            'priority_violation_rate': min(1, max(0, priority_violation_rate)),
            'workflow_completion_rate': min(1, max(0, workflow_completion_rate))
        })
    
    # 生成训练过程指标历史
    training_metrics_history = []
    for i in range(10):
        convergence_speed = 0.4 + i * 0.05 + np.random.normal(0, 0.05)
        stability_score = 0.5 + i * 0.04 + np.random.normal(0, 0.05)
        exploration_efficiency = 0.6 + i * 0.03 + np.random.normal(0, 0.05)
        
        # 生成学习曲线
        learning_curve = []
        for j in range(100):
            # 模拟学习曲线：初期波动大，后期趋于稳定
            base_value = -10 + j * 0.2
            noise = np.random.normal(0, 5 - j * 0.04)
            learning_curve.append(base_value + noise)
        
        # 生成损失曲线
        loss_curve = []
        for j in range(100):
            # 模拟损失曲线：初期高，后期降低
            base_value = 10 - j * 0.08
            noise = np.random.normal(0, 2 - j * 0.015)
            loss_curve.append(max(0, base_value + noise))
        
        # 生成Q值曲线
        q_value_curve = []
        for j in range(100):
            # 模拟Q值曲线：初期低，后期提高
            base_value = j * 0.15
            noise = np.random.normal(0, 3 - j * 0.02)
            q_value_curve.append(base_value + noise)
        
        training_metrics_history.append({
            'convergence_speed': min(1, max(0, convergence_speed)),
            'stability_score': min(1, max(0, stability_score)),
            'exploration_efficiency': min(1, max(0, exploration_efficiency)),
            'learning_curve': learning_curve,
            'loss_curve': loss_curve,
            'q_value_curve': q_value_curve
        })
    
    # 生成基准对比指标历史
    comparison_metrics_history = []
    for i in range(10):
        baseline_makespan = 120 - i * 2 + np.random.normal(0, 5)
        improvement_ratio = 0.1 + i * 0.05 + np.random.normal(0, 0.03)
        speedup_ratio = 1.1 + i * 0.08 + np.random.normal(0, 0.05)
        efficiency_ratio = 0.9 + i * 0.06 + np.random.normal(0, 0.05)
        scalability_score = 0.6 + i * 0.04 + np.random.normal(0, 0.05)
        robustness_score = 0.7 + i * 0.03 + np.random.normal(0, 0.05)
        adaptability_score = 0.65 + i * 0.04 + np.random.normal(0, 0.05)
        
        comparison_metrics_history.append({
            'baseline_makespan': max(20, baseline_makespan),
            'improvement_ratio': min(1, max(0, improvement_ratio)),
            'speedup_ratio': max(1, speedup_ratio),
            'efficiency_ratio': min(1, max(0, efficiency_ratio)),
            'scalability_score': min(1, max(0, scalability_score)),
            'robustness_score': min(1, max(0, robustness_score)),
            'adaptability_score': min(1, max(0, adaptability_score))
        })
    
    # 生成基准算法数据
    baseline_data = {
        'HEFT': {
            'makespan': 110,
            'description': '异构最早完成时间算法'
        },
        'CPOP': {
            'makespan': 115,
            'description': '关键路径算法'
        },
        'PEFT': {
            'makespan': 105,
            'description': '预测最早完成时间算法'
        }
    }
    
    return {
        'system_metrics_history': system_metrics_history,
        'workflow_metrics_history': workflow_metrics_history,
        'training_metrics_history': training_metrics_history,
        'comparison_metrics_history': comparison_metrics_history,
        'baseline_data': baseline_data
    }

def test_evaluation_metrics():
    """测试评估指标系统"""
    logger.info("开始测试评估指标系统...")
    
    # 生成示例数据
    sample_data = generate_sample_data()
    
    # 创建输出目录
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化评估器
    evaluator = ComprehensiveEvaluator()
    
    # 测试系统性能评估
    logger.info("测试系统性能评估...")
    system_metrics = evaluator.evaluate_system_performance(
        sample_data['system_metrics_history'][-1]
    )
    system_report = system_metrics.to_dict()
    logger.info(f"系统性能评分: {system_report.get('load_balance', 0):.3f}")
    
    # 测试工作流调度质量评估
    logger.info("测试工作流调度质量评估...")
    workflow_metrics = evaluator.evaluate_workflow_quality(
        sample_data['workflow_metrics_history'][-1]
    )
    workflow_report = workflow_metrics.to_dict()
    logger.info(f"工作流调度质量评分: {workflow_report.get('parallel_efficiency', 0):.3f}")
    
    # 测试训练过程评估
    logger.info("测试训练过程评估...")
    training_metrics = evaluator.evaluate_training_process(
        sample_data['training_metrics_history']
    )
    training_report = training_metrics.to_dict()
    logger.info(f"训练过程评分: {training_report.get('convergence_speed', 0):.3f}")
    
    # 测试基准对比评估
    logger.info("测试基准对比评估...")
    comparison_metrics = evaluator.compare_with_baselines(
        sample_data['comparison_metrics_history'][-1],
        sample_data['baseline_data']
    )
    comparison_report = comparison_metrics.to_dict()
    logger.info(f"基准对比评分: {comparison_report.get('improvement_ratio', 0):.3f}")
    
    # 测试综合评估
    logger.info("测试综合评估...")
    comprehensive_report = evaluator.generate_comprehensive_report(
        str(output_dir / "comprehensive_report.json")
    )
    logger.info(f"综合评分: {comprehensive_report.get('overall_score', 0):.3f}")
    
    # 保存评估报告
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
    logger.info(f"评估报告已保存到 {report_path}")
    
    return comprehensive_report

def test_visualization():
    """测试可视化功能"""
    logger.info("开始测试可视化功能...")
    
    # 生成示例数据
    sample_data = generate_sample_data()
    
    # 创建输出目录
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化可视化器
    visualizer = EvaluationVisualizer(str(output_dir))
    
    # 测试系统性能图表
    logger.info("生成系统性能图表...")
    system_chart_path = visualizer.plot_system_metrics(
        sample_data['system_metrics_history']
    )
    logger.info(f"系统性能图表已保存到 {system_chart_path}")
    
    # 测试工作流调度质量图表
    logger.info("生成工作流调度质量图表...")
    workflow_chart_path = visualizer.plot_workflow_metrics(
        sample_data['workflow_metrics_history']
    )
    logger.info(f"工作流调度质量图表已保存到 {workflow_chart_path}")
    
    # 测试训练过程图表
    logger.info("生成训练过程图表...")
    training_chart_path = visualizer.plot_training_metrics(
        sample_data['training_metrics_history']
    )
    logger.info(f"训练过程图表已保存到 {training_chart_path}")
    
    # 测试基准对比图表
    logger.info("生成基准对比图表...")
    comparison_chart_path = visualizer.plot_comparison_metrics(
        sample_data['comparison_metrics_history']
    )
    logger.info(f"基准对比图表已保存到 {comparison_chart_path}")
    
    # 测试交互式仪表板
    logger.info("生成交互式仪表板...")
    dashboard_path = visualizer.create_interactive_dashboard({
        'overall_score': 0.85,
        'system_performance': {
            'makespan': {'latest': 70},
            'load_balance': {'latest': 0.85},
            'resource_utilization': {'latest': {'CPU': 0.8, 'Memory': 0.75, 'GPU': 0.7, 'Network': 0.65}}
        },
        'workflow_quality': {
            'parallel_efficiency': {'latest': 0.85},
            'data_locality_score': {'latest': 0.8}
        },
        'training_process': {
            'learning_curve': {'latest': list(range(-10, 90, 2))}
        },
        'baseline_comparison': {
            'improvement_ratio': {'latest': 0.15}
        }
    })
    logger.info(f"交互式仪表板已保存到 {dashboard_path}")
    
    return {
        'system_chart_path': system_chart_path,
        'workflow_chart_path': workflow_chart_path,
        'training_chart_path': training_chart_path,
        'comparison_chart_path': comparison_chart_path,
        'dashboard_path': dashboard_path
    }

def test_report_generator():
    """测试报告生成器"""
    logger.info("开始测试报告生成器...")
    
    # 生成示例数据
    sample_data = generate_sample_data()
    
    # 创建输出目录
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化报告生成器
    report_generator = EvaluationReportGenerator(str(output_dir))
    
    # 生成综合报告
    logger.info("生成综合评估报告...")
    report_path = report_generator.generate_comprehensive_report(
        system_metrics_history=sample_data['system_metrics_history'],
        workflow_metrics_history=sample_data['workflow_metrics_history'],
        training_metrics_history=sample_data['training_metrics_history'],
        comparison_metrics_history=sample_data['comparison_metrics_history'],
        baseline_data=sample_data['baseline_data']
    )
    logger.info(f"综合评估报告已保存到 {report_path}")
    
    return report_path

def main():
    """主函数"""
    logger.info("开始评估指标系统测试...")
    
    # 测试评估指标
    comprehensive_report = test_evaluation_metrics()
    
    # 测试可视化
    visualization_results = test_visualization()
    
    # 测试报告生成器
    report_path = test_report_generator()
    
    # 输出测试结果摘要
    logger.info("=== 评估指标系统测试结果摘要 ===")
    logger.info(f"综合评分: {comprehensive_report.get('overall_score', 0):.3f}")
    logger.info(f"系统性能: {comprehensive_report.get('system_performance', {})}")
    logger.info(f"工作流调度质量: {comprehensive_report.get('workflow_quality', {})}")
    logger.info(f"训练过程: {comprehensive_report.get('training_process', {})}")
    logger.info(f"基准对比: {comprehensive_report.get('baseline_comparison', {})}")
    logger.info(f"生成的图表数量: {len([k for k, v in visualization_results.items() if v])}")
    logger.info(f"综合报告路径: {report_path}")
    
    logger.info("评估指标系统测试完成！")

if __name__ == "__main__":
    main()