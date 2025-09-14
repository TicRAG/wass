#!/usr/bin/env python3
"""
Simple test script for evaluation visualization system
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation_visualizer import EvaluationVisualizer

def test_visualization():
    """Test the evaluation visualization system"""
    print("Testing evaluation visualization system...")
    
    # Create output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = EvaluationVisualizer(str(output_dir))
    
    # Generate sample data for visualization
    system_metrics_history = []
    for i in range(10):
        makespan = 100 - i * 3 + np.random.normal(0, 5)
        load_balance = 0.5 + i * 0.04 + np.random.normal(0, 0.05)
        resource_utilization = {
            'CPU': 0.6 + i * 0.03 + np.random.normal(0, 0.05),
            'Memory': 0.7 + i * 0.02 + np.random.normal(0, 0.05),
            'GPU': 0.5 + i * 0.04 + np.random.normal(0, 0.05),
            'Network': 0.4 + i * 0.03 + np.random.normal(0, 0.05)
        }
        throughput = 0.8 + i * 0.05 + np.random.normal(0, 0.05)
        
        system_metrics_history.append({
            'makespan': max(10, makespan),
            'load_balance': min(1, max(0, load_balance)),
            'resource_utilization': resource_utilization,
            'throughput': max(0, throughput),
            'energy_efficiency': 0.7 + i * 0.02 + np.random.normal(0, 0.05),
            'cost_efficiency': 0.65 + i * 0.03 + np.random.normal(0, 0.05)
        })
    
    # Generate workflow metrics history
    workflow_metrics_history = []
    for i in range(10):
        scheduling_length = 80 - i * 2.5 + np.random.normal(0, 4)
        parallel_efficiency = 0.5 + i * 0.04 + np.random.normal(0, 0.05)
        data_locality_score = 0.6 + i * 0.03 + np.random.normal(0, 0.05)
        
        workflow_metrics_history.append({
            'scheduling_length': max(10, scheduling_length),
            'parallel_efficiency': min(1, max(0, parallel_efficiency)),
            'data_locality_score': min(1, max(0, data_locality_score)),
            'critical_path_ratio': 0.6 + i * 0.02 + np.random.normal(0, 0.05),
            'communication_overhead': 0.3 - i * 0.02 + np.random.normal(0, 0.05),
            'deadline_satisfaction_rate': 0.7 + i * 0.02 + np.random.normal(0, 0.05),
            'priority_violation_rate': 0.3 - i * 0.02 + np.random.normal(0, 0.05)
        })
    
    # Generate training metrics history
    training_metrics_history = []
    for i in range(10):
        learning_curve = []
        for j in range(100):
            base_value = -10 + j * 0.2
            noise = np.random.normal(0, 5 - j * 0.04)
            learning_curve.append(base_value + noise)
        
        loss_curve = []
        for j in range(100):
            base_value = 10 - j * 0.08
            noise = np.random.normal(0, 2 - j * 0.015)
            loss_curve.append(max(0, base_value + noise))
        
        training_metrics_history.append({
            'learning_curve': learning_curve,
            'loss_curve': loss_curve
        })
    
    # Generate comparison metrics history
    comparison_metrics_history = []
    for i in range(10):
        improvement_ratio = 0.1 + i * 0.05 + np.random.normal(0, 0.03)
        speedup_ratio = 1.1 + i * 0.08 + np.random.normal(0, 0.05)
        
        comparison_metrics_history.append({
            'improvement_ratio': min(1, max(0, improvement_ratio)),
            'speedup_ratio': max(1, speedup_ratio)
        })
    
    # Test system metrics visualization
    print("Testing system metrics visualization...")
    system_chart_path = visualizer.plot_system_metrics(system_metrics_history)
    print(f"System metrics chart saved to: {system_chart_path}")
    
    # Test workflow metrics visualization
    print("Testing workflow metrics visualization...")
    workflow_chart_path = visualizer.plot_workflow_metrics(workflow_metrics_history)
    print(f"Workflow metrics chart saved to: {workflow_chart_path}")
    
    # Test training metrics visualization
    print("Testing training metrics visualization...")
    training_chart_path = visualizer.plot_training_metrics(training_metrics_history)
    print(f"Training metrics chart saved to: {training_chart_path}")
    
    # Test comparison metrics visualization
    print("Testing comparison metrics visualization...")
    comparison_chart_path = visualizer.plot_comparison_metrics(comparison_metrics_history)
    print(f"Comparison metrics chart saved to: {comparison_chart_path}")
    
    # Test interactive dashboard
    print("Testing interactive dashboard...")
    dashboard_data = {
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
    }
    dashboard_path = visualizer.create_interactive_dashboard(dashboard_data)
    print(f"Interactive dashboard saved to: {dashboard_path}")
    
    # Test summary report
    print("Testing summary report...")
    report_data = {
        'timestamp': '2023-01-01T00:00:00',
        'overall_score': 0.85,
        'system_performance': {
            'makespan': {'latest': 70, 'mean': 75, 'trend': 'improving'},
            'load_balance': {'latest': 0.85, 'mean': 0.8, 'trend': 'improving'},
            'resource_utilization': {'latest': {'CPU': 0.8, 'Memory': 0.75}}
        },
        'workflow_quality': {
            'parallel_efficiency': {'latest': 0.85, 'mean': 0.8, 'trend': 'improving'},
            'data_locality_score': {'latest': 0.8, 'mean': 0.75, 'trend': 'stable'}
        },
        'training_process': {
            'convergence_speed': {'latest': 0.9, 'mean': 0.85, 'trend': 'improving'},
            'stability_score': {'latest': 0.8, 'mean': 0.75, 'trend': 'stable'}
        },
        'baseline_comparison': {
            'improvement_ratio': {'latest': 0.15, 'mean': 0.12, 'trend': 'improving'},
            'speedup_ratio': {'latest': 1.2, 'mean': 1.15, 'trend': 'stable'}
        },
        'recommendations': [
            'Continue current scheduling strategy',
            'Optimize resource allocation for better load balance',
            'Improve data locality to reduce communication overhead'
        ]
    }
    report_path = visualizer.generate_summary_report(report_data, str(output_dir / "summary_report.html"))
    print(f"Summary report saved to: {report_path}")
    
    print("Evaluation visualization system test completed successfully!")
    
    return {
        'system_chart_path': system_chart_path,
        'workflow_chart_path': workflow_chart_path,
        'training_chart_path': training_chart_path,
        'comparison_chart_path': comparison_chart_path,
        'dashboard_path': dashboard_path,
        'report_path': report_path
    }

if __name__ == "__main__":
    test_visualization()