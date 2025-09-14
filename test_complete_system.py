#!/usr/bin/env python3
"""
Complete test script for the evaluation metrics and visualization system
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation_metrics import ComprehensiveEvaluator
from src.evaluation_visualizer import EvaluationVisualizer

def test_complete_evaluation_system():
    """Test the complete evaluation metrics and visualization system"""
    print("Testing complete evaluation metrics and visualization system...")
    
    # Create output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator and visualizer
    evaluator = ComprehensiveEvaluator()
    visualizer = EvaluationVisualizer(str(output_dir))
    
    # Generate sample data for evaluation
    print("Generating sample data for evaluation...")
    
    # System performance data
    makespan = 70.0
    avg_waiting_time = 15.0
    resource_utilization = {
        'CPU': 0.8,
        'Memory': 0.75,
        'GPU': 0.7,
        'Network': 0.65
    }
    load_balance = 0.85
    throughput = 0.9
    energy_efficiency = 0.75
    cost_efficiency = 0.7
    fairness = 0.8
    
    # Workflow scheduling quality data
    scheduling_length = 60.0
    parallel_efficiency = 0.85
    data_locality_score = 0.8
    communication_overhead = 0.25
    deadline_satisfaction_rate = 0.85
    workflow_completion_rate = 0.9
    priority_violation_rate = 0.15
    critical_path_ratio = 0.65
    
    # Training process data
    episode_count = 10
    learning_curve = []
    loss_curve = []
    reward_curve = []
    q_value_curve = []
    
    for i in range(episode_count):
        learning_curve.append(-10 + i * 2 + np.random.normal(0, 1))
        loss_curve.append(10 - i * 0.8 + np.random.normal(0, 0.5))
        reward_curve.append(-5 + i * 1.5 + np.random.normal(0, 1))
        q_value_curve.append(-8 + i * 1.8 + np.random.normal(0, 0.8))
    
    # Baseline comparison data
    baseline_makespan = 85.0
    baseline_throughput = 0.75
    baseline_energy_efficiency = 0.65
    baseline_cost_efficiency = 0.6
    
    # Test system performance evaluation
    print("Testing system performance evaluation...")
    simulation_result = {
        'makespan': makespan,
        'avg_waiting_time': avg_waiting_time,
        'avg_turnaround_time': avg_waiting_time + makespan / 2,  # Estimate turnaround time
        'total_tasks': 100,  # Assume 100 tasks
        'resource_utilization': resource_utilization,
        'node_loads': [0.8, 0.75, 0.7, 0.65],  # Simulate node loads
        'task_completion_rate': 0.95
    }
    system_performance = evaluator.evaluate_system_performance(simulation_result)
    
    # Test workflow scheduling quality evaluation
    print("Testing workflow scheduling quality evaluation...")
    workflow_result = {
        'scheduling_length': scheduling_length,
        'critical_path_ratio': critical_path_ratio,
        'data_locality_score': data_locality_score,
        'communication_overhead': communication_overhead,
        'deadline_satisfaction_rate': deadline_satisfaction_rate,
        'priority_violation_rate': priority_violation_rate,
        'workflow_completion_rate': workflow_completion_rate,
        'total_tasks': 100,
        'parallel_tasks': 80,
        'sequential_tasks': 20
    }
    workflow_quality = evaluator.evaluate_workflow_quality(workflow_result)
    
    # Test training process evaluation
    print("Testing training process evaluation...")
    training_history = []
    for i in range(episode_count):
        training_history.append({
            'total_reward': reward_curve[i] if i < len(reward_curve) else -10 + i * 1.5,
            'avg_loss': loss_curve[i] if i < len(loss_curve) else 10 - i * 0.8,
            'avg_q_value': q_value_curve[i] if i < len(q_value_curve) else -8 + i * 1.8
        })
    training_process = evaluator.evaluate_training_process(training_history)
    
    # Test baseline comparison
    print("Testing baseline comparison...")
    our_result = {
        'makespan': makespan,
        'throughput': throughput,
        'energy_efficiency': energy_efficiency,
        'cost_efficiency': cost_efficiency,
        'efficiency': 0.85  # Assume overall efficiency
    }
    baseline_results = {
        'HEFT': {
            'makespan': baseline_makespan,
            'throughput': baseline_throughput,
            'energy_efficiency': baseline_energy_efficiency,
            'cost_efficiency': baseline_cost_efficiency,
            'efficiency': 0.75  # Assume baseline efficiency
        },
        'CPOP': {
            'makespan': baseline_makespan * 1.1,
            'throughput': baseline_throughput * 0.9,
            'energy_efficiency': baseline_energy_efficiency * 0.95,
            'cost_efficiency': baseline_cost_efficiency * 0.9,
            'efficiency': 0.7
        }
    }
    baseline_comparison = evaluator.compare_with_baselines(our_result, baseline_results)
    
    # Generate comprehensive report
    print("Generating comprehensive report...")
    report_path = os.path.join(output_dir, "comprehensive_report.json")
    comprehensive_report = evaluator.generate_comprehensive_report(report_path)
    
    # Save evaluation results
    print("Saving evaluation results...")
    results = {
        'system_performance': system_performance.to_dict(),
        'workflow_quality': workflow_quality.to_dict(),
        'training_process': training_process.to_dict(),
        'baseline_comparison': baseline_comparison.to_dict(),
        'comprehensive_report': comprehensive_report
    }
    
    with open(os.path.join(output_dir, "complete_evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
      
    print("Evaluation results saved to:", os.path.join(output_dir, "complete_evaluation_results.json"))
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # System metrics visualization
    system_metrics_history = []
    for i in range(10):
        system_metrics_history.append({
            'makespan': makespan - i * 2 + np.random.normal(0, 2),
            'load_balance': min(1, max(0, load_balance + i * 0.01 + np.random.normal(0, 0.02))),
            'resource_utilization': {
                'CPU': min(1, max(0, resource_utilization['CPU'] + i * 0.01 + np.random.normal(0, 0.02))),
                'Memory': min(1, max(0, resource_utilization['Memory'] + i * 0.01 + np.random.normal(0, 0.02))),
                'GPU': min(1, max(0, resource_utilization['GPU'] + i * 0.01 + np.random.normal(0, 0.02))),
                'Network': min(1, max(0, resource_utilization['Network'] + i * 0.01 + np.random.normal(0, 0.02)))
            },
            'throughput': min(1, max(0, throughput + i * 0.01 + np.random.normal(0, 0.02))),
            'energy_efficiency': min(1, max(0, energy_efficiency + i * 0.01 + np.random.normal(0, 0.02))),
            'cost_efficiency': min(1, max(0, cost_efficiency + i * 0.01 + np.random.normal(0, 0.02)))
        })
    
    system_chart_path = visualizer.plot_system_metrics(system_metrics_history)
    print(f"System metrics chart saved to: {system_chart_path}")
    
    # Workflow metrics visualization
    workflow_metrics_history = []
    for i in range(10):
        workflow_metrics_history.append({
            'scheduling_length': scheduling_length - i * 1.5 + np.random.normal(0, 1.5),
            'parallel_efficiency': min(1, max(0, parallel_efficiency + i * 0.01 + np.random.normal(0, 0.02))),
            'data_locality_score': min(1, max(0, data_locality_score + i * 0.01 + np.random.normal(0, 0.02))),
            'critical_path_ratio': min(1, max(0, critical_path_ratio + i * 0.005 + np.random.normal(0, 0.01))),
            'communication_overhead': min(1, max(0, communication_overhead - i * 0.01 + np.random.normal(0, 0.02))),
            'deadline_satisfaction_rate': min(1, max(0, deadline_satisfaction_rate + i * 0.01 + np.random.normal(0, 0.02))),
            'priority_violation_rate': min(1, max(0, priority_violation_rate - i * 0.01 + np.random.normal(0, 0.02)))
        })
    
    workflow_chart_path = visualizer.plot_workflow_metrics(workflow_metrics_history)
    print(f"Workflow metrics chart saved to: {workflow_chart_path}")
    
    # Training metrics visualization
    training_metrics_history = []
    for i in range(10):
        episode_learning_curve = []
        episode_loss_curve = []
        for j in range(100):
            base_value = -10 + j * 0.2
            noise = np.random.normal(0, 5 - j * 0.04)
            episode_learning_curve.append(base_value + noise)
            
            base_value = 10 - j * 0.08
            noise = np.random.normal(0, 2 - j * 0.015)
            episode_loss_curve.append(max(0, base_value + noise))
        
        training_metrics_history.append({
            'learning_curve': episode_learning_curve,
            'loss_curve': episode_loss_curve
        })
    
    training_chart_path = visualizer.plot_training_metrics(training_metrics_history)
    print(f"Training metrics chart saved to: {training_chart_path}")
    
    # Comparison metrics visualization
    comparison_metrics_history = []
    for i in range(10):
        comparison_metrics_history.append({
            'improvement_ratio': min(1, max(0, baseline_comparison.improvement_ratio + i * 0.005 + np.random.normal(0, 0.01))),
            'speedup_ratio': max(1, baseline_comparison.speedup_ratio + i * 0.01 + np.random.normal(0, 0.02))
        })
    
    comparison_chart_path = visualizer.plot_comparison_metrics(comparison_metrics_history)
    print(f"Comparison metrics chart saved to: {comparison_chart_path}")
    
    # Create interactive dashboard
    dashboard_data = {
        'overall_score': comprehensive_report['overall_score'],
        'system_performance': {
            'load_balance': {'latest': system_performance.load_balance},
            'resource_utilization': {'latest': system_performance.resource_utilization},
            'throughput': {'latest': system_performance.throughput},
            'energy_efficiency': {'latest': system_performance.energy_efficiency},
            'cost_efficiency': {'latest': system_performance.cost_efficiency}
        },
        'workflow_quality': {
            'parallel_efficiency': {'latest': workflow_quality.parallel_efficiency},
            'data_locality_score': {'latest': workflow_quality.data_locality_score},
            'scheduling_length': {'latest': workflow_quality.scheduling_length},
            'critical_path_ratio': {'latest': workflow_quality.critical_path_ratio},
            'communication_overhead': {'latest': workflow_quality.communication_overhead},
            'deadline_satisfaction_rate': {'latest': workflow_quality.deadline_satisfaction_rate},
            'priority_violation_rate': {'latest': workflow_quality.priority_violation_rate},
            'workflow_completion_rate': {'latest': workflow_quality.workflow_completion_rate}
        },
        'training_process': {
            'learning_curve': {'latest': training_process.learning_curve},
            'loss_curve': {'latest': training_process.loss_curve},
            'reward_distribution': {'latest': training_process.reward_distribution},
            'q_value_curve': {'latest': training_process.q_value_curve}
        },
        'baseline_comparison': {
            'improvement_ratio': {'latest': baseline_comparison.improvement_ratio},
            'speedup_ratio': {'latest': baseline_comparison.speedup_ratio},
            'efficiency_ratio': {'latest': baseline_comparison.efficiency_ratio},
            'scalability_score': {'latest': baseline_comparison.scalability_score},
            'robustness_score': {'latest': baseline_comparison.robustness_score},
            'adaptability_score': {'latest': baseline_comparison.adaptability_score}
        }
    }
    dashboard_path = visualizer.create_interactive_dashboard(dashboard_data)
    print(f"Interactive dashboard saved to: {dashboard_path}")
    
    # Generate summary report
    report_data = comprehensive_report
    report_path = visualizer.generate_summary_report(report_data, os.path.join(output_dir, "complete_summary_report.html"))
    print(f"Summary report saved to: {report_path}")
    
    print("\n=== Evaluation Summary ===")
    # Calculate scores manually
    system_score = (system_performance.load_balance + 
                   system_performance.throughput + 
                   system_performance.energy_efficiency + 
                   system_performance.cost_efficiency + 
                   system_performance.fairness + 
                   system_performance.reliability) / 6
    
    workflow_score = (workflow_quality.parallel_efficiency + 
                     workflow_quality.data_locality_score + 
                     (1 - workflow_quality.communication_overhead) + 
                     workflow_quality.deadline_satisfaction_rate + 
                     (1 - workflow_quality.priority_violation_rate) + 
                     workflow_quality.workflow_completion_rate) / 6
    
    training_score = (training_process.convergence_speed + 
                     training_process.stability_score + 
                     training_process.exploration_efficiency) / 3
    
    baseline_score = (baseline_comparison.improvement_ratio + 
                     baseline_comparison.efficiency_ratio + 
                     baseline_comparison.scalability_score + 
                     baseline_comparison.robustness_score + 
                     baseline_comparison.adaptability_score) / 5
    
    print(f"System Performance Score: {system_score:.2f}")
    print(f"Workflow Quality Score: {workflow_score:.2f}")
    print(f"Training Process Score: {training_score:.2f}")
    print(f"Baseline Comparison Score: {baseline_score:.2f}")
    print(f"Overall Score: {comprehensive_report['overall_score']:.2f}")
    
    print("\n=== Recommendations ===")
    for rec in comprehensive_report['recommendations']:
        print(f"- {rec}")
    
    print("Complete evaluation metrics and visualization system test completed successfully!")
    
    return {
        'evaluation_results_path': str(output_dir / "complete_evaluation_results.json"),
        'system_chart_path': system_chart_path,
        'workflow_chart_path': workflow_chart_path,
        'training_chart_path': training_chart_path,
        'comparison_chart_path': comparison_chart_path,
        'dashboard_path': dashboard_path,
        'report_path': report_path
    }

if __name__ == "__main__":
    test_complete_evaluation_system()