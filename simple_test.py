#!/usr/bin/env python3
"""
Simple test script for evaluation metrics system
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation_metrics import ComprehensiveEvaluator

def test_evaluation_metrics():
    """Test the evaluation metrics system"""
    print("Testing evaluation metrics system...")
    
    # Create output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Generate sample system metrics
    system_metrics = {
        'makespan': 75.0,
        'avg_waiting_time': 10.5,
        'avg_turnaround_time': 25.3,
        'total_tasks': 50,
        'resource_utilization': {
            'CPU': 0.8,
            'Memory': 0.75,
            'GPU': 0.7,
            'Network': 0.65
        },
        'node_loads': [0.8, 0.75, 0.7, 0.65],
        'task_completion_rate': 0.95
    }
    
    # Test system performance evaluation
    print("Testing system performance evaluation...")
    system_result = evaluator.evaluate_system_performance(system_metrics)
    print(f"System metrics created successfully")
    print(f"Load balance: {system_result.load_balance:.3f}")
    print(f"Throughput: {system_result.throughput:.3f}")
    
    # Generate sample workflow metrics
    workflow_metrics = {
        'scheduling_length': 60.0,
        'critical_path_ratio': 0.65,
        'data_locality_score': 0.75,
        'communication_overhead': 15.0,
        'deadline_satisfaction_rate': 0.85,
        'priority_violation_rate': 0.15,
        'workflow_completion_rate': 0.9
    }
    
    # Test workflow quality evaluation
    print("\nTesting workflow quality evaluation...")
    workflow_result = evaluator.evaluate_workflow_quality(workflow_metrics)
    print(f"Workflow metrics created successfully")
    print(f"Parallel efficiency: {workflow_result.parallel_efficiency:.3f}")
    
    # Generate sample training metrics
    training_history = []
    for i in range(10):
        episode = {
            'total_reward': -10 + i * 2 + np.random.normal(0, 1),
            'avg_loss': 5.0 - i * 0.4 + np.random.normal(0, 0.5),
            'avg_q_value': i * 1.5 + np.random.normal(0, 1)
        }
        training_history.append(episode)
    
    # Test training process evaluation
    print("\nTesting training process evaluation...")
    training_result = evaluator.evaluate_training_process(training_history)
    print(f"Training metrics created successfully")
    print(f"Convergence speed: {training_result.convergence_speed:.3f}")
    print(f"Stability score: {training_result.stability_score:.3f}")
    
    # Generate sample baseline comparison data
    our_result = {
        'makespan': 75.0,
        'efficiency': 0.8,
        'task_completion_rate': 0.95,
        'task_types': {'compute': 20, 'io': 15, 'network': 15},
        'type_performance': {'compute': 0.85, 'io': 0.8, 'network': 0.75}
    }
    
    baseline_results = {
        'HEFT': {'makespan': 85.0, 'efficiency': 0.75},
        'CPOP': {'makespan': 90.0, 'efficiency': 0.7},
        'Min-Min': {'makespan': 95.0, 'efficiency': 0.65}
    }
    
    # Test baseline comparison
    print("\nTesting baseline comparison...")
    comparison_result = evaluator.compare_with_baselines(our_result, baseline_results)
    print(f"Comparison metrics created successfully")
    print(f"Improvement ratio: {comparison_result.improvement_ratio:.3f}")
    print(f"Speedup ratio: {comparison_result.speedup_ratio:.3f}")
    
    # Test comprehensive report generation
    print("\nTesting comprehensive report generation...")
    report_path = str(output_dir / "comprehensive_report.json")
    comprehensive_report = evaluator.generate_comprehensive_report(report_path)
    print(f"Comprehensive report generated successfully")
    print(f"Overall score: {comprehensive_report.get('overall_score', 0):.3f}")
    
    # Save results to JSON for inspection
    results = {
        'system_metrics': system_result.to_dict(),
        'workflow_metrics': workflow_result.to_dict(),
        'training_metrics': training_result.to_dict(),
        'comparison_metrics': comparison_result.to_dict(),
        'comprehensive_report': comprehensive_report
    }
    
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to {results_path}")
    print("Evaluation metrics system test completed successfully!")
    
    return results

if __name__ == "__main__":
    test_evaluation_metrics()