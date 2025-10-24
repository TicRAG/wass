#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Chart Generation Script
Generate ACM-standard charts based on experimental results
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns

# Set font support for English
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_results():
    """Load experimental results data"""
    results_file = "results/final_experiments/detailed_results.csv"
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Experimental results file does not exist: {results_file}")
    
    df = pd.read_csv(results_file)
    return df.to_dict('records')

def create_scheduler_performance_comparison(results):
    """Create scheduler performance comparison chart"""
    # Group makespan statistics by scheduler
    scheduler_stats = defaultdict(list)
    
    for result in results:
        scheduler = result['scheduler_name']
        makespan = result['makespan']
        scheduler_stats[scheduler].append(makespan)
    
    # Calculate mean and standard deviation
    schedulers = []
    avg_makespans = []
    std_makespans = []
    
    # Sort by expected performance
    # Unified scheduler naming: replace legacy 'WASS-Heuristic'/'WASS-DRL' with 'WASS_DRL'
    scheduler_order = ['FIFO', 'HEFT', 'WASS_DRL', 'WASS_RAG']
    
    for scheduler in scheduler_order:
        if scheduler in scheduler_stats:
            schedulers.append(scheduler)
            avg_makespans.append(np.mean(scheduler_stats[scheduler]))
            std_makespans.append(np.std(scheduler_stats[scheduler]))
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(schedulers, avg_makespans, yerr=std_makespans, capsize=5, 
                  color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
    
    ax.set_xlabel('Scheduler')
    ax.set_ylabel('Average Makespan (seconds)')
    ax.set_title('Scheduler Performance Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (avg, std) in enumerate(zip(avg_makespans, std_makespans)):
        ax.text(i, avg + std + 0.1, f'{avg:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/scheduler_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_makespan_distribution(results):
    """Create makespan distribution chart"""
    # Group by scheduler
    scheduler_data = defaultdict(list)
    
    for result in results:
        scheduler = result['scheduler_name']
        makespan = result['makespan']
        scheduler_data[scheduler].append(makespan)
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by expected performance
    # Unified scheduler naming
    scheduler_order = ['FIFO', 'HEFT', 'WASS_DRL', 'WASS_RAG']
    data_to_plot = [scheduler_data[scheduler] for scheduler in scheduler_order if scheduler in scheduler_data]
    labels = [scheduler for scheduler in scheduler_order if scheduler in scheduler_data]
    
    box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Set colors
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Scheduler')
    ax.set_ylabel('Makespan (seconds)')
    ax.set_title('Makespan Distribution by Scheduler')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/makespan_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cpu_utilization_chart(results):
    """Create CPU utilization chart"""
    # Calculate average CPU utilization for each scheduler
    scheduler_utilization = defaultdict(list)
    
    for result in results:
        scheduler = result['scheduler_name']
        # Calculate average CPU utilization across all nodes
        avg_util = np.mean(list(result['cpu_utilization'].values()))
        scheduler_utilization[scheduler].append(avg_util)
    
    # Calculate averages
    schedulers = []
    avg_utils = []
    
    # Unified scheduler naming
    scheduler_order = ['FIFO', 'HEFT', 'WASS_DRL', 'WASS_RAG']
    
    for scheduler in scheduler_order:
        if scheduler in scheduler_utilization:
            schedulers.append(scheduler)
            avg_utils.append(np.mean(scheduler_utilization[scheduler]) * 100)  # Convert to percentage
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(schedulers, avg_utils, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
    
    ax.set_xlabel('Scheduler')
    ax.set_ylabel('Average CPU Utilization (%)')
    ax.set_title('CPU Utilization Comparison by Scheduler')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, avg in enumerate(avg_utils):
        ax.text(i, avg + 1, f'{avg:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/cpu_utilization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_task_scaling_analysis(results):
    """Create task scaling analysis chart"""
    # Group by task count and scheduler
    task_scaling = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        scheduler = result['scheduler_name']
        task_count = result['task_count']
        makespan = result['makespan']
        task_scaling[task_count][scheduler].append(makespan)
    
    # Calculate average makespan for each task scale and scheduler
    task_counts = sorted(task_scaling.keys())
    scheduler_order = ['FIFO', 'HEFT', 'WASS-Heuristic', 'WASS-DRL', 'WASS-RAG']
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scheduler in scheduler_order:
        avg_makespans = []
        for task_count in task_counts:
            if scheduler in task_scaling[task_count]:
                avg_makespans.append(np.mean(task_scaling[task_count][scheduler]))
            else:
                avg_makespans.append(0)
        
        # Only plot schedulers with data
        if any(avg_makespans):
            ax.plot(task_counts, avg_makespans, marker='o', label=scheduler, linewidth=2, markersize=8)
    
    ax.set_xlabel('Task Count')
    ax.set_ylabel('Average Makespan (seconds)')
    ax.set_title('Scheduler Performance Across Task Scales')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/task_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_improvement_chart(results):
    """Create performance improvement chart"""
    # Calculate performance improvement relative to FIFO
    scheduler_makespans = defaultdict(list)
    
    for result in results:
        scheduler = result['scheduler_name']
        makespan = result['makespan']
        scheduler_makespans[scheduler].append(makespan)
    
    # Calculate average makespan
    avg_makespans = {scheduler: np.mean(makespans) 
                     for scheduler, makespans in scheduler_makespans.items()}
    
    if 'FIFO' not in avg_makespans:
        print("Warning: Missing FIFO baseline data")
        return
    
    fifo_avg = avg_makespans['FIFO']
    # Unified scheduler naming
    scheduler_order = ['HEFT', 'WASS_DRL', 'WASS_RAG']
    
    improvements = []
    labels = []
    
    for scheduler in scheduler_order:
        if scheduler in avg_makespans:
            improvement = (fifo_avg - avg_makespans[scheduler]) / fifo_avg * 100
            improvements.append(improvement)
            labels.append(scheduler)
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, improvements, color=['#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
    
    ax.set_xlabel('Scheduler')
    ax.set_ylabel('Performance Improvement (%)')
    ax.set_title('Performance Improvement Relative to FIFO Scheduler')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, improvement in enumerate(improvements):
        ax.text(i, improvement + 1, f'{improvement:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/performance_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    print("Starting paper chart generation...")
    
    # Ensure charts directory exists
    os.makedirs('charts', exist_ok=True)
    
    try:
        # Load experimental results
        results = load_experiment_results()
        print(f"Loaded {len(results)} experimental results")
        
        # Generate various charts
        print("Generating scheduler performance comparison chart...")
        create_scheduler_performance_comparison(results)
        
        print("Generating makespan distribution chart...")
        create_makespan_distribution(results)
        
        # print("Generating CPU utilization chart...")
        # create_cpu_utilization_chart(results)
        
        print("Generating task scaling analysis chart...")
        create_task_scaling_analysis(results)
        
        print("Generating performance improvement chart...")
        create_performance_improvement_chart(results)
        
        print("All charts generated successfully!")
        print("\nGenerated charts:")
        for chart_file in os.listdir('charts'):
            if chart_file.endswith('.png'):
                print(f"  - {chart_file}")
                
    except Exception as e:
        print(f"Error during chart generation: {e}")
        raise

if __name__ == "__main__":
    main()
