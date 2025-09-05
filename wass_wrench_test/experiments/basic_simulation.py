#!/usr/bin/env python3
"""
WASS-RAG Basic Simulation Experiment

This script demonstrates the basic functionality of WASS-RAG system
with WRENCH integration for workflow simulation.

Usage:
    python basic_simulation.py [--config config.yaml] [--output results/]

Author: WASS-RAG Team
Date: 2024-12
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from wrench_integration.simulator import WRENCHSimulator
from src.config_loader import ConfigLoader


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    else:
        # Return default configuration
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default experiment configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'experiment': {
            'name': 'basic_simulation',
            'description': 'Basic WASS-RAG simulation experiment',
            'version': '1.0.0'
        },
        'platform': {
            'hosts': [
                {'id': 'compute_node_1', 'speed': '1Gf', 'cores': 4},
                {'id': 'compute_node_2', 'speed': '2Gf', 'cores': 8},
                {'id': 'storage_node', 'speed': '500Mf', 'cores': 2}
            ],
            'links': [
                {'id': 'ethernet_1', 'bandwidth': '1GBps', 'latency': '0.001s'},
                {'id': 'ethernet_2', 'bandwidth': '1GBps', 'latency': '0.001s'},
                {'id': 'infiniband', 'bandwidth': '10GBps', 'latency': '0.0001s'}
            ],
            'routes': [
                {'src': 'compute_node_1', 'dst': 'storage_node', 'links': ['ethernet_1']},
                {'src': 'compute_node_2', 'dst': 'storage_node', 'links': ['ethernet_2']},
                {'src': 'compute_node_1', 'dst': 'compute_node_2', 'links': ['infiniband']}
            ]
        },
        'workflows': [
            {
                'name': 'montage_small',
                'description': 'Small Montage astronomy workflow',
                'tasks': [
                    {
                        'id': 'mProject_1',
                        'flops': 1.5e9,  # 1.5 GFlops
                        'bytes_read': 50e6,  # 50 MB
                        'bytes_written': 45e6,  # 45 MB
                        'dependencies': []
                    },
                    {
                        'id': 'mProject_2',
                        'flops': 1.5e9,
                        'bytes_read': 50e6,
                        'bytes_written': 45e6,
                        'dependencies': []
                    },
                    {
                        'id': 'mDiffFit',
                        'flops': 2.5e9,  # 2.5 GFlops
                        'bytes_read': 90e6,  # Input from both mProject tasks
                        'bytes_written': 30e6,  # 30 MB
                        'dependencies': ['mProject_1', 'mProject_2']
                    },
                    {
                        'id': 'mConcatFit',
                        'flops': 0.5e9,  # 0.5 GFlops
                        'bytes_read': 30e6,
                        'bytes_written': 20e6,  # 20 MB
                        'dependencies': ['mDiffFit']
                    },
                    {
                        'id': 'mBgModel',
                        'flops': 1.0e9,  # 1 GFlop
                        'bytes_read': 20e6,
                        'bytes_written': 15e6,  # 15 MB
                        'dependencies': ['mConcatFit']
                    },
                    {
                        'id': 'mBackground_1',
                        'flops': 2.0e9,  # 2 GFlops
                        'bytes_read': 60e6,  # Original + background model
                        'bytes_written': 45e6,
                        'dependencies': ['mProject_1', 'mBgModel']
                    },
                    {
                        'id': 'mBackground_2',
                        'flops': 2.0e9,
                        'bytes_read': 60e6,
                        'bytes_written': 45e6,
                        'dependencies': ['mProject_2', 'mBgModel']
                    },
                    {
                        'id': 'mAdd',
                        'flops': 3.0e9,  # 3 GFlops
                        'bytes_read': 90e6,  # Both background corrected images
                        'bytes_written': 100e6,  # Final mosaic
                        'dependencies': ['mBackground_1', 'mBackground_2']
                    }
                ]
            }
        ],
        'simulation': {
            'logging_level': 'INFO',
            'timeout': 3600,
            'enable_energy': False,
            'enable_noise': False
        },
        'output': {
            'save_results': True,
            'results_dir': 'results',
            'detailed_logs': True
        }
    }


def create_output_directory(base_dir: str) -> str:
    """
    Create output directory with timestamp.
    
    Args:
        base_dir: Base directory for results
    
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"basic_simulation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save simulation results to files.
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
    """
    # Save raw results as JSON
    results_file = os.path.join(output_dir, 'simulation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save analysis as YAML for readability
    if 'analysis' in results:
        analysis_file = os.path.join(output_dir, 'analysis.yaml')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            yaml.dump(results['analysis'], f, default_flow_style=False)
    
    # Save summary as text
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("WASS-RAG Basic Simulation Results\n")
        f.write("=" * 40 + "\n\n")
        
        if 'analysis' in results and 'summary' in results['analysis']:
            f.write(f"Summary: {results['analysis']['summary']}\n\n")
        
        if 'simulation_results' in results:
            sim_results = results['simulation_results']
            f.write(f"Makespan: {sim_results.get('makespan', 'N/A'):.2f}s\n")
            f.write(f"Energy: {sim_results.get('energy_consumption', 'N/A'):.2f}J\n")
            f.write(f"Simulation Time: {sim_results.get('simulation_time', 'N/A'):.2f}s\n\n")
        
        if 'analysis' in results and 'bottlenecks' in results['analysis']:
            bottlenecks = results['analysis']['bottlenecks']
            if bottlenecks:
                f.write("Identified Bottlenecks:\n")
                for bottleneck in bottlenecks:
                    f.write(f"  - {bottleneck}\n")
                f.write("\n")
        
        if 'analysis' in results and 'recommendations' in results['analysis']:
            recommendations = results['analysis']['recommendations']
            if recommendations:
                f.write("Optimization Recommendations:\n")
                for rec in recommendations:
                    f.write(f"  - {rec}\n")
                f.write("\n")
        
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")


def run_basic_simulation(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Run the basic simulation experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory path
    
    Returns:
        Dictionary containing all results
    """
    print("🚀 Starting WASS-RAG Basic Simulation Experiment")
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize simulator
    print("\n📋 Phase 1: Initializing WRENCH Simulator...")
    simulator_config = config.get('simulation', {})
    simulator = WRENCHSimulator(simulator_config)
    
    # Create platform
    print("🏗️  Phase 2: Creating Simulation Platform...")
    platform_config = config['platform']
    platform_file = simulator.create_platform(platform_config)
    print(f"   Platform file: {platform_file}")
    
    # Save platform configuration
    platform_config_file = os.path.join(output_dir, 'platform_config.yaml')
    with open(platform_config_file, 'w', encoding='utf-8') as f:
        yaml.dump(platform_config, f, default_flow_style=False)
    
    all_results = {
        'experiment_config': config,
        'workflows': {},
        'platform_file': platform_file
    }
    
    # Run simulations for each workflow
    workflows = config.get('workflows', [])
    print(f"\n⚙️  Phase 3: Running Simulations for {len(workflows)} workflow(s)...")
    
    for i, workflow_spec in enumerate(workflows, 1):
        workflow_name = workflow_spec['name']
        print(f"\n   🔄 Workflow {i}/{len(workflows)}: {workflow_name}")
        
        # Create workflow
        workflow_id = simulator.create_workflow(workflow_spec)
        
        # Run simulation
        print(f"      ⏳ Running simulation...")
        simulation_results = simulator.run_simulation(platform_file, workflow_id)
        
        # Analyze results
        print(f"      📊 Analyzing results...")
        analysis = simulator.analyze_results(simulation_results)
        
        # Store results
        all_results['workflows'][workflow_name] = {
            'workflow_spec': workflow_spec,
            'simulation_results': simulation_results,
            'analysis': analysis
        }
        
        # Print summary for this workflow
        print(f"      ✅ {analysis['summary']}")
        if analysis['bottlenecks']:
            print(f"         ⚠️  Bottlenecks: {', '.join(analysis['bottlenecks'])}")
    
    return all_results


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print experiment summary to console.
    
    Args:
        results: Complete experiment results
    """
    print("\n" + "=" * 60)
    print("📋 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    workflows = results.get('workflows', {})
    print(f"🔢 Total workflows simulated: {len(workflows)}")
    
    for workflow_name, workflow_data in workflows.items():
        print(f"\n📊 {workflow_name}:")
        sim_results = workflow_data['simulation_results']
        analysis = workflow_data['analysis']
        
        print(f"   • {analysis['summary']}")
        print(f"   • Makespan: {sim_results.get('makespan', 'N/A'):.2f}s")
        print(f"   • Energy: {sim_results.get('energy_consumption', 'N/A'):.2f}J")
        
        metrics = analysis.get('performance_metrics', {})
        if metrics:
            print(f"   • Throughput: {metrics.get('throughput', 'N/A'):.3f} tasks/s")
            print(f"   • CPU Efficiency: {metrics.get('efficiency', 'N/A'):.1%}")
        
        if analysis.get('bottlenecks'):
            print(f"   • Bottlenecks: {len(analysis['bottlenecks'])} identified")
        
        if analysis.get('recommendations'):
            print(f"   • Recommendations: {len(analysis['recommendations'])} available")
    
    print("\n✨ Experiment completed successfully!")


def main():
    """Main experiment execution function"""
    parser = argparse.ArgumentParser(
        description="WASS-RAG Basic Simulation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help="Path to experiment configuration file (YAML or JSON)"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help="Base directory for output files (default: results)"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print("📝 Loading experiment configuration...")
        config = load_experiment_config(args.config)
        
        if args.verbose:
            print(f"   Using config: {args.config if args.config else 'default'}")
            print(f"   Experiment: {config['experiment']['name']}")
            print(f"   Description: {config['experiment']['description']}")
        
        # Create output directory
        output_dir = create_output_directory(args.output)
        
        # Save configuration
        config_file = os.path.join(output_dir, 'experiment_config.yaml')
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Run experiment
        results = run_basic_simulation(config, output_dir)
        
        # Save results
        print("\n💾 Saving results...")
        save_results(results, output_dir)
        
        # Print summary
        print_experiment_summary(results)
        
        print(f"\n📁 All results saved to: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
