#!/usr/bin/env python3
"""
WASS-RAG WRENCH Integration Module

This module provides the main interface between WASS and WRENCH simulation framework.
It handles workflow simulation, platform configuration, and result collection.

Author: WASS-RAG Team
Date: 2024-12
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WRENCHSimulator:
    """
    Main WRENCH simulator interface for WASS-RAG system.
    
    This class provides high-level interface for:
    - Platform configuration and modeling
    - Workflow execution simulation
    - Performance metrics collection
    - Integration with WASS components
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize WRENCH simulator.
        
        Args:
            config: Configuration dictionary with simulation parameters
        """
        self.config = config or {}
        self.simulation = None
        self.platform = None
        self.workflow = None
        self.results = {}
        
        # Configuration defaults
        self.default_config = {
            'logging_level': 'INFO',
            'simulation_timeout': 3600,  # 1 hour
            'enable_energy': False,
            'enable_bandwidth_noise': False,
            'temp_dir': tempfile.gettempdir()
        }
        
        # Merge configurations
        self.config = {**self.default_config, **self.config}
        self._setup_logging()
        
        # Check WRENCH availability
        self._check_wrench_availability()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        level = getattr(logging, self.config['logging_level'].upper())
        logger.setLevel(level)
        
    def _check_wrench_availability(self):
        """Check if WRENCH is properly installed and available"""
        try:
            import wrench
            logger.info(f"WRENCH version {wrench.__version__} detected")
            self.wrench = wrench
            return True
        except ImportError as e:
            logger.error(f"WRENCH not available: {e}")
            logger.error("Please install WRENCH following docs/academic/wrench_setup.md")
            self.wrench = None
            return False
    
    def create_platform(self, platform_config: Dict) -> str:
        """
        Create platform XML configuration from WASS platform specification.
        
        Args:
            platform_config: Dictionary containing platform specification
                Example:
                {
                    'hosts': [
                        {'id': 'node1', 'speed': '1Gf', 'cores': 4},
                        {'id': 'node2', 'speed': '2Gf', 'cores': 8}
                    ],
                    'links': [
                        {'id': 'link1', 'bandwidth': '1GBps', 'latency': '0.001s'}
                    ],
                    'routes': [
                        {'src': 'node1', 'dst': 'node2', 'links': ['link1']}
                    ]
                }
        
        Returns:
            Path to generated platform XML file
        """
        if not self.wrench:
            raise RuntimeError("WRENCH not available")
        
        platform_xml = self._generate_platform_xml(platform_config)
        platform_file = os.path.join(self.config['temp_dir'], 'wass_platform.xml')
        
        with open(platform_file, 'w') as f:
            f.write(platform_xml)
        
        logger.info(f"Platform XML generated: {platform_file}")
        return platform_file
    
    def _generate_platform_xml(self, config: Dict) -> str:
        """
        Generate SimGrid platform XML from configuration.
        
        Args:
            config: Platform configuration dictionary
        
        Returns:
            XML string for SimGrid platform
        """
        xml_lines = [
            '<?xml version="1.0"?>',
            '<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">',
            '<platform version="4.1">',
            '  <zone id="AS0" routing="Full">'
        ]
        
        # Add hosts
        for host in config.get('hosts', []):
            host_line = f'    <host id="{host["id"]}" speed="{host["speed"]}"'
            if 'cores' in host:
                host_line += f' core="{host["cores"]}"'
            host_line += '/>'
            xml_lines.append(host_line)
        
        # Add links
        for link in config.get('links', []):
            link_line = f'    <link id="{link["id"]}" bandwidth="{link["bandwidth"]}" latency="{link["latency"]}"/>'
            xml_lines.append(link_line)
        
        # Add routes
        for route in config.get('routes', []):
            xml_lines.append(f'    <route src="{route["src"]}" dst="{route["dst"]}">')
            for link_id in route['links']:
                xml_lines.append(f'      <link_ctn id="{link_id}"/>')
            xml_lines.append('    </route>')
        
        xml_lines.extend([
            '  </zone>',
            '</platform>'
        ])
        
        return '\n'.join(xml_lines)
    
    def create_workflow(self, workflow_spec: Dict) -> str:
        """
        Create WRENCH workflow from WASS workflow specification.
        
        Args:
            workflow_spec: WASS workflow specification
                Example:
                {
                    'name': 'test_workflow',
                    'tasks': [
                        {
                            'id': 'task1',
                            'flops': 1000000000,  # 1 GFlop
                            'bytes_read': 1000000,  # 1 MB
                            'bytes_written': 500000,  # 500 KB
                            'dependencies': []
                        },
                        {
                            'id': 'task2',
                            'flops': 2000000000,  # 2 GFlops
                            'bytes_read': 500000,
                            'bytes_written': 1000000,
                            'dependencies': ['task1']
                        }
                    ]
                }
        
        Returns:
            Workflow identifier for simulation
        """
        if not self.wrench:
            raise RuntimeError("WRENCH not available")
        
        # For now, store workflow specification
        # In actual implementation, this would create WRENCH workflow objects
        self.workflow_spec = workflow_spec
        workflow_id = workflow_spec.get('name', 'default_workflow')
        
        logger.info(f"Workflow created: {workflow_id} with {len(workflow_spec['tasks'])} tasks")
        return workflow_id
    
    def run_simulation(self, platform_file: str, workflow_id: str, 
                      scheduler_config: Optional[Dict] = None) -> Dict:
        """
        Run WRENCH simulation with specified platform and workflow.
        
        Args:
            platform_file: Path to platform XML file
            workflow_id: Workflow identifier
            scheduler_config: Scheduler configuration (optional)
        
        Returns:
            Dictionary containing simulation results
        """
        if not self.wrench:
            logger.warning("WRENCH not available, returning mock results")
            return self._mock_simulation_results()
        
        try:
            # Initialize simulation
            simulation = self.wrench.Simulation()
            
            # Add platform
            simulation.add_platform(platform_file)
            
            # TODO: Add actual workflow and scheduler configuration
            # This is a placeholder for the real implementation
            
            # Run simulation
            logger.info("Starting WRENCH simulation...")
            simulation.start()
            
            # Collect results
            results = self._collect_simulation_results(simulation)
            
            logger.info("Simulation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    def _mock_simulation_results(self) -> Dict:
        """
        Generate mock simulation results for testing when WRENCH is not available.
        
        Returns:
            Dictionary with mock simulation results
        """
        import random
        import time
        
        num_tasks = len(self.workflow_spec.get('tasks', []))
        
        return {
            'simulation_time': round(random.uniform(100, 1000), 2),
            'makespan': round(random.uniform(50, 500), 2),
            'energy_consumption': round(random.uniform(1000, 5000), 2),
            'task_execution_times': {
                f"task_{i}": round(random.uniform(10, 100), 2)
                for i in range(num_tasks)
            },
            'resource_utilization': {
                'cpu_avg': round(random.uniform(0.3, 0.9), 3),
                'memory_avg': round(random.uniform(0.2, 0.8), 3),
                'network_avg': round(random.uniform(0.1, 0.6), 3)
            },
            'workflow_id': self.workflow_spec.get('name', 'mock_workflow'),
            'timestamp': time.time(),
            'mock_data': True
        }
    
    def _collect_simulation_results(self, simulation) -> Dict:
        """
        Collect and process simulation results from WRENCH.
        
        Args:
            simulation: WRENCH simulation object
        
        Returns:
            Dictionary containing processed results
        """
        # TODO: Implement actual result collection from WRENCH
        # This is a placeholder for the real implementation
        
        results = {
            'simulation_time': 0.0,
            'makespan': 0.0,
            'energy_consumption': 0.0,
            'task_execution_times': {},
            'resource_utilization': {},
            'workflow_id': self.workflow_spec.get('name', 'unknown'),
            'timestamp': 0.0,
            'mock_data': False
        }
        
        logger.info("Results collected from WRENCH simulation")
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """
        Analyze simulation results and compute performance metrics.
        
        Args:
            results: Raw simulation results
        
        Returns:
            Dictionary containing analysis and metrics
        """
        analysis = {
            'performance_metrics': {
                'makespan': results.get('makespan', 0),
                'throughput': len(self.workflow_spec.get('tasks', [])) / max(results.get('makespan', 1), 1),
                'efficiency': results.get('resource_utilization', {}).get('cpu_avg', 0),
                'energy_efficiency': results.get('makespan', 1) / max(results.get('energy_consumption', 1), 1)
            },
            'bottlenecks': self._identify_bottlenecks(results),
            'recommendations': self._generate_recommendations(results),
            'summary': self._generate_summary(results)
        }
        
        return analysis
    
    def _identify_bottlenecks(self, results: Dict) -> List[str]:
        """Identify performance bottlenecks from simulation results"""
        bottlenecks = []
        
        cpu_util = results.get('resource_utilization', {}).get('cpu_avg', 0)
        if cpu_util < 0.5:
            bottlenecks.append("Low CPU utilization - consider task parallelization")
        
        network_util = results.get('resource_utilization', {}).get('network_avg', 0)
        if network_util > 0.8:
            bottlenecks.append("High network utilization - potential network bottleneck")
        
        return bottlenecks
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate optimization recommendations based on results"""
        recommendations = []
        
        efficiency = results.get('resource_utilization', {}).get('cpu_avg', 0)
        if efficiency < 0.6:
            recommendations.append("Consider increasing task parallelism or reducing task granularity")
        
        if results.get('mock_data', False):
            recommendations.append("Install WRENCH for accurate simulation results")
        
        return recommendations
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate human-readable summary of simulation results"""
        makespan = results.get('makespan', 0)
        num_tasks = len(self.workflow_spec.get('tasks', []))
        mock_note = " (simulated data)" if results.get('mock_data', False) else ""
        
        return f"Executed {num_tasks} tasks in {makespan:.2f} seconds{mock_note}"


class WRENCHIntegrationError(Exception):
    """Exception raised for WRENCH integration errors"""
    pass


def test_wrench_integration():
    """
    Test function for WRENCH integration.
    This can be run independently to verify the setup.
    """
    print("🧪 Testing WRENCH Integration...")
    
    # Test platform configuration
    platform_config = {
        'hosts': [
            {'id': 'node1', 'speed': '1Gf', 'cores': 4},
            {'id': 'node2', 'speed': '2Gf', 'cores': 8}
        ],
        'links': [
            {'id': 'link1', 'bandwidth': '1GBps', 'latency': '0.001s'}
        ],
        'routes': [
            {'src': 'node1', 'dst': 'node2', 'links': ['link1']}
        ]
    }
    
    # Test workflow specification
    workflow_spec = {
        'name': 'test_workflow',
        'tasks': [
            {
                'id': 'task1',
                'flops': 1000000000,
                'bytes_read': 1000000,
                'bytes_written': 500000,
                'dependencies': []
            },
            {
                'id': 'task2',
                'flops': 2000000000,
                'bytes_read': 500000,
                'bytes_written': 1000000,
                'dependencies': ['task1']
            }
        ]
    }
    
    try:
        # Initialize simulator
        simulator = WRENCHSimulator()
        print("✅ Simulator initialized")
        
        # Create platform
        platform_file = simulator.create_platform(platform_config)
        print(f"✅ Platform created: {platform_file}")
        
        # Create workflow
        workflow_id = simulator.create_workflow(workflow_spec)
        print(f"✅ Workflow created: {workflow_id}")
        
        # Run simulation
        results = simulator.run_simulation(platform_file, workflow_id)
        print(f"✅ Simulation completed")
        
        # Analyze results
        analysis = simulator.analyze_results(results)
        print(f"✅ Results analyzed")
        
        # Print summary
        print("\n📊 Simulation Summary:")
        print(f"   {analysis['summary']}")
        print(f"   Makespan: {results['makespan']:.2f}s")
        print(f"   Energy: {results['energy_consumption']:.2f}J")
        
        if analysis['bottlenecks']:
            print(f"\n⚠️  Bottlenecks: {', '.join(analysis['bottlenecks'])}")
        
        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   - {rec}")
        
        print("\n🎉 WRENCH integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test when module is executed directly
    success = test_wrench_integration()
    sys.exit(0 if success else 1)
