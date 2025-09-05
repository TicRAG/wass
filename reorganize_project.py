#!/usr/bin/env python3
"""
Project Cleanup and Reorganization Script

This script cleans up the current concept proof implementation and
prepares the project for Level 2: High-Fidelity Simulation development.
"""

import os
import shutil
from pathlib import Path
import json

class ProjectReorganizer:
    """Reorganize project for academic research focus"""
    
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.backup_dir = self.root / "backup_concept_proof"
        
    def create_backup(self):
        """Create backup of current implementation"""
        print("📦 Creating backup of concept proof implementation...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup key files
        backup_items = [
            "scripts/reproduce_paper_benchmark.py",
            "scripts/enhanced_benchmark.py", 
            "scripts/gap_analysis.py",
            "results/",
            "CREDIBILITY_ANALYSIS.md",
            "CREDIBILITY_CONCLUSION.md",
            "FINAL_SUMMARY.md",
            "PAPER_ANALYSIS.md",
            "EXPERIMENT_GUIDE.md"
        ]
        
        for item in backup_items:
            src = self.root / item
            if src.exists():
                if src.is_dir():
                    dst = self.backup_dir / item
                    shutil.copytree(src, dst)
                    print(f"  Backed up directory: {item}")
                else:
                    dst = self.backup_dir / item
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    print(f"  Backed up file: {item}")
    
    def clean_outdated_files(self):
        """Remove outdated files and directories"""
        print("\n🧹 Cleaning outdated files...")
        
        files_to_remove = [
            "demo.py",
            "configs_wrench_demo.yaml",
            "experiment_analysis_report.md",
            "experiment_comparison.csv",
            "CREDIBILITY_ANALYSIS.md",
            "CREDIBILITY_CONCLUSION.md", 
            "FINAL_SUMMARY.md",
            "PAPER_ANALYSIS.md",
            "EXPERIMENT_GUIDE.md"
        ]
        
        dirs_to_remove = [
            "results/"
        ]
        
        for file_path in files_to_remove:
            full_path = self.root / file_path
            if full_path.exists():
                full_path.unlink()
                print(f"  Removed file: {file_path}")
        
        for dir_path in dirs_to_remove:
            full_path = self.root / dir_path
            if full_path.exists():
                shutil.rmtree(full_path)
                print(f"  Removed directory: {dir_path}")
    
    def reorganize_scripts(self):
        """Reorganize scripts directory"""
        print("\n📁 Reorganizing scripts directory...")
        
        scripts_dir = self.root / "scripts"
        
        # Remove concept proof scripts
        concept_scripts = [
            "reproduce_paper_benchmark.py",
            "enhanced_benchmark.py",
            "gap_analysis.py",
            "run_lf_experiments.py",
            "run_model_comparison.py",
            "analyze_results.py"
        ]
        
        for script in concept_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                script_path.unlink()
                print(f"  Removed: {script}")
        
        # Keep only essential scripts
        keep_scripts = [
            "gen_fake_data.py"  # Might be useful for data generation
        ]
        
        print(f"  Kept essential scripts: {', '.join(keep_scripts)}")
    
    def create_new_structure(self):
        """Create new directory structure for academic research"""
        print("\n🏗️  Creating new project structure...")
        
        new_dirs = [
            "wrench_integration",
            "ml/gnn", 
            "ml/drl",
            "ml/rag",
            "datasets/real_workflows",
            "datasets/synthetic", 
            "experiments/benchmarks",
            "experiments/ablation",
            "analysis/performance",
            "analysis/visualization",
            "docs/academic",
            "tests/unit",
            "tests/integration"
        ]
        
        for dir_path in new_dirs:
            full_path = self.root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {dir_path}/")
            
            # Add __init__.py for Python packages
            if not any(x in dir_path for x in ['datasets', 'experiments', 'analysis', 'docs', 'tests']):
                init_file = full_path / "__init__.py"
                init_file.write_text("# WASS-RAG Academic Research Implementation\n")
    
    def update_configs(self):
        """Update configuration files for academic research"""
        print("\n⚙️  Updating configuration files...")
        
        # Create academic-focused config
        academic_config = {
            "project": {
                "name": "WASS-RAG Academic Research",
                "version": "2.0.0-academic",
                "description": "High-Fidelity Simulation for Workflow-Aware Scheduling"
            },
            "simulation": {
                "engine": "wrench",
                "platform": "simgrid",
                "log_level": "INFO"
            },
            "machine_learning": {
                "framework": "pytorch",
                "graph_library": "dgl",
                "device": "auto"
            },
            "experiments": {
                "output_dir": "experiments/results",
                "log_dir": "experiments/logs",
                "checkpoint_dir": "experiments/checkpoints"
            },
            "datasets": {
                "real_workflows_dir": "datasets/real_workflows",
                "synthetic_dir": "datasets/synthetic",
                "cache_dir": "datasets/cache"
            }
        }
        
        config_path = self.root / "configs" / "academic.yaml"
        config_path.parent.mkdir(exist_ok=True)
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(academic_config, f, default_flow_style=False, indent=2)
        
        print(f"  Created: {config_path}")
    
    def clean_src_directory(self):
        """Clean and reorganize src directory"""
        print("\n🔧 Cleaning src directory...")
        
        src_dir = self.root / "src"
        
        # Keep core infrastructure
        keep_files = [
            "__init__.py",
            "config_loader.py", 
            "factory.py",
            "interfaces.py",
            "utils.py"
        ]
        
        # Remove concept proof implementations
        remove_files = [
            "pipeline_run.py",
            "pipeline_enhanced.py",
            "architecture.md"
        ]
        
        for file_name in remove_files:
            file_path = src_dir / file_name
            if file_path.exists():
                file_path.unlink()
                print(f"  Removed: {file_name}")
        
        # Clean subdirectories but keep structure
        subdirs = ["data", "drl", "eval", "graph", "label_model", "labeling", "rag"]
        
        for subdir in subdirs:
            subdir_path = src_dir / subdir
            if subdir_path.exists():
                # Keep __init__.py, remove implementation files
                for file_path in subdir_path.iterdir():
                    if file_path.name != "__init__.py":
                        if file_path.is_file():
                            file_path.unlink()
                            print(f"  Removed: {subdir}/{file_path.name}")
    
    def create_academic_readme(self):
        """Create new README focused on academic research"""
        print("\n📝 Creating academic README...")
        
        readme_content = """# WASS-RAG: Academic Research Implementation

A high-fidelity simulation framework for Workflow-Aware Scheduling with Retrieval-Augmented Generation.

## 🎯 Research Focus

This implementation targets **Level 2: High-Fidelity Simulation** for academic research purposes:

- WRENCH/SimGrid integration for realistic workflow simulation
- Complete GNN+DRL+RAG implementation  
- Large-scale benchmark datasets
- Reproducible experimental framework

## 🏗️ Architecture

```
WASS-RAG Academic
├── wrench_integration/    # WRENCH simulator integration
├── ml/                   # Machine learning components
│   ├── gnn/             # Graph Neural Networks
│   ├── drl/             # Deep Reinforcement Learning  
│   └── rag/             # Retrieval-Augmented Generation
├── datasets/            # Workflow datasets
├── experiments/         # Experimental scripts
├── analysis/           # Results analysis
└── src/                # Core infrastructure
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- WRENCH Simulator
- PyTorch
- DGL/PyG

### Installation
```bash
# Clone repository
git clone <repository-url>
cd wass

# Create environment  
python -m venv wass_env
source wass_env/bin/activate  # Linux/Mac
# or wass_env\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install WRENCH (see docs/academic/wrench_setup.md)
```

### Running Experiments
```bash
# Basic workflow simulation
python experiments/basic_simulation.py

# Full ML training
python experiments/train_rag_agent.py

# Benchmark comparison
python experiments/benchmark_comparison.py
```

## 📊 Research Goals

1. **High-Fidelity Simulation**: Realistic workflow execution modeling
2. **Advanced ML**: State-of-the-art GNN+DRL+RAG implementation
3. **Comprehensive Evaluation**: Large-scale benchmarks and analysis
4. **Academic Publication**: Top-tier conference/journal submission

## 📚 Documentation

- [Academic Roadmap](ACADEMIC_ROADMAP.md)
- [WRENCH Integration Guide](docs/academic/wrench_integration.md)
- [ML Implementation Details](docs/academic/ml_architecture.md)
- [Experiment Framework](docs/academic/experiments.md)

## 🔗 Related Work

Based on the paper: "WASS-RAG: A Knowledge-Retrieval Augmented DRL Framework for Workflow-Aware Scheduling on Slurm"

## 📄 License

[Add appropriate academic license]

## 🤝 Contributing

This is an academic research project. Contributions welcome for:
- WRENCH integration improvements
- ML algorithm enhancements  
- Additional benchmark workflows
- Experimental analysis tools

---
*This is the academic research version focusing on high-fidelity simulation.*
"""
        
        readme_path = self.root / "readme.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        print(f"  Updated: readme.md")
    
    def update_requirements(self):
        """Update requirements for academic research"""
        print("\n📦 Updating requirements...")
        
        academic_requirements = """# WASS-RAG Academic Research Requirements

# Core Python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Configuration
PyYAML>=6.0
hydra-core>=1.1.0
omegaconf>=2.1.0

# Machine Learning
torch>=1.12.0
dgl>=0.9.0
# torch-geometric>=2.1.0  # Alternative to DGL

# Knowledge Base
faiss-cpu>=1.7.0
# faiss-gpu>=1.7.0  # For GPU acceleration
h5py>=3.6.0

# Experiment Tracking
mlflow>=1.20.0
tensorboard>=2.8.0
wandb>=0.12.0  # Optional

# Visualization
plotly>=5.6.0
dash>=2.0.0  # For interactive analysis

# Development
pytest>=6.2.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.910

# Documentation
sphinx>=4.3.0
sphinx-rtd-theme>=1.0.0

# Utilities
tqdm>=4.62.0
tabulate>=0.8.9
click>=8.0.0

# WRENCH Dependencies (to be installed separately)
# See docs/academic/wrench_setup.md for WRENCH installation
"""
        
        req_path = self.root / "requirements.txt"
        req_path.write_text(academic_requirements, encoding='utf-8')
        print(f"  Updated: requirements.txt")
    
    def create_placeholder_files(self):
        """Create placeholder files for key components"""
        print("\n📄 Creating placeholder files...")
        
        placeholders = {
            "wrench_integration/simulator.py": '''"""
WRENCH Simulator Integration

This module provides the main interface to WRENCH for high-fidelity
workflow simulation.
"""

class WRENCHSimulator:
    """Main WRENCH simulator interface"""
    
    def __init__(self, config):
        self.config = config
        # TODO: Initialize WRENCH
        
    def simulate_workflow(self, workflow, scheduler):
        """Simulate workflow execution with given scheduler"""
        # TODO: Implement WRENCH simulation
        pass
''',
            
            "ml/gnn/graph_encoder.py": '''"""
Graph Neural Network Encoder

Implements GNN-based encoding of workflow and cluster state.
"""

import torch
import torch.nn as nn

class WorkflowGraphEncoder(nn.Module):
    """GNN encoder for workflow state"""
    
    def __init__(self, config):
        super().__init__()
        # TODO: Implement GNN architecture
        
    def forward(self, graph):
        """Encode workflow graph to embedding"""
        # TODO: Implement forward pass
        pass
''',

            "ml/drl/ppo_agent.py": '''"""
PPO Agent Implementation

Deep Reinforcement Learning agent using Proximal Policy Optimization.
"""

import torch
import torch.nn as nn

class PPOAgent:
    """PPO agent for workflow scheduling"""
    
    def __init__(self, config):
        self.config = config
        # TODO: Initialize PPO components
        
    def select_action(self, state):
        """Select scheduling action given current state"""
        # TODO: Implement action selection
        pass
        
    def train_step(self, experiences):
        """Perform one training step"""
        # TODO: Implement PPO training
        pass
''',

            "ml/rag/knowledge_base.py": '''"""
Knowledge Base and RAG Implementation

Retrieval-Augmented Generation for informed reward signals.
"""

import faiss
import numpy as np

class KnowledgeBase:
    """Vector knowledge base for historical experiences"""
    
    def __init__(self, config):
        self.config = config
        # TODO: Initialize FAISS index
        
    def add_experience(self, workflow_graph, actions, performance):
        """Add new experience to knowledge base"""
        # TODO: Implement experience storage
        pass
        
    def retrieve_similar(self, workflow_graph, k=5):
        """Retrieve k most similar historical experiences"""
        # TODO: Implement similarity search
        pass
''',

            "experiments/basic_simulation.py": '''"""
Basic WRENCH Simulation Experiment

Simple experiment to verify WRENCH integration.
"""

def main():
    """Run basic simulation experiment"""
    print("🧪 Running basic WRENCH simulation...")
    
    # TODO: Implement basic experiment
    print("⚠️  Not implemented yet - placeholder")
    
if __name__ == "__main__":
    main()
''',

            "docs/academic/wrench_setup.md": '''# WRENCH Setup Guide

## Installation

### Prerequisites
- CMake 3.10+
- GCC 7+ or Clang 6+
- Boost libraries
- SimGrid 3.25+

### Building WRENCH
```bash
# Clone WRENCH
git clone https://github.com/wrench-project/wrench.git
cd wrench

# Build
mkdir build && cd build
cmake ..
make -j4

# Install
sudo make install
```

### Python Bindings
```bash
pip install wrench-python-api
```

## Integration with WASS-RAG

[TODO: Add integration instructions]
''',

            "tests/test_wrench_integration.py": '''"""
Tests for WRENCH integration
"""

import pytest

class TestWRENCHIntegration:
    """Test WRENCH simulator integration"""
    
    def test_simulator_initialization(self):
        """Test basic simulator setup"""
        # TODO: Implement test
        pass
        
    def test_workflow_simulation(self):
        """Test workflow simulation"""
        # TODO: Implement test  
        pass
'''
        }
        
        for file_path, content in placeholders.items():
            full_path = self.root / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            print(f"  Created: {file_path}")
    
    def run_reorganization(self):
        """Run complete project reorganization"""
        print("🚀 Starting WASS-RAG Academic Research Reorganization")
        print("=" * 60)
        
        self.create_backup()
        self.clean_outdated_files()
        self.reorganize_scripts()
        self.create_new_structure()
        self.update_configs()
        self.clean_src_directory()
        self.create_academic_readme()
        self.update_requirements()
        self.create_placeholder_files()
        
        print("\n" + "=" * 60)
        print("✅ Project reorganization completed!")
        print(f"📦 Concept proof backed up to: {self.backup_dir}")
        print("🎯 Ready for Level 2: High-Fidelity Simulation development")
        print("\nNext steps:")
        print("1. Review ACADEMIC_ROADMAP.md")
        print("2. Install WRENCH (see docs/academic/wrench_setup.md)")
        print("3. Start with experiments/basic_simulation.py")

def main():
    """Main execution function"""
    reorganizer = ProjectReorganizer()
    reorganizer.run_reorganization()

if __name__ == "__main__":
    main()
