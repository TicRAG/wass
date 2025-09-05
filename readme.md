# WASS-RAG: Academic Research Implementation

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
# or wass_env\Scripts\activate  # Windows

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
