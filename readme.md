# WASS-RAG: Workflow-Aware Scheduling System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WASS-RAG** is an intelligent workflow scheduling system that combines **Deep Reinforcement Learning (DRL)** with **Retrieval-Augmented Generation (RAG)** to make data-driven scheduling decisions based on historical experience.

## 🎯 Key Features

- 🧠 **AI-Enhanced Scheduling**: Deep reinforcement learning with historical knowledge retrieval
- 📚 **RAG Knowledge Base**: FAISS-powered vector database with 40,000+ scheduling cases  
- 🔬 **Academic Research Ready**: Complete experimental framework with baseline comparisons
- ⚡ **Production-Grade**: Optimized performance predictor with real-time inference
- 🔍 **Explainable Decisions**: Transparent reasoning with historical case justification

## 🚀 Quick Start

### Prerequisites
```bash
# Required dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#pip install torch torchvision torchaudio
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install -r requirements.txt
```

### 3. Run Experiments
```bash
python scripts/generate_kb_dataset.py
python scripts/train_predictor_from_kb.py
python scripts/train_drl_agent.py
python experiments/real_experiment_framework.py

```

## 📁 Project Structure

```
wass/
├── src/                           # Core system implementation
│   ├── ai_schedulers.py          # Main AI scheduling logic
│   ├── interfaces.py             # System interfaces
│   ├── config_loader.py          # Configuration management
│   ├── factory.py                # Component factory
│   └── utils.py                  # Utility functions
├── scripts/                      # Training and setup scripts
│   └── retrain_performance_predictor.py
├── experiments/                  # Research experiments
│   ├── real_experiment_framework.py
│   └── run_pipeline.py
├── configs/                      # YAML configuration files
│   ├── academic.yaml
│   ├── rag.yaml
│   └── experiment.yaml
├── results/                      # Experimental results
└── test_predictions.py          # System validation
```

## 🔬 System Architecture

### Core Components

1. **Performance Predictor**: Deep neural network predicting task execution times
2. **RAG Knowledge Base**: Vector database storing historical scheduling decisions  
3. **Policy Network**: Reinforcement learning agent for decision making
4. **Feature Engineering**: 96-dimensional feature vectors capturing task-node interactions

### AI Pipeline Flow

```
Scheduling Request → Feature Extraction → RAG Retrieval → Performance Prediction → Decision
```

## 📊 Experimental Results

### Performance Metrics
- **Model Accuracy**: R² = 0.9791, MSE = 0.24
- **Knowledge Base**: 40,571 historical cases
- **Prediction Range**: 1-180 seconds execution time
- **Feature Dimensionality**: 96D with task-node interaction features

### Baseline Comparisons
- Heuristic algorithms (FIFO, SJF, LJF)
- Traditional ML approaches
- Pure DRL without RAG enhancement

## 🛠️ Configuration

### Model Settings
```yaml
# configs/rag.yaml
knowledge_base:
  vector_dim: 32
  similarity_threshold: 0.7
  top_k_retrieval: 5

performance_predictor:
  input_dim: 96
  hidden_dim: 128
  learning_rate: 0.001
```

### Feature Engineering
The system uses 14 core interaction features:
- CPU/Memory matching scores
- Performance compatibility metrics  
- Data locality indicators
- Load balancing factors

## � Usage Examples

### Basic Scheduling
```python
from src.ai_schedulers import WASSRAGScheduler

# Initialize scheduler (models will be created/loaded automatically)
scheduler = WASSRAGScheduler()

# Make scheduling decision
decision = scheduler.schedule(workflow_state, cluster_state)
print(f"Assigned {decision.task} to {decision.node}")
```

### Custom Training
```python
# Generate training data and retrain models
from scripts.retrain_performance_predictor import main
main()  # Trains with 5000 synthetic scenarios
```

## 🔧 Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed
```bash
pip install torch faiss-cpu numpy
```

**Model Loading**: Train models if they don't exist
```bash
python scripts/retrain_performance_predictor.py
```

**Performance Issues**: Use GPU acceleration
```bash
pip uninstall faiss-cpu && pip install faiss-gpu
```

## 📝 Citation

```bibtex
@article{wass-rag-2025,
  title={WASS-RAG: Workflow-Aware Scheduling with Retrieval-Augmented Generation},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔗 Related Work

- [WRENCH](https://github.com/wrench-project/wrench): Workflow simulation framework
- [FAISS](https://github.com/facebookresearch/faiss): Vector similarity search
- [PyTorch](https://pytorch.org/): Deep learning framework

---

**Built with ❤️ for the scientific computing community**
cat results/real_experiments/paper_tables.json

# Detailed analysis
cat results/real_experiments/experiment_analysis.json
```

## 🎯 Core Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `wass_wrench_simulator.py` | Real WRENCH integration | ✅ Production |
| `wass_academic_platform.py` | Academic workflow management | ✅ Production |
| `experiments/real_experiment_framework.py` | Paper data collection | ✅ Ready |

## 📈 Experimental Results

Current validated performance:
- **Execution Time**: 23.1s (real WRENCH simulation)
- **Throughput**: 1.82 GFlops/s  
- **System Efficiency**: 76.5%
- **CPU Utilization**: 85%
- **Data Source**: Real WRENCH 0.3-dev
- **Reproducibility**: High

## 🔬 Research Applications

Perfect for academic research in:
- **Workflow Scheduling**: Real task dependencies and resource allocation
- **Distributed Systems**: Multi-host performance analysis  
---

**Built with ❤️ for the scientific computing community**
*This is the academic research version focusing on high-fidelity simulation.*
