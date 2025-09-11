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

## 🔗 Related Work

- [WRENCH](https://github.com/wrench-project/wrench): Workflow simulation framework
- [FAISS](https://github.com/facebookresearch/faiss): Vector similarity search
- [PyTorch](https://pytorch.org/): Deep learning framework
