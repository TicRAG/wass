# WRENCH Setup Guide for WASS-RAG

## Overview

WRENCH (Workflow Execution and Resource Noise Characterization) is a simulation framework for scientific workflows on distributed platforms. This guide covers installation and setup for WASS-RAG integration.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Compiler**: g++ 9+ or clang++ 10+
- **Python**: 3.8+
- **Memory**: 8GB+ RAM recommended
- **Disk**: 5GB+ free space

### Required Dependencies

#### Core Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake git
sudo apt install -y libboost-all-dev
sudo apt install -y python3-dev python3-pip

# macOS (with Homebrew)
brew install cmake boost python@3.9

# Windows (WSL2 Ubuntu)
# Follow Ubuntu instructions above
```

#### SimGrid Installation
WRENCH depends on SimGrid for the underlying simulation engine.

```bash
# Download SimGrid
wget https://gitlab.inria.fr/simgrid/simgrid/-/archive/v3.32/simgrid-v3.32.tar.gz
tar -xzf simgrid-v3.32.tar.gz
cd simgrid-v3.32

# Configure and build
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install

# Verify installation
simgrid_update_xml --version
```

## WRENCH Installation

### Method 1: From Source (Recommended for Development)

```bash
# Clone WRENCH repository
git clone https://github.com/wrench-project/wrench.git
cd wrench

# Create build directory
mkdir build && cd build

# Configure with Python bindings
cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DENABLE_BATSCHED=OFF \
      -DENABLE_PYTHON=ON \
      ..

# Build (this may take 20-30 minutes)
make -j$(nproc)

# Install
sudo make install

# Update library path
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Method 2: Using Conda (Alternative)

```bash
# Install via conda-forge (if available)
conda install -c conda-forge wrench-project

# Or create conda environment
conda create -n wrench python=3.9
conda activate wrench
# Then follow source installation in this environment
```

## Python Bindings Setup

### Install Python Package

```bash
# Navigate to WRENCH Python directory
cd /path/to/wrench/python

# Install in development mode
pip install -e .

# Or install from PyPI (if available)
pip install wrench-project
```

### Verify Python Installation

```python
# Test in Python interpreter
import wrench
print(wrench.__version__)

# Run basic test
from wrench import Simulation
sim = Simulation()
print("WRENCH Python bindings working!")
```

## Configuration for WASS

### Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# WRENCH environment
export WRENCH_ROOT=/usr/local
export LD_LIBRARY_PATH=$WRENCH_ROOT/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$WRENCH_ROOT/lib/python3.x/site-packages:$PYTHONPATH

# SimGrid configuration
export SIMGRID_PATH=/usr/local
```

### Python Virtual Environment

Create dedicated environment for WASS-WRENCH development:

```bash
# Create virtual environment
python -m venv venv_wass_wrench
source venv_wass_wrench/bin/activate

# Install required packages
pip install -r requirements_wrench.txt
```

## Verification Tests

### Test 1: Basic WRENCH Example

Create `test_wrench.py`:

```python
#!/usr/bin/env python3

import wrench
from wrench import Simulation

def test_basic_simulation():
    """Test basic WRENCH simulation functionality"""
    try:
        # Create simulation
        simulation = Simulation()
        
        # Add a simple platform
        simulation.add_platform("simple_platform.xml")
        
        # Create and start simulation
        simulation.start()
        print("✅ Basic WRENCH simulation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic simulation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_simulation()
    exit(0 if success else 1)
```

### Test 2: Platform Configuration

Create simple platform file `simple_platform.xml`:

```xml
<?xml version='1.0'?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="host1" speed="1Gf" core="4"/>
    <host id="host2" speed="1Gf" core="4"/>
    <link id="link1" bandwidth="1GBps" latency="0.001s"/>
    <route src="host1" dst="host2">
      <link_ctn id="link1"/>
    </route>
  </zone>
</platform>
```

### Test 3: Workflow Simulation

Create `test_workflow.py`:

```python
#!/usr/bin/env python3

import wrench

def test_workflow_simulation():
    """Test workflow simulation with WRENCH"""
    try:
        # Create simulation
        simulation = wrench.Simulation()
        
        # TODO: Add workflow creation and execution test
        # This will be implemented in Phase 1
        
        print("✅ Workflow simulation framework ready!")
        return True
        
    except Exception as e:
        print(f"❌ Workflow simulation test failed: {e}")
        return False

if __name__ == "__main__":
    test_workflow_simulation()
```

## Troubleshooting

### Common Issues

#### 1. SimGrid Not Found
```bash
# Error: Could not find SimGrid
# Solution: Ensure SimGrid is properly installed and in PATH
export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
```

#### 2. Python Bindings Import Error
```bash
# Error: ImportError: No module named 'wrench'
# Solution: Check PYTHONPATH and installation
python -c "import sys; print(sys.path)"
```

#### 3. Compilation Errors
```bash
# Error: Boost libraries not found
# Solution: Install development headers
sudo apt install libboost-all-dev

# Error: C++14 required
# Solution: Use newer compiler
export CXX=g++-9
```

## Unified Workflow Generation (WASS Paper Experiments)

To ensure fair comparison between training (PPO+RAG) and evaluation baselines (FIFO / HEFT / WASS-*), a shared workflow generator was introduced.

Key points:

- Module: `src/workflow_generator_shared.py`
- Used by: `scripts/train_wass_paper_aligned.py` and `scripts/evaluate_paper_methods.py`
- Parameters unified:
  - FLOPS range: 1e9 – 12e9
  - File size range: 5 KB – 50 KB
  - Dependency probability: 0.35
- Eliminates distribution shift observed earlier (training used small random size 10–20, evaluation used fixed larger set 5..80 with different flops & dep probability).

Baseline Adjustments:

- `HEFTScheduler` adapter now estimates earliest finish time (EFT = current load proxy + exec_time) instead of a crude capacity heuristic.
- `WASS-Heuristic`, `WASS-DRL` adapter placeholders remain simplified; future work can plug in full historical logic from `legacy_archived` if needed.

Re-running Evaluation:

```bash
python scripts/evaluate_paper_methods.py configs/experiment.yaml
```

If you retrain PPO after these changes, increase episodes (e.g. 300–500) and consider a rag_weight schedule (warmup) to avoid early over-reliance on noisy RAG reward.

Planned Next Improvements:

- Add critical-path approximation to dense env reward.
- Calibrate RAG reward scale versus normalized makespan deltas.
- Introduce curriculum over workflow sizes (progressively larger DAGs).

This section was added after refactor commit introducing the shared generator so older results are not directly comparable.

#### 4. Runtime Library Errors
```bash
# Error: shared library not found
# Solution: Update library path
sudo ldconfig
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Performance Tuning

#### Memory Settings
```bash
# For large simulations
export SIMGRID_CONTEXT_STACK_SIZE=8192
```

#### Logging Configuration
```bash
# Control SimGrid logging
export SIMGRID_LOG_THRESHOLD=warning
```

## Resources

- [WRENCH Documentation](https://wrench-project.org/wrench/latest/)
- [SimGrid Documentation](https://simgrid.org/doc/latest/)
- [WRENCH GitHub Repository](https://github.com/wrench-project/wrench)
- [WRENCH Tutorials](https://wrench-project.org/wrench/latest/quickstart.html)

---

**Status**: Ready for Phase 1 implementation  
**Next**: Begin WRENCH installation and verification  
**Contact**: Check issues in GitHub or WRENCH community forums
