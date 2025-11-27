# Environment Setup

## Prerequisites

- Python 3.8 or higher
- pip or conda package manager

## Installation Options

### Option 1: Conda Environment (Recommended)

```bash
# Create environment
conda create -n willmore python=3.10

# Activate environment
conda activate willmore

# Install PyTorch (CPU version)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

The `requirements.txt` includes:
- **PyTorch** ≥2.0.0 - Deep learning framework with automatic differentiation
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **PyYAML** - Configuration file parsing

## Verification

Test your installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import yaml; import numpy; import matplotlib; print('✓ All dependencies installed')"
```

## Device Support

The package automatically detects available compute devices:
- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (M1/M2/M3)
- **CPU**: Fallback (recommended for this application)

Note: MPS backend has incomplete support for some linear algebra operations needed for second derivatives. CPU is recommended.

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`, ensure the environment is activated:
```bash
conda activate willmore  # or source venv/bin/activate
```

### PyTorch Installation
For GPU support or specific CUDA versions, see: https://pytorch.org/get-started/locally/

### M1/M2 Mac Issues
If MPS causes errors, force CPU mode in `hyperparameters.yaml`:
```yaml
device: "cpu"
```
