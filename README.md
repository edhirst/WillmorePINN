# Willmore Energy Minimization with Neural Networks

Learn optimal surface embeddings φ:(u,v)→(x,y,z) that minimize the Willmore energy functional ∫∫ H² dA using PyTorch.

## Quick Start

```bash
# Setup (see environment/README.md for details)
conda create -n willmore python=3.10
conda activate willmore
pip install -r environment/requirements.txt

# Train
python run.py

# Visualize training evolution
python visualisation/visualise.py

# Visualize analytical reference surfaces
python visualisation/visualise_analytic.py --genus 1

# Visualize supervised pretraining results
python visualisation/visualise_supervised.py --genus 1
```

## Core Concept

The network learns **residual corrections** to reference surfaces, minimizing:

$$W = \int\int H^2 \sqrt{EG - F^2} \, du \, dv$$

where H is mean curvature computed via automatic differentiation of the embedding.

## Usage

```bash
# Basic training
python run.py

# Visualize training evolution
python visualisation/visualise.py --mode both

# Visualize analytical surfaces (genus 0, 1, or 2)
python visualisation/visualise_analytic.py --genus 1

# Train and visualize supervised pretraining
python visualisation/visualise_supervised.py --genus 1 --points 20000

# Analysis
python utils.py --checkpoint checkpoints/run_1/best_model.pt
```

Edit `hyperparameters.yaml` to configure model architecture, training parameters, topology (genus), and domain settings.

## Files

- **`model.py`** - EmbeddingNetwork: φ(u,v)→(x,y,z) with Fourier features for periodicity
- **`losses.py`** - Willmore energy ∫∫H²dA computed via autodiff of fundamental forms
- **`sampling.py`** - Parameter space sampling for torus, sphere, double torus, etc.
- **`run.py`** - Training loop with checkpointing
- **`utils.py`** - Model analysis, curvature statistics, and utility functions
- **`visualisation/`** - Visualization scripts:
  - `visualise.py` - Training evolution visualization
  - `visualise_analytic.py` - Analytical reference surface visualization
  - `visualise_supervised.py` - Supervised pretraining visualization

## Theory

**Willmore Energy**: W = ∫∫ H² √(EG-F²) du dv

where H = (EN-2FM+GL)/(2(EG-F²)) is mean curvature, computed from:
- First fundamental form: E = ⟨φ_u,φ_u⟩, F = ⟨φ_u,φ_v⟩, G = ⟨φ_v,φ_v⟩
- Second fundamental form: L = ⟨φ_uu,n⟩, M = ⟨φ_uv,n⟩, N = ⟨φ_vv,n⟩
- Unit normal: n = (φ_u × φ_v) / |φ_u × φ_v|

**Key Results**:
- Clifford torus (R=r=√2): W = 2π² ≈ 19.74 (optimal)
- Standard torus (R=2, r=1): W ≈ 49.35
- This package achieves W ≈ 21 (7% above optimal) via gradient descent

## Output

Training creates:
- `logs/checkpoints/` - Model states (best_model.pt, latest_model.pt, epoch checkpoints)
- `logs/training_history.json` - Loss curves and metrics
- `logs/*.png` - Visualisations (when running visualise.py)