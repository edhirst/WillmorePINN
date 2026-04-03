# Willmore Energy Minimization with Neural Networks

Learn optimal surface embeddings φ:(u,v)→(x,y,z) that minimize the Willmore energy functional ∫∫ H² dA using PyTorch.

## Quick Start

```bash
# Setup (see environment/README.md for details)
conda create -n willmore python=3.10
conda activate willmore
pip install -r environment/requirements.txt

# Train (defaults to configs/config_genus2.yaml)
python run.py

# Train with a specific config
python run.py --config configs/config_genus1.yaml

# Visualize training evolution
python visualisation/visualise.py

# Visualize analytical reference surfaces
python visualisation/visualise_analytic.py --genus 1

# Visualize supervised pretraining results
python visualisation/visualise_supervised.py --genus 1
```

## Core Concept

The network learns a surface embedding φ:(u,v)→(x,y,z), minimizing:

$$W = \int\int H^2 \sqrt{EG - F^2} \, du \, dv$$

where H is mean curvature computed via automatic differentiation of the embedding.

## Usage

```bash
# Resume from a checkpoint
python run.py --config configs/config_genus1.yaml --resume checkpoints/run_1/latest_model.pt

# Visualize with both surface and loss plots
python visualisation/visualise.py --mode both

# Visualize analytical surfaces (genus 0, 1, or 2)
python visualisation/visualise_analytic.py --genus 1

# Supervised pretraining visualization
python visualisation/visualise_supervised.py --genus 1 --points 20000
```

Config files are in `configs/`. The `--config` flag accepts any path; it defaults to `configs/config_genus2.yaml`.

## Files

- **`model.py`** - EmbeddingNetwork: φ(u,v)→(x,y,z) with Fourier features for periodicity
- **`losses.py`** - Willmore energy ∫∫H²dA computed via autodiff of fundamental forms
- **`sampling.py`** - Parameter space sampling for torus, sphere, double torus, etc.
- **`run.py`** - Training loop with checkpointing
- **`utils.py`** - Visualization utilities and run management
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

**Known minima**:
- Genus 0: round sphere, W = 4π ≈ 12.566
- Genus 1: Clifford torus, W = 2π² ≈ 19.74
- Genus 2: Lawson surface ξ_{2,1} (conjectured), W = 4π² ≈ 39.48

## Output

Training creates:
- `checkpoints/run_N/` - Model states (`best_model.pt`, `latest_model.pt`, epoch checkpoints)
- `logs/run_N/training_history.json` - Loss curves and metrics
- `logs/run_N/loss_curves.png` - Automatically generated after training