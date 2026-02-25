
"""
Train and visualise surface embeddings after supervised pretraining for different genus values.
"""

# =====================
# Hyperparameters
# =====================
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_POINTS_PER_EPOCH = 5000
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 1e-3
FIGSIZE = (15, 10)

# =====================
# Imports and Setup
# =====================

import sys
import os
# Ensure parent directory is in sys.path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import copy
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from model import create_embedding_model
from sampling import sample_parameters, get_domain_for_genus, get_reference_embedding
from utils import get_next_run_number, plot_fundamental_domain_coloring, hsv_to_rgb_colors, plot_surface_3d

def train_surface(model, optimizer, uv, xyz_target, num_epochs):
    """Train model(uv) to xyz_target for num_epochs."""
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(uv)
        loss = torch.nn.functional.mse_loss(pred, xyz_target)
        loss.backward()
        optimizer.step()
    return model


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a trained model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    label = checkpoint.get('label', os.path.basename(checkpoint_path))
    
    model = create_embedding_model(config, device, skip_init=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, label


def visualise_supervised_genus0(output_dir: str, num_points: int, config_path: str, device: torch.device):
    """Train and visualize supervised ellipsoid embeddings (genus 0)."""
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Quantitative labels based on hyperparams
    ellipsoid_configs = [
        {'a': 1.0, 'b': 1.0, 'c': 1.0, 'label': 'a=1, b=1, c=1'},
        {'a': 2.0, 'b': 1.0, 'c': 1.0, 'label': 'a=2, b=1, c=1'},
        {'a': 2.0, 'b': 2.0, 'c': 1.0, 'label': 'a=2, b=2, c=1'},
        {'a': 1.5, 'b': 1.0, 'c': 0.5, 'label': 'a=1.5, b=1, c=0.5'},
        {'a': 1.2, 'b': 0.6, 'c': 0.3, 'label': 'a=1.2, b=0.6, c=0.3'},
    ]
    
    model_paths = []
    for cfg in ellipsoid_configs:
        config = copy.deepcopy(base_config)
        config['topology'] = config.get('topology', {})
        config['topology']['genus'] = 0
        config['topology']['ellipsoid'] = {'a': cfg['a'], 'b': cfg['b'], 'c': cfg['c']}
        config['model']['supervised_pretraining']['num_epochs'] = DEFAULT_NUM_EPOCHS
        config['model']['supervised_pretraining']['num_points_per_epoch'] = DEFAULT_NUM_POINTS_PER_EPOCH
        config['model']['supervised_pretraining']['batch_size'] = DEFAULT_BATCH_SIZE
        model = create_embedding_model(config, device, skip_init=False)
        uv = sample_parameters(num_points, domain="ellipsoid", device=device, dtype=torch.float32)
        with torch.no_grad():
            xyz_target = get_reference_embedding(uv, genus=0, topology_params={'ellipsoid': cfg})
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
        model = train_surface(model, optimizer, uv, xyz_target, DEFAULT_NUM_EPOCHS)
        safe_label = cfg['label'].replace(" ", "_").replace("=", "_").replace("+", "_").replace(".", "_").replace(",", "_")
        model_path = os.path.join(output_dir, f'model_genus0_{safe_label}.pt')
        torch.save({'model_state_dict': model.state_dict(), 'config': config, 'label': cfg['label']}, model_path)
        model_paths.append((model_path, cfg['label'], uv))
    
    # Visualize with fundamental domain
    n_models = len(model_paths)
    n_plots = n_models + 1  # +1 for domain plot
    n_rows = int(np.ceil(n_plots / 2))
    fig = plt.figure(figsize=(12, 5 * n_rows))
    
    # Add fundamental domain coloring first
    ax_domain = fig.add_subplot(n_rows, 2, 1)
    plot_fundamental_domain_coloring(ax_domain, genus=0)
    
    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}

    # First pass: collect xyz and metadata
    _plot_data = []
    for model_path, label, uv in model_paths:
        model, config, _ = load_model_from_checkpoint(model_path, device)
        genus = config['topology']['genus']
        domain = get_domain_for_genus(genus)
        print(f"Visualizing model from {model_path}")
        print(f"Embedding model created for genus {genus} ({genus_names.get(genus, 'unknown')})")
        print(f"  Domain: {domain}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        with torch.no_grad():
            xyz_pred = model(uv).cpu()
        _plot_data.append((xyz_pred, uv, label, genus))

    # Shared scale across all subplots
    _all_xyz = np.concatenate([d[0].numpy() for d in _plot_data])
    _global_range = np.array([
        _all_xyz[:, 0].max() - _all_xyz[:, 0].min(),
        _all_xyz[:, 1].max() - _all_xyz[:, 1].min(),
        _all_xyz[:, 2].max() - _all_xyz[:, 2].min()
    ]).max() / 2.0

    for idx, (xyz_pred, uv, label, genus) in enumerate(_plot_data):
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz_pred, uv, label, genus=genus, global_range=_global_range)
        ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'supervised_genus0.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved genus 0 comparison to {output_path}")
    plt.close()


def visualise_supervised_genus1(output_dir: str, num_points: int, config_path: str, device: torch.device):
    """Train and visualize supervised torus embeddings (genus 1)."""
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    tau_values = [
        ("1j", "τ = 1.0i"),
        ("0.5j", "τ = 0.5i"),
        ("1.0+0.5j", "τ = 1.0+0.5i"),
        ("0.2+0.2j", "τ = 0.2+0.2i"),
        ("1.0+0.2j", "τ = 1.0+0.2i"),
    ]
    
    model_paths = []
    from sampling import transform_square_to_parallelogram, sample_rectangular_domain
    for tau_str, label in tau_values:
        config = copy.deepcopy(base_config)
        config['topology'] = config.get('topology', {})
        config['topology']['genus'] = 1
        config['topology']['torus'] = config['topology'].get('torus', {})
        config['topology']['torus']['tau'] = tau_str
        config['model']['supervised_pretraining']['num_epochs'] = DEFAULT_NUM_EPOCHS
        config['model']['supervised_pretraining']['num_points_per_epoch'] = DEFAULT_NUM_POINTS_PER_EPOCH
        config['model']['supervised_pretraining']['batch_size'] = DEFAULT_BATCH_SIZE
        tau = complex(tau_str.replace('i', 'j'))
        uv = sample_rectangular_domain(num_points, (0, 2*np.pi), (0, 2*np.pi), device)
        model = create_embedding_model(config, device, skip_init=False)
        with torch.no_grad():
            xyz_target = get_reference_embedding(uv, domain="torus", tau=tau)
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
        model = train_surface(model, optimizer, uv, xyz_target, DEFAULT_NUM_EPOCHS)
        safe_label = label.replace(" ", "_").replace("=", "_").replace("+", "_").replace(".", "_").replace(",", "_")
        model_path = os.path.join(output_dir, f'model_genus1_{safe_label}.pt')
        torch.save({'model_state_dict': model.state_dict(), 'config': config, 'label': label}, model_path)
        model_paths.append((model_path, label, uv))
    
    # Visualize with fundamental domain
    n_tau = len(tau_values)
    n_plots = n_tau + 1  # +1 for domain plot
    n_rows = int(np.ceil(n_plots / 2))
    fig = plt.figure(figsize=(12, 5 * n_rows))

    ax_domain = fig.add_subplot(n_rows, 2, 1)
    # Use tau=i for the domain plot to match visualise_analytic.py
    tau = 1j
    plot_fundamental_domain_coloring(ax_domain, genus=1, tau1=tau)

    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}

    # First pass: collect xyz and metadata
    _plot_data = []
    for model_path, label, uv in model_paths:
        model, config, _ = load_model_from_checkpoint(model_path, device)
        genus = config['topology']['genus']
        domain = get_domain_for_genus(genus)
        print(f"Visualizing model from {model_path}")
        print(f"Embedding model created for genus {genus} ({genus_names.get(genus, 'unknown')})")
        print(f"  Domain: {domain}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        with torch.no_grad():
            xyz_pred = model(uv).cpu()
        _plot_data.append((xyz_pred, uv, label, genus))

    # Shared scale across all subplots
    _all_xyz = np.concatenate([d[0].numpy() for d in _plot_data])
    _global_range = np.array([
        _all_xyz[:, 0].max() - _all_xyz[:, 0].min(),
        _all_xyz[:, 1].max() - _all_xyz[:, 1].min(),
        _all_xyz[:, 2].max() - _all_xyz[:, 2].min()
    ]).max() / 2.0

    for idx, (xyz_pred, uv, label, genus) in enumerate(_plot_data):
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz_pred, uv, label, genus=genus, global_range=_global_range)
        ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'supervised_genus1.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved genus 1 comparison to {output_path}")
    plt.close()


def visualise_supervised_genus2(output_dir: str, num_points: int, config_path: str, device: torch.device):
    """Train and visualize supervised double torus embeddings (genus 2).
    
    Uses Fenchel-Nielsen coordinates:
    - l₁, l₂, l₃ > 0: lengths of the 3 gluing geodesics
    - τ₁, τ₂, τ₃ ∈ ℝ: twist parameters
    
    The fundamental domain is 4 right-angled hexagons (2 per pair of pants).
    """
    # Use the same configs as analytic for demonstration
    # Use the same configs as analytic script
    dt_configs = [
        {'tau1': {'real': 0.0, 'imag': 1.0}, 'tau2': {'real': 0.0, 'imag': 1.0}, 'bridge_radius': 0.25, 'neck_twist': 0.0, 'scale': 1.2, 'label': 'τ₁=τ₂=i (symmetric)'},
        {'tau1': {'real': 0.0, 'imag': 0.5}, 'tau2': {'real': 0.0, 'imag': 2.0}, 'bridge_radius': 0.2, 'neck_twist': 0.0, 'scale': 1.2, 'label': 'τ₁=0.5i (thick), τ₂=2i (thin)'},
        {'tau1': {'real': 0.5, 'imag': 1.0}, 'tau2': {'real': -0.5, 'imag': 1.0}, 'bridge_radius': 0.25, 'neck_twist': 0.0, 'scale': 1.2, 'label': 'τ₁=0.5+i, τ₂=-0.5+i (twisted)'},
        {'tau1': {'real': 1.0, 'imag': 0.8}, 'tau2': {'real': 0.0, 'imag': 1.2}, 'bridge_radius': 0.2, 'neck_twist': 0.0, 'scale': 1.2, 'label': 'τ₁=1+0.8i (strong twist)'},
        {'tau1': {'real': 0.0, 'imag': 0.7}, 'tau2': {'real': 0.0, 'imag': 0.7}, 'bridge_radius': 0.12, 'neck_twist': 0.0, 'scale': 1.2, 'label': 'bridge_radius=0.12 (narrow)'},
    ]
    model_paths = []
    for cfg in dt_configs:
        label = cfg['label']
        topology_params = {'double_torus': {k: v for k, v in cfg.items() if k != 'label'}}
        uv = sample_parameters(num_points, domain="double_torus", device=device, dtype=torch.float32)
        with torch.no_grad():
            xyz_target = get_reference_embedding(uv, genus=2, topology_params=topology_params)
        config = yaml.safe_load(open(config_path, 'r'))
        config['topology'] = config.get('topology', {})
        config['topology']['genus'] = 2
        config['topology']['double_torus'] = {k: v for k, v in cfg.items() if k != 'label'}
        config['model']['supervised_pretraining']['num_epochs'] = DEFAULT_NUM_EPOCHS
        config['model']['supervised_pretraining']['num_points_per_epoch'] = DEFAULT_NUM_POINTS_PER_EPOCH
        config['model']['supervised_pretraining']['batch_size'] = DEFAULT_BATCH_SIZE
        model = create_embedding_model(config, device, skip_init=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
        model = train_surface(model, optimizer, uv, xyz_target, DEFAULT_NUM_EPOCHS)
        safe_label = label.replace(" ", "_").replace("=", "_").replace("+", "_").replace(".", "_").replace(",", "_")
        model_path = os.path.join(output_dir, f'model_genus2_{safe_label}.pt')
        torch.save({'model_state_dict': model.state_dict(), 'config': config, 'label': label}, model_path)
        model_paths.append((model_path, label, uv))
    # Visualize with fundamental domain
    n_models = len(model_paths)
    n_plots = n_models + 1  # +1 for domain plot
    n_rows = int(np.ceil(n_plots / 2))
    fig = plt.figure(figsize=(12, 5 * n_rows))
    # Add fundamental domain coloring first
    ax_domain = fig.add_subplot(n_rows, 2, 1)
    first_cfg = dt_configs[0]
    tau1 = complex(first_cfg['tau1']['real'], first_cfg['tau1']['imag'])
    tau2 = complex(first_cfg['tau2']['real'], first_cfg['tau2']['imag'])
    plot_fundamental_domain_coloring(ax_domain, genus=2, tau1=tau1, tau2=tau2, neck_radius=0.3)
    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}

    # First pass: collect xyz and metadata
    _plot_data = []
    for model_path, label, uv in model_paths:
        model, config, _ = load_model_from_checkpoint(model_path, device)
        genus = config['topology']['genus']
        domain = get_domain_for_genus(genus)
        print(f"Visualizing model from {model_path}")
        print(f"Embedding model created for genus {genus} ({genus_names.get(genus, 'unknown')})")
        print(f"  Domain: {domain}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        with torch.no_grad():
            xyz_pred = model(uv).cpu()
        _plot_data.append((xyz_pred, uv, label, genus))

    # Shared scale across all subplots
    _all_xyz = np.concatenate([d[0].numpy() for d in _plot_data])
    _global_range = np.array([
        _all_xyz[:, 0].max() - _all_xyz[:, 0].min(),
        _all_xyz[:, 1].max() - _all_xyz[:, 1].min(),
        _all_xyz[:, 2].max() - _all_xyz[:, 2].min()
    ]).max() / 2.0

    for idx, (xyz_pred, uv, label, genus) in enumerate(_plot_data):
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz_pred, uv, label, genus=genus, global_range=_global_range)
        ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'supervised_genus2.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved genus 2 comparison to {output_path}")
    plt.close()


def main():
    """Main function to train and visualise supervised embeddings."""
    import argparse
    
    # Default config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, 'hyperparameters.yaml')
    
    parser = argparse.ArgumentParser(description="Train and visualize supervised surface embeddings")
    parser.add_argument('--config', type=str, default=default_config,
                       help='Path to config file')
    parser.add_argument('--points', type=int, default=20000,
                       help='Number of points for visualization')
    parser.add_argument('--genus', type=int, default=None, choices=[0, 1, 2],
                       help='Genus to train/visualize (0, 1, or 2). If not specified, do all.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generate logs/supervised_run_#)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine output directory
    if args.output_dir is None:
        base_dir = os.path.join(project_root, 'logs')
        run_num = get_next_run_number(base_dir, "supervised_run_")
        output_dir = os.path.join(base_dir, f"supervised_run_{run_num}")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SUPERVISED PRETRAINING VISUALISATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Points for visualization: {args.points}")
    
    genera_to_train = [args.genus] if args.genus is not None else [0, 1, 2]
    
    for genus in genera_to_train:
        print(f"\n{'='*60}")
        print(f"Training and visualizing Genus {genus} surfaces...")
        print(f"{'='*60}")
        
        if genus == 0:
            visualise_supervised_genus0(output_dir, args.points, args.config, device)
        elif genus == 1:
            visualise_supervised_genus1(output_dir, args.points, args.config, device)
        elif genus == 2:
            visualise_supervised_genus2(output_dir, args.points, args.config, device)
    
    print(f"\n{'='*60}")
    print("TRAINING AND VISUALISATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
