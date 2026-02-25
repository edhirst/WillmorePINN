"""
Visualize the true (analytical/untrained) surface embeddings for different genus values and parameters.
This helps verify that the reference embeddings are smooth and correctly defined.
"""
import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml

from sampling import get_reference_embedding, sample_parameters, get_domain_for_genus
from utils import get_next_run_number, plot_fundamental_domain_coloring, hsv_to_rgb_colors, plot_surface_3d


def visualise_analytic_genus0(output_dir: str, num_points: int = 10000, config_path: str = None):
    """Visualize analytic ellipsoid embeddings (genus 0) for different semi-axes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Different ellipsoid configurations with quantitative labels
    ellipsoid_configs = [
        {'a': 1.0, 'b': 1.0, 'c': 1.0, 'label': 'a=1, b=1, c=1'},
        {'a': 2.0, 'b': 1.0, 'c': 1.0, 'label': 'a=2, b=1, c=1'},
        {'a': 2.0, 'b': 2.0, 'c': 1.0, 'label': 'a=2, b=2, c=1'},
        {'a': 1.5, 'b': 1.0, 'c': 0.5, 'label': 'a=1.5, b=1, c=0.5'},
        {'a': 1.2, 'b': 0.6, 'c': 0.3, 'label': 'a=1.2, b=0.6, c=0.3'},
    ]
    
    n_configs = len(ellipsoid_configs)
    n_plots = n_configs + 1  # +1 for domain plot
    n_rows = int(np.ceil(n_plots / 2))
    fig = plt.figure(figsize=(12, 5 * n_rows))
    
    # Add fundamental domain coloring
    ax_domain = fig.add_subplot(n_rows, 2, 1)
    plot_fundamental_domain_coloring(ax_domain, genus=0)
    
    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}

    # First pass: collect xyz for all surfaces
    _surfaces = []
    for idx, cfg in enumerate(ellipsoid_configs):
        genus = 0
        domain = "ellipsoid"
        print(f"Visualizing analytic surface {idx+1} for genus {genus} ({genus_names.get(genus, 'unknown')})")
        print(f"  Domain: {domain}")
        dummy_config = {'topology': {'genus': genus, 'ellipsoid': cfg}, 'model': {}}
        from model import create_embedding_model
        model = create_embedding_model(dummy_config, device, skip_init=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        topology_params = {'ellipsoid': cfg}
        uv = sample_parameters(num_points, domain=domain, device=device, dtype=dtype)
        with torch.no_grad():
            xyz = get_reference_embedding(uv, genus=genus, topology_params=topology_params)
        _surfaces.append((cfg, uv, xyz))

    # Shared scale across all subplots
    _all_xyz = np.concatenate([s[2].cpu().numpy() for s in _surfaces])
    _global_range = np.array([
        _all_xyz[:, 0].max() - _all_xyz[:, 0].min(),
        _all_xyz[:, 1].max() - _all_xyz[:, 1].min(),
        _all_xyz[:, 2].max() - _all_xyz[:, 2].min()
    ]).max() / 2.0

    for idx, (cfg, uv, xyz) in enumerate(_surfaces):
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz, uv, cfg['label'], genus=0, global_range=_global_range)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'analytic_genus0.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved genus 0 (ellipsoid) visualisation to {output_path}")
    plt.close()


def visualise_analytic_genus1(output_dir: str, num_points: int = 10000, config_path: str = None):
    """Visualize analytic torus embeddings (genus 1) for different tau values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    tau_values = [
        ("1j", "τ = 1.0i"),
        ("0.5j", "τ = 0.5i"),
        ("1.0+0.5j", "τ = 1.0+0.5i"),
        ("0.2+0.2j", "τ = 0.2+0.2i"),
        ("1.0+0.2j", "τ = 1.0+0.2i"),
    ]
    
    n_tau = len(tau_values)
    n_plots = n_tau + 1  # +1 for domain plot
    n_rows = int(np.ceil(n_plots / 2))
    fig = plt.figure(figsize=(12, 5 * n_rows))

    ax_domain = fig.add_subplot(n_rows, 2, 1)
    # Use the first tau value for the domain plot
    tau = complex(tau_values[0][0].replace('i', 'j'))
    plot_fundamental_domain_coloring(ax_domain, genus=1, tau1=tau)

    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}
    from sampling import transform_square_to_parallelogram, sample_rectangular_domain

    # First pass: collect xyz for all surfaces
    _surfaces = []
    for idx, (tau_str, label) in enumerate(tau_values):
        genus = 1
        domain = "torus"
        print(f"Visualizing analytic surface {idx+1} for genus {genus} ({genus_names.get(genus, 'unknown')})")
        print(f"  Domain: {domain}")
        dummy_config = {'topology': {'genus': genus, 'torus': {'tau': tau_str}}, 'model': {}}
        from model import create_embedding_model
        model = create_embedding_model(dummy_config, device, skip_init=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        tau = complex(tau_str.replace('i', 'j'))
        uv = sample_rectangular_domain(num_points, (0, 2*np.pi), (0, 2*np.pi), device)
        with torch.no_grad():
            xyz = get_reference_embedding(uv, domain=domain, tau=tau)
        _surfaces.append((label, uv, xyz))

    # Shared scale across all subplots
    _all_xyz = np.concatenate([s[2].cpu().numpy() for s in _surfaces])
    _global_range = np.array([
        _all_xyz[:, 0].max() - _all_xyz[:, 0].min(),
        _all_xyz[:, 1].max() - _all_xyz[:, 1].min(),
        _all_xyz[:, 2].max() - _all_xyz[:, 2].min()
    ]).max() / 2.0

    for idx, (label, uv, xyz) in enumerate(_surfaces):
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz, uv, label, genus=1, global_range=_global_range)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'analytic_genus1.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved genus 1 (torus) visualisation to {output_path}")
    plt.close()


def visualise_analytic_genus2(output_dir: str, num_points: int = 15000, config_path: str = None):
    """Visualize analytic double torus embeddings (genus 2).
    
    Two independent tori connected by a catenoid bridge.
    
    Parameters:
    - τ₁, τ₂ ∈ ℂ: modular parameters of each torus
    - neck_length > 0: length of catenoid bridge
    - neck_twist ∈ ℝ: twist at bridge (placeholder)
    
    Three fundamental domains: T1 parallelogram, catenoid rectangle, T2 parallelogram.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Different configurations showing parameter variations
    # Use distinct parameters to show visible differences:
    # - Im(τ) controls tube thickness (larger → thinner tube, r ~ exp(-k*Im(τ)))
    # - Re(τ) controls helical twist (shear of fundamental domain)
    # - bridge_radius controls the catenoid neck narrowness
    dt_configs = [
        # Config 1: Standard symmetric double torus
        {'tau1': {'real': 0.0, 'imag': 1.0}, 'tau2': {'real': 0.0, 'imag': 1.0},
         'bridge_radius': 0.25, 'neck_twist': 0.0, 'scale': 1.2,
         'label': 'τ₁=τ₂=i (symmetric)'},
        # Config 2: Asymmetric tube thicknesses (T1 thicker, T2 thinner)
        {'tau1': {'real': 0.0, 'imag': 0.5}, 'tau2': {'real': 0.0, 'imag': 2.0},
         'bridge_radius': 0.2, 'neck_twist': 0.0, 'scale': 1.2,
         'label': 'τ₁=0.5i (thick), τ₂=2i (thin)'},
        # Config 3: Opposite twists on each torus
        {'tau1': {'real': 0.5, 'imag': 1.0}, 'tau2': {'real': -0.5, 'imag': 1.0},
         'bridge_radius': 0.25, 'neck_twist': 0.0, 'scale': 1.2,
         'label': 'τ₁=0.5+i, τ₂=-0.5+i (twisted)'},
        # Config 4: Strong twist on T1 only
        {'tau1': {'real': 1.0, 'imag': 0.8}, 'tau2': {'real': 0.0, 'imag': 1.2},
         'bridge_radius': 0.2, 'neck_twist': 0.0, 'scale': 1.2,
         'label': 'τ₁=1+0.8i (strong twist)'},
        # Config 5: Narrow neck catenoid
        {'tau1': {'real': 0.0, 'imag': 0.7}, 'tau2': {'real': 0.0, 'imag': 0.7},
         'bridge_radius': 0.12, 'neck_twist': 0.0, 'scale': 1.2,
         'label': 'bridge_radius=0.12 (narrow)'},
    ]
    
    n_configs = len(dt_configs)
    n_plots = n_configs + 1  # +1 for domain plot
    n_rows = int(np.ceil(n_plots / 2))
    fig = plt.figure(figsize=(12, 5 * n_rows))
    
    # Use first config for the fundamental domain plot
    first_cfg = dt_configs[0]
    tau1 = complex(first_cfg['tau1']['real'], first_cfg['tau1']['imag'])
    tau2 = complex(first_cfg['tau2']['real'], first_cfg['tau2']['imag'])
    
    ax_domain = fig.add_subplot(n_rows, 2, 1)
    plot_fundamental_domain_coloring(ax_domain, genus=2, 
                                     tau1=tau1, tau2=tau2,
                                     neck_radius=0.3)  # Legacy param name for domain plot
    
    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}

    # First pass: collect xyz for all surfaces
    _surfaces = []
    for idx, cfg in enumerate(dt_configs):
        genus = 2
        domain = "double_torus"
        label = cfg.pop('label')
        print(f"Visualizing analytic surface {idx+1} for genus {genus} ({genus_names.get(genus, 'unknown')})")
        print(f"  Domain: {domain}")
        dummy_config = {'topology': {'genus': genus, 'double_torus': cfg}, 'model': {}}
        from model import create_embedding_model
        model = create_embedding_model(dummy_config, device, skip_init=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        topology_params = {'double_torus': cfg}
        uv = sample_parameters(num_points, domain=domain, device=device, dtype=dtype)
        with torch.no_grad():
            xyz = get_reference_embedding(uv, genus=genus, topology_params=topology_params)
        cfg['label'] = label  # Restore for summary
        _surfaces.append((label, uv, xyz))

    # Shared scale across all subplots
    _all_xyz = np.concatenate([s[2].cpu().numpy() for s in _surfaces])
    _global_range = np.array([
        _all_xyz[:, 0].max() - _all_xyz[:, 0].min(),
        _all_xyz[:, 1].max() - _all_xyz[:, 1].min(),
        _all_xyz[:, 2].max() - _all_xyz[:, 2].min()
    ]).max() / 2.0

    for idx, (label, uv, xyz) in enumerate(_surfaces):
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz, uv, label, genus=2, global_range=_global_range)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'analytic_genus2.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved genus 2 (double torus) visualisation to {output_path}")
    plt.close()


def main():
    """Main function to visualise analytic embeddings."""
    import argparse
    
    # Default config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, 'hyperparameters.yaml')
    
    parser = argparse.ArgumentParser(description="Visualize analytical surface embeddings")
    parser.add_argument('--config', type=str, default=default_config,
                       help='Path to config file')
    parser.add_argument('--points', type=int, default=10000,
                       help='Number of points to sample')
    parser.add_argument('--genus', type=int, default=None, choices=[0, 1, 2],
                       help='Genus to visualize (0, 1, or 2). If not specified, visualize all.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generate logs/analytic_run_#)')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        base_dir = os.path.join(project_root, 'logs')
        run_num = get_next_run_number(base_dir, "analytic_run_")
        output_dir = os.path.join(base_dir, f"analytic_run_{run_num}")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ANALYTIC SURFACE VISUALISATION")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Points per surface: {args.points}")
    
    genera_to_vis = [args.genus] if args.genus is not None else [0, 1, 2]
    
    for genus in genera_to_vis:
        print(f"\n{'='*60}")
        print(f"Visualizing Genus {genus} surfaces...")
        print(f"{'='*60}")
        
        if genus == 0:
            visualise_analytic_genus0(output_dir, args.points, args.config)
        elif genus == 1:
            visualise_analytic_genus1(output_dir, args.points, args.config)
        elif genus == 2:
            visualise_analytic_genus2(output_dir, args.points, args.config)
    
    print(f"\n{'='*60}")
    print("VISUALISATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
