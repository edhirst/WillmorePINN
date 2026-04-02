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

from sampling import get_reference_embedding, sample_parameters, get_domain_for_genus, sample_torus_excluding_disk
from utils import get_next_run_number, plot_fundamental_domain_coloring, hsv_to_rgb_colors, plot_surface_3d, make_genus2_colors


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
    """Visualize the genus-2 reference embedding used during training.

    Four panels: parameter-space domain coloring, T₁ alone, T₂ alone, and
    both tori combined.  Parameters are read from config_path if provided,
    otherwise the defaults from config_genus2.yaml are used.
    """
    from model import create_embedding_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Load config
    dt_params = {
        'tau1': '0.7j', 'tau2': '0.7j',
        'disk_radius': 0.65,
        'disk_center_T1': [0.0, 0.0],
    }
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg_file = yaml.safe_load(f)
        dt_params.update(cfg_file.get('topology', {}).get('double_torus', {}))

    tau1 = complex(str(dt_params['tau1']).replace('i', 'j'))
    tau2 = complex(str(dt_params['tau2']).replace('i', 'j'))
    disk_radius = float(dt_params['disk_radius'])
    annular_width_factor = float(dt_params.get('annular_width_factor', 2.0))
    disk_center_T1 = tuple(float(x) for x in dt_params.get('disk_center_T1', [0.0, 0.0]))
    label = f'τ₁={dt_params["tau1"]}, τ₂={dt_params["tau2"]}, δ={disk_radius}'

    config = {'topology': {'genus': 2, 'double_torus': dt_params}, 'model': {}}
    model = create_embedding_model(config, device, skip_init=True)
    c_T1 = model.disk_center_T1
    c_T2 = model.disk_center_T2
    print(f"Genus-2 reference embedding: tau1={tau1}, tau2={tau2}, δ={disk_radius}")
    print(f"  disk_center_T1={c_T1}, disk_center_T2={c_T2}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")

    n_t1 = num_points // 2
    n_t2 = num_points - n_t1
    with torch.no_grad():
        uv_t1 = sample_torus_excluding_disk(n_t1, disk_center=c_T1, disk_radius=disk_radius,
                                            device=device, dtype=dtype)
        uv_t2 = sample_torus_excluding_disk(n_t2, disk_center=c_T2, disk_radius=disk_radius,
                                            device=device, dtype=dtype)
        xyz_t1 = model.reference_torus1(uv_t1)
        xyz_t2 = model.reference_torus2(uv_t2)
        xyz_both = torch.cat([xyz_t1, xyz_t2], dim=0)
        uv_both  = torch.cat([uv_t1,  uv_t2],  dim=0)

    colors_both = make_genus2_colors(uv_t1, uv_t2, disk_center_T1=c_T1, disk_center_T2=c_T2,
                                      collar_radius=disk_radius * annular_width_factor)
    colors_t1   = make_genus2_colors(uv_t1, torch.zeros(0, 2), disk_center_T1=c_T1, disk_center_T2=c_T2,
                                      collar_radius=disk_radius * annular_width_factor)
    colors_t2   = make_genus2_colors(torch.zeros(0, 2), uv_t2,  disk_center_T1=c_T1, disk_center_T2=c_T2,
                                      collar_radius=disk_radius * annular_width_factor)

    all_xyz = xyz_both.cpu().numpy()
    global_range = np.array([
        all_xyz[:, 0].max() - all_xyz[:, 0].min(),
        all_xyz[:, 1].max() - all_xyz[:, 1].min(),
        all_xyz[:, 2].max() - all_xyz[:, 2].min(),
    ]).max() / 2.0

    fig = plt.figure(figsize=(12, 10))

    # Panel 1: parameter-space domain coloring
    ax_domain = fig.add_subplot(2, 2, 1)
    plot_fundamental_domain_coloring(
        ax_domain, genus=2, tau1=tau1, tau2=tau2,
        neck_radius=disk_radius,
        disk_center_T1=disk_center_T1,
        annular_width_factor=annular_width_factor,
    )

    # Panel 2: T₁ alone
    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_surface_3d(ax1, xyz_t1, uv_t1, f'T₁  ({label})', genus=2,
                    global_range=global_range, color_values=colors_t1)

    # Panel 3: T₂ alone
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_surface_3d(ax2, xyz_t2, uv_t2, f'T₂  ({label})', genus=2,
                    global_range=global_range, color_values=colors_t2)

    # Panel 4: both tori combined
    ax3 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_surface_3d(ax3, xyz_both, uv_both, f'T₁ ∪ T₂  ({label})', genus=2,
                    global_range=global_range, color_values=colors_both)

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
    default_config = os.path.join(project_root, 'configs', 'config_genus2.yaml')
    
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
