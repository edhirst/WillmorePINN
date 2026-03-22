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

from sampling import get_reference_embedding, sample_parameters, get_domain_for_genus, sample_torus_excluding_disk, sample_bridge_domain
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
    """Visualize analytic double torus reference embeddings (genus 2).

    Two independent tori connected by a catenoid bridge, each with a disk
    excised.  Reference embeddings come from model.reference_torus1/bridge/torus2.
    """
    from model import create_embedding_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    dt_configs = [
        # T2 disk at (π,0) faces T1's disk at (0,0) so the bridge goes right→left
        # torus2_offset = (R+r1) + (R+r2) + gap  (R=1, r=Im(τ), gap=1.0)
        {'tau1': '1j', 'tau2': '1j',
         'disk_radius': 0.3,
         'disk_center_T1': [0.0, 0.0], 'disk_center_T2': [3.14159265359, 0.0],
         'torus2_offset': [5.0, 0.0, 0.0],
         'label': 'τ₁=τ₂=i (symmetric)'},
        {'tau1': '0.5j', 'tau2': '0.7j',
         'disk_radius': 0.3,
         'disk_center_T1': [0.0, 0.0], 'disk_center_T2': [3.14159265359, 0.0],
         'torus2_offset': [4.2, 0.0, 0.0],
         'label': 'τ₁=0.5i, τ₂=0.7i (asymmetric)'},
        {'tau1': '1.0+0.5j', 'tau2': '-1.0+0.5j',
         'disk_radius': 0.3,
         'disk_center_T1': [0.0, 0.0], 'disk_center_T2': [3.14159265359, 0.0],
         'torus2_offset': [4.0, 0.0, 0.0],
         'label': 'τ₁=1+0.5i, τ₂=-1+0.5i (twisted)'},
        {'tau1': '0.5+0.3j', 'tau2': '0.5j',
         'disk_radius': 0.3,
         'disk_center_T1': [0.0, 0.0], 'disk_center_T2': [3.14159265359, 0.0],
         'torus2_offset': [3.8, 0.0, 0.0],
         'label': 'τ₁=0.5+0.3i (twist+thin)'},
        {'tau1': '0.7j', 'tau2': '0.7j',
         'disk_radius': 0.12,
         'disk_center_T1': [0.0, 0.0], 'disk_center_T2': [3.14159265359, 0.0],
         'torus2_offset': [4.4, 0.0, 0.0],
         'label': 'disk_radius=0.12 (narrow neck)'},
    ]

    n_configs = len(dt_configs)
    n_plots = n_configs + 1
    n_rows = int(np.ceil(n_plots / 2))
    fig = plt.figure(figsize=(12, 5 * n_rows))

    first_cfg = dt_configs[0]
    tau1_plot = complex(first_cfg['tau1'].replace('i', 'j'))
    tau2_plot = complex(first_cfg['tau2'].replace('i', 'j'))
    ax_domain = fig.add_subplot(n_rows, 2, 1)
    plot_fundamental_domain_coloring(
        ax_domain, genus=2, tau1=tau1_plot, tau2=tau2_plot,
        neck_radius=first_cfg['disk_radius'],
        disk_center_T1=tuple(first_cfg['disk_center_T1']),
        disk_center_T2=tuple(first_cfg['disk_center_T2']),
    )

    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}

    _surfaces = []
    for idx, cfg in enumerate(dt_configs):
        label = cfg['label']
        dt_params = {k: v for k, v in cfg.items() if k != 'label'}
        print(f"Visualizing analytic surface {idx+1} for genus 2 ({genus_names[2]})")
        config = {'topology': {'genus': 2, 'double_torus': dt_params}, 'model': {}}
        model = create_embedding_model(config, device, skip_init=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        n_each = num_points // 3
        delta = float(dt_params['disk_radius'])
        c_T1 = model.disk_center_T1
        c_T2 = model.disk_center_T2
        with torch.no_grad():
            uv_t1 = sample_torus_excluding_disk(n_each, disk_center=c_T1, disk_radius=delta, device=device, dtype=dtype)
            ut_br = sample_bridge_domain(n_each, device=device, dtype=dtype)
            uv_t2 = sample_torus_excluding_disk(n_each, disk_center=c_T2, disk_radius=delta, device=device, dtype=dtype)
            xyz = torch.cat([model.reference_torus1(uv_t1),
                             model.reference_bridge(ut_br),
                             model.reference_torus2(uv_t2)], dim=0)
            uv  = torch.cat([uv_t1, ut_br, uv_t2], dim=0)
        _surfaces.append((label, uv, xyz, c_T1, c_T2, uv_t1, ut_br, uv_t2))

    _all_xyz = np.concatenate([s[2].cpu().numpy() for s in _surfaces])
    _global_range = np.array([
        _all_xyz[:, 0].max() - _all_xyz[:, 0].min(),
        _all_xyz[:, 1].max() - _all_xyz[:, 1].min(),
        _all_xyz[:, 2].max() - _all_xyz[:, 2].min()
    ]).max() / 2.0

    for idx, (label, uv, xyz, c_T1, c_T2, uv_t1, ut_br, uv_t2) in enumerate(_surfaces):
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        colors = make_genus2_colors(uv_t1, ut_br, uv_t2,
                                    disk_center_T1=c_T1, disk_center_T2=c_T2)
        plot_surface_3d(ax, xyz, uv, label, genus=2, global_range=_global_range,
                        color_values=colors)

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
