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
from utils import get_next_run_number


def hsv_to_rgb_colors(values: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Convert normalized values to rainbow RGB colors (HSV-like)."""
    v_norm = (values % period) / period  # Normalize to [0, 1]
    hue = v_norm
    h = hue * 6.0
    x = 1.0 - np.abs(h % 2.0 - 1.0)
    
    colors = np.zeros((len(values), 3))
    mask0 = (h >= 0) & (h < 1)
    mask1 = (h >= 1) & (h < 2)
    mask2 = (h >= 2) & (h < 3)
    mask3 = (h >= 3) & (h < 4)
    mask4 = (h >= 4) & (h < 5)
    mask5 = (h >= 5) & (h < 6)
    
    colors[mask0] = np.stack([np.ones(np.sum(mask0)), x[mask0], np.zeros(np.sum(mask0))], axis=1)
    colors[mask1] = np.stack([x[mask1], np.ones(np.sum(mask1)), np.zeros(np.sum(mask1))], axis=1)
    colors[mask2] = np.stack([np.zeros(np.sum(mask2)), np.ones(np.sum(mask2)), x[mask2]], axis=1)
    colors[mask3] = np.stack([np.zeros(np.sum(mask3)), x[mask3], np.ones(np.sum(mask3))], axis=1)
    colors[mask4] = np.stack([x[mask4], np.zeros(np.sum(mask4)), np.ones(np.sum(mask4))], axis=1)
    colors[mask5] = np.stack([np.ones(np.sum(mask5)), np.zeros(np.sum(mask5)), x[mask5]], axis=1)
    
    return colors


def plot_surface_3d(ax, xyz, uv, title, genus=1):
    """Plot a 3D surface with improved coloring."""
    xyz_np = xyz.detach().cpu().numpy()
    uv_np = uv.cpu().numpy()
    v = uv_np[:, 1]
    
    # Determine period based on genus (for color mapping)
    if genus == 0:
        period = np.pi  # v goes from 0 to π for ellipsoid
    elif genus == 2:
        period = 4 * np.pi  # v goes from 0 to 4π for double torus
    else:
        period = 2 * np.pi  # v goes from 0 to 2π for torus
    
    colors = hsv_to_rgb_colors(v, period)
    
    ax.scatter(
        xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2],
        c=colors, alpha=0.6, s=2
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)
    
    # Equal aspect ratio
    max_range = np.array([
        xyz_np[:, 0].max() - xyz_np[:, 0].min(),
        xyz_np[:, 1].max() - xyz_np[:, 1].min(),
        xyz_np[:, 2].max() - xyz_np[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (xyz_np[:, 0].max() + xyz_np[:, 0].min()) * 0.5
    mid_y = (xyz_np[:, 1].max() + xyz_np[:, 1].min()) * 0.5
    mid_z = (xyz_np[:, 2].max() + xyz_np[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_fundamental_domain_coloring(ax, genus=1, num_points=100):
    """Plot the fundamental domain with the periodic coloring."""
    # Domain depends on genus
    if genus == 0:
        u_max, v_max = 2 * np.pi, np.pi
        v_label = 'π'
    elif genus == 2:
        u_max, v_max = 2 * np.pi, 4 * np.pi
        v_label = '4π'
    else:
        u_max, v_max = 2 * np.pi, 2 * np.pi
        v_label = '2π'
    
    u = np.linspace(0, u_max, num_points)
    v = np.linspace(0, v_max, num_points)
    U, V = np.meshgrid(u, v)
    
    # Apply rainbow coloring
    v_norm = (V % v_max) / v_max
    hue = v_norm
    h = hue * 6.0
    x = 1.0 - np.abs(h % 2.0 - 1.0)
    
    R = np.where((h >= 0) & (h < 1), 1.0, np.where((h >= 1) & (h < 2), x, np.where((h >= 4) & (h < 5), x, np.where((h >= 5) & (h < 6), 1.0, 0.0))))
    G = np.where((h >= 0) & (h < 1), x, np.where((h >= 1) & (h < 3), 1.0, np.where((h >= 3) & (h < 4), x, 0.0)))
    B = np.where((h >= 2) & (h < 3), x, np.where((h >= 3) & (h < 5), 1.0, np.where((h >= 5) & (h < 6), x, 0.0)))
    
    colors = np.stack([R, G, B], axis=-1)
    
    ax.imshow(colors, extent=[0, u_max, 0, v_max], origin='lower', aspect='auto')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f'Fundamental Domain (Genus {genus})\n(Rainbow gradient in v)', fontsize=10)
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels(['0', 'π', '2π'])
    ax.set_yticks([0, v_max/2, v_max])
    ax.set_yticklabels(['0', f'{v_label}/2', v_label])


def visualise_analytic_genus0(output_dir: str, num_points: int = 10000, config_path: str = None):
    """Visualize analytic ellipsoid embeddings (genus 0) for different semi-axes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Different ellipsoid configurations with quantitative labels
    ellipsoid_configs = [
        {'a': 1.0, 'b': 1.0, 'c': 1.0, 'label': 'a=1, b=1, c=1'},
        {'a': 1.5, 'b': 1.0, 'c': 1.0, 'label': 'a=1.5, b=1, c=1'},
        {'a': 1.0, 'b': 1.5, 'c': 1.0, 'label': 'a=1, b=1.5, c=1'},
        {'a': 1.5, 'b': 1.0, 'c': 0.7, 'label': 'a=1.5, b=1, c=0.7'},
        {'a': 2.0, 'b': 0.8, 'c': 0.8, 'label': 'a=2, b=0.8, c=0.8'},
    ]
    
    fig = plt.figure(figsize=(15, 10))
    
    # Add fundamental domain coloring
    ax_domain = fig.add_subplot(2, 3, 1)
    plot_fundamental_domain_coloring(ax_domain, genus=0)
    
    for idx, cfg in enumerate(ellipsoid_configs):
        topology_params = {'ellipsoid': cfg}
        uv = sample_parameters(num_points, domain="ellipsoid", device=device, dtype=dtype)
        
        with torch.no_grad():
            xyz = get_reference_embedding(uv, genus=0, topology_params=topology_params)
        
        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz, uv, cfg['label'], genus=0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'analytic_genus0_ellipsoids.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved genus 0 (ellipsoid) visualisation to {output_path}")
    plt.close()


def visualise_analytic_genus1(output_dir: str, num_points: int = 10000, config_path: str = None):
    """Visualize analytic torus embeddings (genus 1) for different tau values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    tau_values = [
        ("1j", "τ = i"),
        ("0.3+0.95j", "τ = 0.3+0.95i"),
        ("0.5+0.87j", "τ = 0.5+0.87i"),
        ("0.7+0.7j", "τ = 0.7+0.7i"),
        ("-0.4+0.9j", "τ = -0.4+0.9i"),
    ]
    
    fig = plt.figure(figsize=(15, 10))
    
    ax_domain = fig.add_subplot(2, 3, 1)
    plot_fundamental_domain_coloring(ax_domain, genus=1)
    
    for idx, (tau_str, label) in enumerate(tau_values):
        tau = complex(tau_str.replace('i', 'j'))
        uv = sample_parameters(num_points, domain="torus", device=device, dtype=dtype)
        
        with torch.no_grad():
            xyz = get_reference_embedding(uv, domain="torus", tau=tau)
        
        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz, uv, label, genus=1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'analytic_genus1_tori.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved genus 1 (torus) visualisation to {output_path}")
    plt.close()


def visualise_analytic_genus2(output_dir: str, num_points: int = 15000, config_path: str = None):
    """Visualize analytic double torus embeddings (genus 2) for different parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Different double torus configurations with quantitative labels
    dt_configs = [
        {'torus_separation': 3.0, 'torus_major_radius': 1.5, 'torus_minor_radius': 0.6, 
         'bridge_radius': 0.4, 'bridge_length': 1.0, 'label': 'R=1.5, r=0.6, sep=3'},
        {'torus_separation': 2.5, 'torus_major_radius': 1.2, 'torus_minor_radius': 0.5, 
         'bridge_radius': 0.3, 'bridge_length': 0.8, 'label': 'R=1.2, r=0.5, sep=2.5'},
        {'torus_separation': 4.0, 'torus_major_radius': 1.8, 'torus_minor_radius': 0.7, 
         'bridge_radius': 0.5, 'bridge_length': 1.5, 'label': 'R=1.8, r=0.7, sep=4'},
        {'torus_separation': 3.0, 'torus_major_radius': 1.5, 'torus_minor_radius': 0.4, 
         'bridge_radius': 0.35, 'bridge_length': 1.2, 'label': 'R=1.5, r=0.4, sep=3'},
        {'torus_separation': 3.5, 'torus_major_radius': 2.0, 'torus_minor_radius': 0.8, 
         'bridge_radius': 0.6, 'bridge_length': 1.0, 'label': 'R=2, r=0.8, sep=3.5'},
    ]
    
    fig = plt.figure(figsize=(15, 10))
    
    ax_domain = fig.add_subplot(2, 3, 1)
    plot_fundamental_domain_coloring(ax_domain, genus=2)
    
    for idx, cfg in enumerate(dt_configs):
        label = cfg.pop('label')
        topology_params = {'double_torus': cfg}
        uv = sample_parameters(num_points, domain="double_torus", device=device, dtype=dtype)
        
        with torch.no_grad():
            xyz = get_reference_embedding(uv, genus=2, topology_params=topology_params)
        
        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz, uv, label, genus=2)
        cfg['label'] = label  # Restore for summary
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'analytic_genus2_double_tori.png')
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
