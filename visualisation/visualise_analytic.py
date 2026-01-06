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
        period = 5 * np.pi  # v goes from 0 to 5π for double torus
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
    
    # Set consistent viewing angle
    ax.view_init(elev=30, azim=-60)


def plot_fundamental_domain_coloring(ax, genus=1, num_points=100, tau1=1j, tau2=1j, neck_radius=0.3):
    """Plot the fundamental domain with the periodic coloring.
    
    For genus 2, draws 2 parallelogram domains (one per torus) with the neck region indicated.
    
    Args:
        ax: Matplotlib axes
        genus: Surface genus (0, 1, or 2)
        num_points: Grid resolution for genus 0/1
        tau1, tau2: Complex modular parameters for genus 2 tori
        neck_radius: Neck radius for genus 2
    """
    if genus == 2:
        # Draw two parallelograms for connected sum T² # T²
        plot_connected_sum_domain(ax, tau1=tau1, tau2=tau2, neck_radius=neck_radius)
        return
    
    # Domain depends on genus
    if genus == 0:
        u_max, v_max = 2 * np.pi, np.pi
        v_label = 'π'
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
        ax.set_title(f'Fundamental Domain (Genus {genus})', fontsize=10)
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_xticklabels(['0', 'π', '2π'])
        ax.set_yticks([0, v_max/2, v_max])
        ax.set_yticklabels(['0', f'{v_label}/2', v_label])
    else:
        # For genus 1, plot the square fundamental domain for tau = i
        u = np.linspace(0, 2*np.pi, num_points)
        v = np.linspace(0, 2*np.pi, num_points)
        U, V = np.meshgrid(u, v)
        # Apply rainbow coloring along v
        v_norm = (V % (2*np.pi)) / (2*np.pi)
        hue = v_norm
        h = hue * 6.0
        x = 1.0 - np.abs(h % 2.0 - 1.0)
        R = np.where((h >= 0) & (h < 1), 1.0, np.where((h >= 1) & (h < 2), x, np.where((h >= 4) & (h < 5), x, np.where((h >= 5) & (h < 6), 1.0, 0.0))))
        G = np.where((h >= 0) & (h < 1), x, np.where((h >= 1) & (h < 3), 1.0, np.where((h >= 3) & (h < 4), x, 0.0)))
        B = np.where((h >= 2) & (h < 3), x, np.where((h >= 3) & (h < 5), 1.0, np.where((h >= 5) & (h < 6), x, 0.0)))
        colors = np.stack([R, G, B], axis=-1)
        # Plot the colored square domain
        ax.imshow(colors, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower', aspect='auto')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_title(f'Fundamental Domain (Genus 1, τ = i)', fontsize=10)
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_xticklabels(['0', 'π', '2π'])
        ax.set_yticks([0, np.pi, 2*np.pi])
        ax.set_yticklabels(['0', 'π', '2π'])


def plot_connected_sum_domain(ax, tau1=1j, tau2=1j, neck_radius=0.3, num_points=100):
    """
    Draw the fundamental domain for connected sum T² # T².
    
    Three regions stacked vertically:
        - v ∈ [0, 2π): Torus 1 (parallelogram with shear from Re(τ₁))
        - v ∈ [2π, 3π): Catenoid bridge
        - v ∈ [3π, 4π): Torus 2 (parallelogram with shear from Re(τ₂))
    
    Domain: u ∈ [0, 2π], v ∈ [0, 4π]
    
    Coloring is continuous across boundaries (matching colors where regions join).
    """
    from matplotlib.patches import Polygon, FancyArrowPatch
    from matplotlib.collections import PatchCollection
    import matplotlib.colors as mcolors
    
    # Parse complex parameters
    if isinstance(tau1, (int, float)):
        tau1 = complex(0, float(tau1))
    if isinstance(tau2, (int, float)):
        tau2 = complex(0, float(tau2))
    
    tau1_re, tau1_im = tau1.real, max(tau1.imag, 0.3)
    tau2_re, tau2_im = tau2.real, max(tau2.imag, 0.3)
    
    # Create colored grid for the full domain
    u = np.linspace(0, 2*np.pi, num_points)
    v = np.linspace(0, 5*np.pi, num_points)
    U, V = np.meshgrid(u, v)
    # Use the same HSV mapping as the 3D plot: v in [0, 5π], period=5π
    v_norm = (V % (5 * np.pi)) / (5 * np.pi)
    hue = v_norm
    h = hue * 6.0
    x = 1.0 - np.abs(h % 2.0 - 1.0)
    R = np.where((h >= 0) & (h < 1), 1.0, np.where((h >= 1) & (h < 2), x, np.where((h >= 4) & (h < 5), x, np.where((h >= 5) & (h < 6), 1.0, 0.0))))
    G = np.where((h >= 0) & (h < 1), x, np.where((h >= 1) & (h < 3), 1.0, np.where((h >= 3) & (h < 4), x, 0.0)))
    B = np.where((h >= 2) & (h < 3), x, np.where((h >= 3) & (h < 5), 1.0, np.where((h >= 5) & (h < 6), x, 0.0)))
    colors = np.stack([R, G, B], axis=-1)
    # Display the colored domain
    ax.imshow(colors, extent=[0, 2*np.pi, 0, 5*np.pi], origin='lower', aspect='auto')
    
    # Draw boundaries between regions
    # T1 -> Catenoid boundary at v = 2π
    ax.axhline(y=2*np.pi, color='white', linewidth=2, linestyle='-')
    ax.axhline(y=2*np.pi, color='black', linewidth=1, linestyle='--')
    
    # Catenoid -> T2 boundary at v = 3π
    ax.axhline(y=3*np.pi, color='white', linewidth=2, linestyle='-')
    ax.axhline(y=3*np.pi, color='black', linewidth=1, linestyle='--')
    
    # Add labels for each region
    ax.text(np.pi, np.pi, f'Torus 1\nτ₁ = {tau1_re:.1f}+{tau1_im:.1f}i', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax.text(np.pi, 2.5*np.pi, 'Catenoid', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax.text(np.pi, 4*np.pi, f'Torus 2\nτ₂ = {tau2_re:.1f}+{tau2_im:.1f}i', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Draw parallelogram outlines showing the shear from Re(τ)
    shear1 = tau1_re * 0.3  # Scale shear for visibility
    shear2 = tau2_re * 0.3
    
    # Parallelogram edges for torus 1 (showing shear)
    if abs(shear1) > 0.01:
        ax.plot([0, shear1], [0, 2*np.pi], 'w-', linewidth=1.5, alpha=0.7)
        ax.plot([2*np.pi, 2*np.pi + shear1], [0, 2*np.pi], 'w-', linewidth=1.5, alpha=0.7)
        ax.annotate('', xy=(shear1/2, np.pi), xytext=(0, np.pi),
                   arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        ax.text(shear1/4, np.pi + 0.3, f'Re(τ₁)={tau1_re:.1f}', fontsize=7, color='white')
    
    # Parallelogram edges for torus 2 (showing shear) - T2 at v ∈ [3π, 5π)
    if abs(shear2) > 0.01:
        ax.plot([0, shear2], [3*np.pi, 5*np.pi], 'w-', linewidth=1.5, alpha=0.7)
        ax.plot([2*np.pi, 2*np.pi + shear2], [3*np.pi, 5*np.pi], 'w-', linewidth=1.5, alpha=0.7)
        ax.annotate('', xy=(shear2/2, 4*np.pi), xytext=(0, 4*np.pi),
                   arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        ax.text(shear2/4, 4*np.pi + 0.3, f'Re(τ₂)={tau2_re:.1f}', fontsize=7, color='white')
    
    # Axis setup
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title('Three-Region Domain: T₁ + Catenoid + T₂', fontsize=10)
    
    ax.set_xlim(-0.5, 2*np.pi + 0.8)
    ax.set_ylim(-0.3, 5*np.pi + 0.3)
    
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels(['0', 'π', '2π'])
    ax.set_yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi, 5*np.pi])
    ax.set_yticklabels(['0', 'π', '2π', '3π', '4π', '5π'])
    

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
    for idx, cfg in enumerate(ellipsoid_configs):
        genus = 0
        domain = "ellipsoid"
        print(f"Visualizing analytic surface {idx+1} for genus {genus} ({genus_names.get(genus, 'unknown')})")
        print(f"  Domain: {domain}")
        # Create dummy config for parameter count
        dummy_config = {'topology': {'genus': genus, 'ellipsoid': cfg}, 'model': {}}
        from model import create_embedding_model
        model = create_embedding_model(dummy_config, device, skip_init=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        topology_params = {'ellipsoid': cfg}
        uv = sample_parameters(num_points, domain=domain, device=device, dtype=dtype)
        with torch.no_grad():
            xyz = get_reference_embedding(uv, genus=genus, topology_params=topology_params)
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz, uv, cfg['label'], genus=genus)
    
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
    for idx, (tau_str, label) in enumerate(tau_values):
        genus = 1
        domain = "torus"
        print(f"Visualizing analytic surface {idx+1} for genus {genus} ({genus_names.get(genus, 'unknown')})")
        print(f"  Domain: {domain}")
        # Create dummy config for parameter count
        dummy_config = {'topology': {'genus': genus, 'torus': {'tau': tau_str}}, 'model': {}}
        from model import create_embedding_model
        model = create_embedding_model(dummy_config, device, skip_init=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        tau = complex(tau_str.replace('i', 'j'))
        uv = sample_rectangular_domain(num_points, (0, 2*np.pi), (0, 2*np.pi), device)
        with torch.no_grad():
            xyz = get_reference_embedding(uv, domain=domain, tau=tau)
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz, uv, label, genus=genus)

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
    for idx, cfg in enumerate(dt_configs):
        genus = 2
        domain = "double_torus"
        label = cfg.pop('label')
        print(f"Visualizing analytic surface {idx+1} for genus {genus} ({genus_names.get(genus, 'unknown')})")
        print(f"  Domain: {domain}")
        # Create dummy config for parameter count
        dummy_config = {'topology': {'genus': genus, 'double_torus': cfg}, 'model': {}}
        from model import create_embedding_model
        model = create_embedding_model(dummy_config, device, skip_init=True)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())} trainable")
        topology_params = {'double_torus': cfg}
        uv = sample_parameters(num_points, domain=domain, device=device, dtype=dtype)
        with torch.no_grad():
            xyz = get_reference_embedding(uv, genus=genus, topology_params=topology_params)
        ax = fig.add_subplot(n_rows, 2, idx + 2, projection='3d')
        plot_surface_3d(ax, xyz, uv, label, genus=genus)
        cfg['label'] = label  # Restore for summary
    
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
