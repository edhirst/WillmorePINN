"""
Utility functions for analysis of learned embeddings.

This module provides functions to analyze embeddings, compute curvatures,
and compare with theoretical values.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Optional, Dict
import os
import yaml
import glob
import re

from model import create_embedding_model
from sampling import sample_parameters, compute_reference_willmore_energy


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


def plot_surface_3d(ax, xyz, uv, title, genus=1, alpha=0.6, global_range=None):
    """Plot a 3D surface with improved coloring.

    Args:
        ax: Matplotlib 3D axes.
        xyz: Surface points, shape (N, 3).
        uv: Parameter coordinates, shape (N, 2).
        title: Subplot title.
        genus: Surface genus (for colour period).
        alpha: Point transparency.
        global_range: Shared half-extent for all three axes. Each axis is set to
            [mid - global_range, mid + global_range], where mid is the midpoint
            of that axis for this subplot. Passing the same value to every subplot
            in a figure enforces a uniform spatial scale.
    """
    xyz_np = xyz.detach().cpu().numpy() if torch.is_tensor(xyz) else xyz
    uv_np = uv.cpu().numpy() if torch.is_tensor(uv) else uv

    if genus == 2:
        # uv is Poincaré-disk (x, y); colour by radius to match domain plot
        rad = np.sqrt(uv_np[:, 0]**2 + uv_np[:, 1]**2)
        r_oct = _octagon_disk_radius_val()
        norm_rad = np.clip(rad / r_oct, 0.0, 1.0)
        colors = _GENUS2_CMAP(norm_rad)
    else:
        v = uv_np[:, 1]
        # Determine period based on genus (for color mapping)
        period = np.pi if genus == 0 else 2 * np.pi
        colors = hsv_to_rgb_colors(v, period)
    
    ax.scatter(
        xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2],
        c=colors, alpha=alpha, s=2
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)
    
    mid_x = (xyz_np[:, 0].max() + xyz_np[:, 0].min()) * 0.5
    mid_y = (xyz_np[:, 1].max() + xyz_np[:, 1].min()) * 0.5
    mid_z = (xyz_np[:, 2].max() + xyz_np[:, 2].min()) * 0.5

    if global_range is None:
        global_range = np.array([
            xyz_np[:, 0].max() - xyz_np[:, 0].min(),
            xyz_np[:, 1].max() - xyz_np[:, 1].min(),
            xyz_np[:, 2].max() - xyz_np[:, 2].min()
        ]).max() / 2.0

    ax.set_xlim(mid_x - global_range, mid_x + global_range)
    ax.set_ylim(mid_y - global_range, mid_y + global_range)
    ax.set_zlim(mid_z - global_range, mid_z + global_range)
    ax.set_box_aspect([1, 1, 1])


def plot_fundamental_domain_coloring(ax, genus=1, num_points=100, tau1=1j, tau2=1j, neck_radius=0.3):
    """Plot the fundamental domain with the periodic coloring.

    For genus 2, draws the regular hyperbolic octagon in the Poincaré disk with
    edge-identification labels and a rainbow coloring by angle from the origin.

    Args:
        ax: Matplotlib axes
        genus: Surface genus (0, 1, or 2)
        num_points: Grid resolution for genus 0/1
        tau1, tau2: Unused for genus 2 (kept for backward-compatible signature)
        neck_radius: Unused for genus 2
    """
    if genus == 2:
        _plot_hyperbolic_octagon_domain(ax, num_points=num_points)
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


# Colormap used for genus-2 domain and 3-D scatter (radius from centre).
_GENUS2_CMAP = plt.cm.plasma


def _octagon_disk_radius_val() -> float:
    """Return r_oct = tanh(arccosh(cot(π/8)) / 2) ≈ 0.6436."""
    import math
    return math.tanh(math.acosh(1.0 / math.tan(math.pi / 8)) / 2.0)


def _plot_hyperbolic_octagon_domain(ax, num_points=300):
    """Draw the fundamental domain as a regular octagon colored by radius from centre.

    The octagon is drawn in a plain square region with no axis decorations.
    Color encodes distance from the centre using the plasma colormap (shared
    with the genus-2 3-D scatter plots).

    Args:
        ax: Matplotlib axes
        num_points: Resolution of the pixel fill grid
    """
    import math

    r_oct = _octagon_disk_radius_val()
    angles_v = [math.pi / 8 + k * math.pi / 4 for k in range(8)]
    verts = np.array([[r_oct * math.cos(a), r_oct * math.sin(a)] for a in angles_v])

    try:
        from spectral import is_inside_octagon
        _inside_fn = lambda pts: is_inside_octagon(pts, verts)
    except ImportError:
        def _inside_fn(pts):
            inside = np.ones(len(pts), dtype=bool)
            for i in range(8):
                v1, v2 = verts[i], verts[(i + 1) % 8]
                edge = v2 - v1
                dp = pts - v1
                inside &= (edge[0] * dp[:, 1] - edge[1] * dp[:, 0]) >= -1e-10
            return inside

    pad = r_oct * 0.12
    lin = np.linspace(-r_oct - pad, r_oct + pad, num_points)
    gx, gy = np.meshgrid(lin, lin)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    mask = _inside_fn(pts)

    rad = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    norm_rad = np.clip(rad / r_oct, 0.0, 1.0)

    rgba = np.zeros((len(pts), 4))
    rgba[mask] = _GENUS2_CMAP(norm_rad[mask])
    color_img = rgba.reshape(num_points, num_points, 4)

    extent = [lin[0], lin[-1], lin[0], lin[-1]]
    ax.imshow(color_img, origin='lower', extent=extent,
              interpolation='bilinear', aspect='equal')

    # Octagon outline
    ox = list(verts[:, 0]) + [verts[0, 0]]
    oy = list(verts[:, 1]) + [verts[0, 1]]
    ax.plot(ox, oy, 'k-', linewidth=1.5)

    ax.set_xlim(lin[0], lin[-1])
    ax.set_ylim(lin[0], lin[-1])
    ax.set_aspect('equal')

    tick_vals = [-r_oct, 0.0, r_oct]
    tick_labels = [f'−{r_oct:.2f}', '0', f'{r_oct:.2f}']
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Fundamental Domain (Genus 2)', fontsize=10)


def get_next_run_number(base_dir: str, prefix: str = "run_") -> int:
    """
    Find the next available run number for the given prefix.
    
    Args:
        base_dir: Base directory containing run folders
        prefix: Prefix for run folders (e.g., "run_", "analytic_run_", "supervised_run_")
    
    Returns:
        Next available run number (1 if no existing runs)
    """
    os.makedirs(base_dir, exist_ok=True)
    existing = glob.glob(os.path.join(base_dir, f"{prefix}*"))
    if not existing:
        return 1
    
    numbers = []
    for path in existing:
        dirname = os.path.basename(path)
        match = re.search(rf'{prefix}(\d+)', dirname)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) + 1 if numbers else 1


def load_checkpoint(checkpoint_path: str, config: dict, device: torch.device):
    """
    Load a trained embedding model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model on
    
    Returns:
        Loaded model, epoch, loss
    """
    model = create_embedding_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    return model, epoch, loss


def compute_embedding_statistics(
    model: torch.nn.Module,
    uv: torch.Tensor,
    domain: str = 'torus'
) -> Dict:
    """
    Compute statistics about the learned embedding.
    
    Args:
        model: Trained embedding model
        uv: Parameter coordinates
        domain: Surface type
    
    Returns:
        Dictionary with embedding statistics
    """
    with torch.no_grad():
        # Get embedding
        xyz = model(uv)
        
        # Compute fundamental forms
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv)
        L, M, N, normal = model.compute_second_fundamental_form(uv, phi_u, phi_v)
        
        # Mean curvature
        H = model.compute_mean_curvature(E, F, G, L, M, N)
        
        # Area element
        area_element = torch.sqrt(torch.abs(E * G - F * F) + 1e-8)
        
        # Gaussian curvature K = (LN - M²) / (EG - F²)
        K = (L * N - M * M) / (E * G - F * F + 1e-8)
        
        stats = {
            # Embedding bounds
            'xyz_min': xyz.min(dim=0)[0].cpu().numpy().tolist(),
            'xyz_max': xyz.max(dim=0)[0].cpu().numpy().tolist(),
            'xyz_mean': xyz.mean(dim=0).cpu().numpy().tolist(),
            'xyz_std': xyz.std(dim=0).cpu().numpy().tolist(),
            
            # First fundamental form
            'E_mean': E.mean().item(),
            'E_std': E.std().item(),
            'F_mean': F.mean().item(),
            'F_std': F.std().item(),
            'G_mean': G.mean().item(),
            'G_std': G.std().item(),
            
            # Second fundamental form
            'L_mean': L.mean().item(),
            'L_std': L.std().item(),
            'M_mean': M.mean().item(),
            'M_std': M.std().item(),
            'N_mean': N.mean().item(),
            'N_std': N.std().item(),
            
            # Curvatures
            'H_mean': H.mean().item(),
            'H_std': H.std().item(),
            'H_min': H.min().item(),
            'H_max': H.max().item(),
            'K_mean': K.mean().item(),
            'K_std': K.std().item(),
            
            # Area
            'total_area': area_element.sum().item() * (2*np.pi)**2 / len(uv),
        }
    
    return stats


def compute_willmore_energy(
    model: torch.nn.Module,
    uv: torch.Tensor,
    domain: str = 'torus'
) -> float:
    """
    Compute Willmore energy for a given embedding.
    
    Args:
        model: Trained embedding model
        uv: Parameter coordinates
        domain: Surface type
    
    Returns:
        Willmore energy value
    """
    with torch.no_grad():
        # Compute fundamental forms
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv)
        L, M, N, normal = model.compute_second_fundamental_form(uv, phi_u, phi_v)
        
        # Mean curvature
        H = model.compute_mean_curvature(E, F, G, L, M, N)
        
        # Area element
        area_element = torch.sqrt(torch.abs(E * G - F * F) + 1e-8)
        
        # Willmore integrand: H² * area_element
        integrand = H * H * area_element
        
        # Domain area
        if domain == 'torus':
            domain_area = (2 * np.pi) ** 2
        elif domain == 'sphere':
            domain_area = 4 * np.pi
        else:
            domain_area = (2 * np.pi) ** 2
        
        # Monte Carlo integration
        W = (domain_area / len(uv)) * integrand.sum()
        
    return W.item()


def plot_training_history(history_path: str = 'logs/training_history.json', 
                          save_path: Optional[str] = None):
    """
    Plot training history from JSON log file.
    
    Args:
        history_path: Path to training history JSON
        save_path: Path to save figure
    """
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    willmore = history['willmore_energy']
    total_loss = history['total_loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Willmore energy
    axes[0].plot(epochs, willmore, 'b-', linewidth=2, label='Willmore Energy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Willmore Energy')
    axes[0].set_title('Willmore Energy vs Epoch')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Total loss
    axes[1].plot(epochs, total_loss, 'r-', linewidth=2, label='Total Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Total Loss')
    axes[1].set_title('Total Loss vs Epoch')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_curves(history: dict, output_path: str):
    """Plot training loss curves.
    
    Args:
        history: Dictionary containing training history with keys:
                 'epoch', 'total_loss', 'willmore_energy', 'regularity'
        output_path: Path to save the plot
    """
    genus = history.get('genus', 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history['epoch']
    
    # Get genus-specific title
    genus_names = {0: "Sphere/Ellipsoid", 1: "Torus", 2: "Double Torus"}
    fig.suptitle(f"Training Progress - Genus {genus} ({genus_names.get(genus, 'Unknown')})", 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss (log scale)', fontsize=12)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3, which='both')
    
    # Plot 2: Willmore Energy
    axes[0, 1].plot(epochs, history['willmore_energy'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Willmore Energy (log scale)', fontsize=12)
    axes[0, 1].set_title('Willmore Energy', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3, which='both')
    
    # Add theoretical minimum line
    from sampling import get_theoretical_minimum_willmore
    try:
        theoretical_min = get_theoretical_minimum_willmore(genus)
        axes[0, 1].axhline(y=theoretical_min, color='g', linestyle='--', 
                           linewidth=2, alpha=0.7, label=f'Theoretical min: {theoretical_min:.2f}')
        axes[0, 1].legend(fontsize=10)
    except:
        pass
    
    # Plot 3: Regularity Loss
    axes[1, 0].plot(epochs, history['regularity'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Regularity Loss (log scale)', fontsize=12)
    axes[1, 0].set_title('Regularity Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Plot 4: All losses together
    axes[1, 1].plot(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss', alpha=0.7)
    axes[1, 1].plot(epochs, history['willmore_energy'], 'r-', linewidth=2, label='Willmore Energy', alpha=0.7)
    axes[1, 1].plot(epochs, history['regularity'], 'g-', linewidth=2, label='Regularity', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss (log scale)', fontsize=12)
    axes[1, 1].set_title('All Loss Components', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=10, loc='best')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Loss curves saved to {output_path}")
    
    plt.close()


def plot_curvature_distribution(
    model: torch.nn.Module,
    uv: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Plot distribution of mean and Gaussian curvatures.
    
    Args:
        model: Trained embedding model
        uv: Parameter coordinates
        save_path: Path to save figure
    """
    with torch.no_grad():
        # Compute fundamental forms
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv)
        L, M, N, normal = model.compute_second_fundamental_form(uv, phi_u, phi_v)
        
        # Curvatures
        H = model.compute_mean_curvature(E, F, G, L, M, N).cpu().numpy()
        K = ((L * N - M * M) / (E * G - F * F + 1e-8)).cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mean curvature
    axes[0].hist(H, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=H.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {H.mean():.4f}')
    axes[0].set_xlabel('Mean Curvature H')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Mean Curvature Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gaussian curvature
    axes[1].hist(K, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=K.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {K.mean():.4f}')
    axes[1].set_xlabel('Gaussian Curvature K')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Gaussian Curvature Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved curvature distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_model(
    checkpoint_path: str = 'logs/checkpoints/best_model.pt',
    config_path: str = 'configs/hyperparameters.yaml',
    num_test_points: int = 5000,
    device: torch.device = torch.device('cpu'),
    output_dir: str = 'logs/analysis'
):
    """
    Comprehensive analysis of a trained embedding model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        num_test_points: Number of points for analysis
        device: Device to run on
        output_dir: Directory to save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    domain = config['sampling']['domain']
    domain_params = config['sampling'].get('domain_params', {})
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, epoch, loss = load_checkpoint(checkpoint_path, config, device)
    
    # Sample test points
    print(f"Sampling {num_test_points} test points on {domain}...")
    uv = sample_parameters(num_test_points, domain, device)
    
    # Compute statistics
    print("Computing embedding statistics...")
    stats = compute_embedding_statistics(model, uv, domain)
    
    # Compute Willmore energy
    print("Computing Willmore energy...")
    W = compute_willmore_energy(model, uv, domain)
    W_ref = compute_reference_willmore_energy(uv, domain, **domain_params)
    
    stats['willmore_energy'] = W
    stats['reference_willmore'] = W_ref
    stats['willmore_ratio'] = W / W_ref if W_ref > 0 else None
    stats['epoch'] = epoch
    
    # Print statistics
    print("\n" + "="*60)
    print("EMBEDDING ANALYSIS")
    print("="*60)
    print(f"Epoch: {epoch}")
    print(f"Domain: {domain}")
    print(f"\nWillmore Energy:")
    print(f"  Learned: {W:.6f}")
    print(f"  Reference: {W_ref:.6f}")
    print(f"  Ratio: {W/W_ref:.4f}x")
    print(f"\nMean Curvature H:")
    print(f"  Mean: {stats['H_mean']:.6f} ± {stats['H_std']:.6f}")
    print(f"  Range: [{stats['H_min']:.6f}, {stats['H_max']:.6f}]")
    print(f"\nGaussian Curvature K:")
    print(f"  Mean: {stats['K_mean']:.6f} ± {stats['K_std']:.6f}")
    print(f"\nTotal Surface Area: {stats['total_area']:.6f}")
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'embedding_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")
    
    # Generate plots
    print("\nGenerating curvature distribution plot...")
    plot_curvature_distribution(
        model, uv,
        save_path=os.path.join(output_dir, 'curvature_distribution.png')
    )
    
    print("\nGenerating training history plot...")
    plot_training_history(
        history_path='logs/training_history.json',
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a trained Willmore embedding model")
    parser.add_argument("--checkpoint", type=str, default='logs/checkpoints/best_model.pt',
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default='configs/hyperparameters.yaml',
                       help="Path to configuration file")
    parser.add_argument("--num_points", type=int, default=5000,
                       help="Number of test points")
    parser.add_argument("--output_dir", type=str, default="logs/analysis",
                       help="Output directory for analysis")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Get device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Run analysis
    analyze_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_test_points=args.num_points,
        device=device,
        output_dir=args.output_dir
    )
