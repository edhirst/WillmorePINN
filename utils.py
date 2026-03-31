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
import functools
print = functools.partial(print, flush=True)


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


def compute_mean_curvature_at_points(
    model,
    uv: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Compute |H| (mean curvature magnitude) at each (u,v) point.

    Enables grad on uv (needed for second fundamental form) while keeping
    model parameters frozen.  Safe to call inside torch.no_grad() blocks.

    Args:
        model: Trained EmbeddingNetwork.
        uv: Parameter coords, shape (N, 2).
        epsilon: Small constant for numerical stability.

    Returns:
        |H| values, shape (N,), detached from the computation graph.
    """
    with torch.enable_grad():
        uv_g = uv.detach().requires_grad_(True)
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv_g)
        L, M, N, _ = model.compute_second_fundamental_form(uv_g, phi_u, phi_v)
        H = model.compute_mean_curvature(E, F, G, L, M, N, epsilon)
    return H.abs().detach()


def plot_surface_3d(ax, xyz, uv, title, genus=1, alpha=0.6, global_range=None,
                   color_values=None, cmap='hot_r', vmin=None, vmax=None):
    """Plot a 3D surface with improved coloring.

    Args:
        ax: Matplotlib 3D axes.
        xyz: Surface points, shape (N, 3).
        uv: Parameter coordinates, shape (N, 2).
        title: Subplot title.
        genus: Surface genus (for colour period).
        alpha: Point transparency.
        global_range: Shared half-extent for all three axes.
        color_values: Optional (N,) array of scalar values to use as colours
            instead of the default HSV-by-parameter colouring.  Useful for
            visualising |H|, area element, etc.
        cmap: Matplotlib colormap name used when color_values is provided.
        vmin, vmax: Colour scale limits for color_values (None = data range).
    """
    xyz_np = xyz.detach().cpu().numpy() if torch.is_tensor(xyz) else xyz
    uv_np = uv.cpu().numpy() if torch.is_tensor(uv) else uv

    if color_values is not None:
        cv = color_values.detach().cpu().numpy() if torch.is_tensor(color_values) else np.asarray(color_values)
        if cv.ndim == 2:  # pre-computed (N, 3) or (N, 4) RGB array
            scatter_kwargs = dict(c=cv, alpha=alpha, s=2)
        else:
            scatter_kwargs = dict(c=cv, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, s=2)
    else:
        # Genus 2: colour by u ∈ [0, 2π] (the tube angle), full spectrum per point.
        # Other genera: colour by v.
        if genus == 2:
            coord = uv_np[:, 0]  # u ∈ [0, 2π]
            period = 2 * np.pi
        else:
            coord = uv_np[:, 1]  # v
            if genus == 0:
                period = np.pi       # v ∈ [0, π] for ellipsoid
            else:
                period = 2 * np.pi  # v ∈ [0, 2π] for torus
        scatter_kwargs = dict(c=hsv_to_rgb_colors(coord, period), alpha=alpha, s=2)

    ax.scatter(
        xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2],
        **scatter_kwargs
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


def make_genus2_colors(
    uv_t1, uv_t2,
    disk_center_T1=(0.0, 0.0),
    disk_center_T2=(np.pi, 0.0),
) -> np.ndarray:
    """Compute per-point RGB colours for a genus-2 multi-chart surface.

    T₁ and T₂ points are coloured by u (hue = u / 2π, matching the domain
    plot).

    Args:
        uv_t1: (N₁, 2) parameter coords for T₁ (u, v).
        uv_t2: (N₂, 2) parameter coords for T₂ (u, v).
        disk_center_T1: (u₀, v₀) disk centre on T₁.
        disk_center_T2: (u₀, v₀) disk centre on T₂.

    Returns:
        (N₁+N₂, 3) float32 RGB array in [0, 1].
    """
    def _to_np(x):
        return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)

    uv1 = _to_np(uv_t1)
    uv2 = _to_np(uv_t2)

    c1 = hsv_to_rgb_colors(uv1[:, 0], 2 * np.pi)  # (N₁, 3)
    c2 = hsv_to_rgb_colors(uv2[:, 0], 2 * np.pi)  # (N₂, 3)

    return np.concatenate([c1, c2], axis=0).astype(np.float32)


def plot_fundamental_domain_coloring(ax, genus=1, num_points=100, tau1=1j, tau2=1j,
                                     neck_radius=0.3,
                                     disk_center_T1=(0.0, 0.0),
                                     disk_center_T2=None):
    """Plot the fundamental domain with the periodic coloring.

    For genus 2, draws the two parameter-space charts (T₁, T₂) with
    excised disks D₁ and D₂ indicated.

    Args:
        ax: Matplotlib axes
        genus: Surface genus (0, 1, or 2)
        num_points: Grid resolution
        tau1, tau2: Complex modular parameters for genus 2 tori
        neck_radius: Parameter-space disk radius δ
        disk_center_T1: (u₀, v₀) of D₁ in T₁ parameter space
        disk_center_T2: (u₀, v₀) of D₂ in T₂ parameter space (default (π, 0))
    """
    if genus == 2:
        if disk_center_T2 is None:
            disk_center_T2 = (np.pi, 0.0)
        _plot_connected_sum_domain(
            ax, tau1=tau1, tau2=tau2, neck_radius=neck_radius,
            disk_center_T1=disk_center_T1, disk_center_T2=disk_center_T2,
            num_points=num_points,
        )
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


def _plot_connected_sum_domain(ax, tau1=1j, tau2=1j, neck_radius=0.3,
                                disk_center_T1=(0.0, 0.0),
                                disk_center_T2=(np.pi, 0.0),
                                num_points=80):
    """
    Two-chart parameter domain for the genus-2 multi-chart architecture.

    Two charts arranged horizontally, each coloured by u ∈ [0,2π] to match
    the 3D surface colour scheme:
      T₁: [0,2π]² with disk D₁ excised at disk_center_T1
      T₂: [0,2π]² with disk D₂ excised at disk_center_T2
    """
    from matplotlib.patches import Circle

    if isinstance(tau1, (int, float)):
        tau1 = complex(0, float(tau1))
    if isinstance(tau2, (int, float)):
        tau2 = complex(0, float(tau2))
    tau1_re = tau1.real
    tau1_im = max(abs(tau1.imag), 0.1)
    tau2_re = tau2.real
    tau2_im = max(abs(tau2.imag), 0.1)

    N = num_points
    u_arr = np.linspace(0, 2 * np.pi, N)
    v_arr = np.linspace(0, 2 * np.pi, N)
    U, _ = np.meshgrid(u_arr, v_arr)

    def _rgb_from_u(U_grid):
        hue = U_grid / (2 * np.pi)
        h = hue * 6.0
        x = 1.0 - np.abs(h % 2.0 - 1.0)
        R_ = np.where((h >= 0) & (h < 1), 1., np.where((h >= 1) & (h < 2), x,
             np.where((h >= 4) & (h < 5), x, np.where((h >= 5) & (h < 6), 1., 0.))))
        G_ = np.where((h >= 0) & (h < 1), x, np.where((h >= 1) & (h < 3), 1.,
             np.where((h >= 3) & (h < 4), x, 0.)))
        B_ = np.where((h >= 2) & (h < 3), x, np.where((h >= 3) & (h < 5), 1.,
             np.where((h >= 5) & (h < 6), x, 0.)))
        return np.stack([R_, G_, B_], axis=-1)

    rgb = _rgb_from_u(U)
    L = 2 * np.pi          # chart side length
    gap = np.pi * 0.7      # horizontal gap between charts

    x0_T1, x1_T1 = 0.0, L
    x0_T2, x1_T2 = L + gap, 2 * L + gap

    # --- Coloured backgrounds ---
    ax.imshow(rgb, extent=[x0_T1, x1_T1, 0, L], origin='lower', aspect='auto',
              interpolation='nearest', zorder=1)
    ax.imshow(rgb, extent=[x0_T2, x1_T2, 0, L], origin='lower', aspect='auto',
              interpolation='nearest', zorder=1)

    # --- Chart borders ---
    for x0, x1 in [(x0_T1, x1_T1), (x0_T2, x1_T2)]:
        ax.plot([x0, x1, x1, x0, x0], [0, 0, L, L, 0], 'k-', lw=1.5, zorder=10)

    # --- Excised disks clipped to chart boundaries ---
    # Full circles are drawn at every periodic copy of the center, then clipped
    # to their chart rectangle so only the interior arc shows:
    #   T₁ disk at corner (0,0) → 4 quarter-circles at the 4 corners
    #   T₂ disk at edge  (π,0) → 2 semi-circles at bottom and top edges
    from matplotlib.patches import PathPatch, Circle
    from matplotlib.path import Path as MplPath

    def _chart_clip(x0, x1, y0, y1):
        verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
        p = PathPatch(MplPath(verts, codes), transform=ax.transData, visible=False)
        ax.add_patch(p)
        return p

    delta = neck_radius
    T1_clip = _chart_clip(x0_T1, x1_T1, 0, L)
    T2_clip = _chart_clip(x0_T2, x1_T2, 0, L)

    u0_T1, v0_T1 = disk_center_T1
    u0_T2, v0_T2 = disk_center_T2

    for cx, cy in [
        (x0_T1 + u0_T1,     v0_T1),
        (x0_T1 + u0_T1 + L, v0_T1),
        (x0_T1 + u0_T1,     v0_T1 + L),
        (x0_T1 + u0_T1 + L, v0_T1 + L),
    ]:
        c = Circle((cx, cy), radius=delta, fc='white', ec='black', lw=1.5, zorder=11)
        ax.add_patch(c)
        c.set_clip_path(T1_clip)

    for cx, cy in [
        (x0_T2 + u0_T2, v0_T2),
        (x0_T2 + u0_T2, v0_T2 + L),
    ]:
        c = Circle((cx, cy), radius=delta, fc='white', ec='black', lw=1.5, zorder=11)
        ax.add_patch(c)
        c.set_clip_path(T2_clip)

    # --- Chart labels ---
    tau1_str = (f'{tau1_re:.1f}+{tau1_im:.1f}i' if abs(tau1_re) > 0.02
                else f'{tau1_im:.1f}i')
    tau2_str = (f'{tau2_re:.1f}+{tau2_im:.1f}i' if abs(tau2_re) > 0.02
                else f'{tau2_im:.1f}i')
    ax.text((x0_T1 + x1_T1) / 2, L + 0.12, f'T₁  τ₁={tau1_str}',
            ha='center', va='bottom', fontsize=8)
    ax.text((x0_T2 + x1_T2) / 2, L + 0.12, f'T₂  τ₂={tau2_str}',
            ha='center', va='bottom', fontsize=8)

    # --- Axis setup ---
    ax.set_xlim(-0.4, x1_T2 + 0.4)
    ax.set_ylim(-0.5, L + 0.9)
    ax.set_title('Multi-Chart Parameter Domain', fontsize=10)
    ax.set_xlabel('u')
    ax.set_ylabel('v')

    xt = [0, np.pi, 2 * np.pi,
          x0_T2, x0_T2 + np.pi, x1_T2]
    xl = ['0', 'π', '2π', '0', 'π', '2π']
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7)
    ax.set_yticks([0, np.pi, 2 * np.pi])
    ax.set_yticklabels(['0', 'π', '2π'], fontsize=8)


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    For genus 0/1: 2×2 grid (total, willmore, regularity, all components).
    For genus 2: 2×3 grid — adds a dedicated gluing-loss panel and
    includes gluing in the combined "All Loss Components" plot.

    Args:
        history: Dictionary containing training history with keys:
                 'epoch', 'total_loss', 'willmore_energy', 'regularity',
                 and optionally 'gluing' (genus 2).
        output_path: Path to save the plot
    """
    genus = history.get('genus', 1)
    gluing = history.get('gluing', [])
    has_gluing = genus == 2 and len(gluing) > 0 and any(v > 0 for v in gluing)

    genus_names = {0: "Sphere/Ellipsoid", 1: "Torus", 2: "Double Torus"}
    epochs = history['epoch']

    from sampling import get_theoretical_minimum_willmore

    if has_gluing:
        # 2×3 grid for genus 2: add gluing panel
        fig, axes = plt.subplots(2, 3, figsize=(21, 10))
        ax_total    = axes[0, 0]
        ax_willmore = axes[0, 1]
        ax_gluing   = axes[0, 2]
        ax_reg      = axes[1, 0]
        # Span the combined plot over the last two columns of the bottom row
        axes[1, 1].remove()
        axes[1, 2].remove()
        ax_all = fig.add_subplot(2, 3, (5, 6))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_total    = axes[0, 0]
        ax_willmore = axes[0, 1]
        ax_reg      = axes[1, 0]
        ax_all      = axes[1, 1]
        ax_gluing   = None

    fig.suptitle(f"Training Progress - Genus {genus} ({genus_names.get(genus, 'Unknown')})",
                 fontsize=14, fontweight='bold')

    # --- Total Loss ---
    ax_total.plot(epochs, history['total_loss'], 'b-', linewidth=2)
    ax_total.set_xlabel('Epoch', fontsize=12)
    ax_total.set_ylabel('Total Loss (log scale)', fontsize=12)
    ax_total.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax_total.set_yscale('log')
    ax_total.grid(True, alpha=0.3, which='both')

    # --- Willmore Energy ---
    ax_willmore.plot(epochs, history['willmore_energy'], 'r-', linewidth=2)
    ax_willmore.set_xlabel('Epoch', fontsize=12)
    ax_willmore.set_ylabel('Willmore Energy (log scale)', fontsize=12)
    ax_willmore.set_title('Willmore Energy', fontsize=14, fontweight='bold')
    ax_willmore.set_yscale('log')
    ax_willmore.grid(True, alpha=0.3, which='both')
    try:
        theoretical_min = get_theoretical_minimum_willmore(genus)
        ax_willmore.axhline(y=theoretical_min, color='g', linestyle='--',
                            linewidth=2, alpha=0.7,
                            label=f'Theoretical min: {theoretical_min:.2f}')
        ax_willmore.legend(fontsize=10)
    except Exception:
        pass

    # --- Gluing Loss (genus 2 only) ---
    if ax_gluing is not None:
        ax_gluing.plot(epochs, gluing, color='purple', linewidth=2)
        ax_gluing.set_xlabel('Epoch', fontsize=12)
        ax_gluing.set_ylabel('Gluing Loss (log scale)', fontsize=12)
        ax_gluing.set_title('Gluing Loss (C⁰+C¹+C²)', fontsize=14, fontweight='bold')
        ax_gluing.set_yscale('log')
        ax_gluing.grid(True, alpha=0.3, which='both')

    # --- Regularity Loss ---
    ax_reg.plot(epochs, history['regularity'], 'g-', linewidth=2)
    ax_reg.set_xlabel('Epoch', fontsize=12)
    ax_reg.set_ylabel('Regularity Loss (log scale)', fontsize=12)
    ax_reg.set_title('Regularity Loss', fontsize=14, fontweight='bold')
    ax_reg.set_yscale('log')
    ax_reg.grid(True, alpha=0.3, which='both')

    # --- All Loss Components ---
    ax_all.plot(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss', alpha=0.7)
    ax_all.plot(epochs, history['willmore_energy'], 'r-', linewidth=2, label='Willmore Energy', alpha=0.7)
    ax_all.plot(epochs, history['regularity'], 'g-', linewidth=2, label='Regularity', alpha=0.7)
    if has_gluing:
        ax_all.plot(epochs, gluing, color='purple', linewidth=2,
                    label='Gluing Loss', alpha=0.7)
    ax_all.set_xlabel('Epoch', fontsize=12)
    ax_all.set_ylabel('Loss (log scale)', fontsize=12)
    ax_all.set_title('All Loss Components', fontsize=14, fontweight='bold')
    ax_all.set_yscale('log')
    ax_all.legend(fontsize=10, loc='best')
    ax_all.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

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
    config_path: str = 'configs/config_genus2.yaml',
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
    parser.add_argument("--config", type=str, default='configs/config_genus2.yaml',
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
