"""
Train and visualise surface embeddings after supervised pretraining for different genus values.

This script trains surfaces with supervised pretraining only (fast), then generates visualizations.
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
import copy
from pathlib import Path

from model import create_embedding_model
from sampling import sample_parameters, get_domain_for_genus
from utils import get_next_run_number


def hsv_to_rgb_colors(values: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Convert normalized values to rainbow RGB colors (HSV-like)."""
    v_norm = (values % period) / period
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


def train_surface(config: dict, label: str, output_dir: str, device: torch.device) -> str:
    """
    Train a single surface with supervised pretraining.
    
    Returns:
        Path to saved model checkpoint
    """
    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"{'='*60}\n")
    
    # Create model with supervised pretraining
    model = create_embedding_model(config, device, skip_init=False)
    
    # Generate safe filename from label
    safe_label = label.replace(" ", "_").replace("=", "_").replace("+", "_").replace(".", "_").replace(",", "_")
    model_path = os.path.join(output_dir, f'model_{safe_label}.pt')
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'label': label
    }
    torch.save(checkpoint, model_path)
    print(f"Saved model to {model_path}\n")
    
    return model_path


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a trained model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    label = checkpoint.get('label', os.path.basename(checkpoint_path))
    
    model = create_embedding_model(config, device, skip_init=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, label


def plot_surface_3d(ax, model, num_points: int, domain: str, genus: int,
                    device: torch.device, title: str, alpha: float = 0.7):
    """Plot a surface in 3D."""
    uv = sample_parameters(num_points, domain, device, torch.float32)
    
    with torch.no_grad():
        xyz = model(uv).cpu().numpy()
    
    uv_np = uv.cpu().numpy()
    v = uv_np[:, 1]
    
    # Determine period based on genus
    if genus == 0:
        period = np.pi
    elif genus == 2:
        period = 4 * np.pi
    else:
        period = 2 * np.pi
    
    colors = hsv_to_rgb_colors(v, period)
    
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, alpha=alpha, s=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10)
    
    # Equal aspect ratio
    max_range = np.array([
        xyz[:, 0].max() - xyz[:, 0].min(),
        xyz[:, 1].max() - xyz[:, 1].min(),
        xyz[:, 2].max() - xyz[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (xyz[:, 0].max() + xyz[:, 0].min()) * 0.5
    mid_y = (xyz[:, 1].max() + xyz[:, 1].min()) * 0.5
    mid_z = (xyz[:, 2].max() + xyz[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualise_supervised_genus0(output_dir: str, num_points: int, config_path: str, device: torch.device):
    """Train and visualize supervised ellipsoid embeddings (genus 0)."""
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Quantitative labels based on hyperparams
    ellipsoid_configs = [
        {'a': 1.0, 'b': 1.0, 'c': 1.0, 'label': 'a=1, b=1, c=1'},
        {'a': 1.5, 'b': 1.0, 'c': 1.0, 'label': 'a=1.5, b=1, c=1'},
        {'a': 1.0, 'b': 1.5, 'c': 1.0, 'label': 'a=1, b=1.5, c=1'},
        {'a': 1.5, 'b': 1.0, 'c': 0.7, 'label': 'a=1.5, b=1, c=0.7'},
        {'a': 2.0, 'b': 0.8, 'c': 0.8, 'label': 'a=2, b=0.8, c=0.8'},
    ]
    
    # Configure for supervised pretraining
    model_paths = []
    for cfg in ellipsoid_configs:
        config = copy.deepcopy(base_config)
        config['topology'] = config.get('topology', {})
        config['topology']['genus'] = 0
        config['topology']['ellipsoid'] = {'a': cfg['a'], 'b': cfg['b'], 'c': cfg['c']}
        config['model']['supervised_pretraining']['num_epochs'] = 100
        config['model']['supervised_pretraining']['num_points_per_epoch'] = 5000
        config['model']['supervised_pretraining']['batch_size'] = 512
        
        model_path = train_surface(config, f"genus0_{cfg['label']}", output_dir, device)
        model_paths.append((model_path, cfg['label']))
    
    # Visualize with fundamental domain
    fig = plt.figure(figsize=(15, 10))
    
    # Add fundamental domain coloring first
    ax_domain = fig.add_subplot(2, 3, 1)
    plot_fundamental_domain_coloring(ax_domain, genus=0)
    
    for idx, (model_path, label) in enumerate(model_paths):
        model, config, _ = load_model_from_checkpoint(model_path, device)
        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
        plot_surface_3d(ax, model, num_points, 'ellipsoid', 0, device, label)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'supervised_genus0_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved genus 0 comparison to {output_path}")
    plt.close()


def visualise_supervised_genus1(output_dir: str, num_points: int, config_path: str, device: torch.device):
    """Train and visualize supervised torus embeddings (genus 1)."""
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    tau_values = [
        ("1j", "τ = i"),
        ("0.3+0.95j", "τ = 0.3+0.95i"),
        ("0.5+0.87j", "τ = 0.5+0.87i"),
        ("0.7+0.7j", "τ = 0.7+0.7i"),
        ("-0.4+0.9j", "τ = -0.4+0.9i"),
    ]
    
    model_paths = []
    for tau_str, label in tau_values:
        config = copy.deepcopy(base_config)
        config['topology'] = config.get('topology', {})
        config['topology']['genus'] = 1
        config['topology']['torus'] = config['topology'].get('torus', {})
        config['topology']['torus']['tau'] = tau_str
        config['sampling'] = config.get('sampling', {})
        config['sampling']['domain_params'] = config['sampling'].get('domain_params', {})
        config['sampling']['domain_params']['tau'] = tau_str
        config['model']['supervised_pretraining']['num_epochs'] = 100
        config['model']['supervised_pretraining']['num_points_per_epoch'] = 5000
        config['model']['supervised_pretraining']['batch_size'] = 512
        
        model_path = train_surface(config, f"genus1_{label}", output_dir, device)
        model_paths.append((model_path, label, tau_str))
    
    # Visualize with fundamental domain
    fig = plt.figure(figsize=(15, 10))
    
    # Add fundamental domain coloring first
    ax_domain = fig.add_subplot(2, 3, 1)
    plot_fundamental_domain_coloring(ax_domain, genus=1)
    
    for idx, (model_path, label, tau_str) in enumerate(model_paths):
        model, config, _ = load_model_from_checkpoint(model_path, device)
        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
        plot_surface_3d(ax, model, num_points, 'torus', 1, device, label)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'supervised_genus1_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved genus 1 comparison to {output_path}")
    plt.close()


def visualise_supervised_genus2(output_dir: str, num_points: int, config_path: str, device: torch.device):
    """Train and visualize supervised double torus embeddings (genus 2)."""
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Quantitative labels based on hyperparams
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
    
    model_paths = []
    for cfg in dt_configs:
        label = cfg.pop('label')
        config = copy.deepcopy(base_config)
        config['topology'] = config.get('topology', {})
        config['topology']['genus'] = 2
        config['topology']['double_torus'] = cfg.copy()
        config['model']['supervised_pretraining']['num_epochs'] = 100
        config['model']['supervised_pretraining']['num_points_per_epoch'] = 5000
        config['model']['supervised_pretraining']['batch_size'] = 512
        
        model_path = train_surface(config, f"genus2_{label}", output_dir, device)
        model_paths.append((model_path, label))
        cfg['label'] = label  # Restore
    
    # Visualize with fundamental domain
    fig = plt.figure(figsize=(15, 10))
    
    # Add fundamental domain coloring first
    ax_domain = fig.add_subplot(2, 3, 1)
    plot_fundamental_domain_coloring(ax_domain, genus=2)
    
    for idx, (model_path, label) in enumerate(model_paths):
        model, config, _ = load_model_from_checkpoint(model_path, device)
        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
        plot_surface_3d(ax, model, num_points, 'double_torus', 2, device, label)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'supervised_genus2_comparison.png')
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
