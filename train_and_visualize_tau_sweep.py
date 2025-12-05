"""
Train and visualize torus embeddings with different tau values.

This script trains multiple tau values sequentially, then generates a comparison visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os
from pathlib import Path

from model import create_embedding_model
from sampling import sample_parameters


def train_tau_value(tau_str, tau_label, output_dir, config):
    """
    Train a single tau value directly (no subprocess).
    
    Args:
        tau_str: String representation of tau (e.g., "1j", "0.5+0.866j")
        tau_label: Display label for tau (e.g., "i", "0.5+0.866i")
        output_dir: Directory to save the model
        config: Configuration dictionary
    
    Returns:
        Path to saved model checkpoint
    """
    print(f"\n{'='*60}")
    print(f"Training with τ = {tau_label}")
    print(f"{'='*60}\n")
    
    # Make a deep copy of config for this tau value to avoid cross-contamination
    import copy
    config_copy = copy.deepcopy(config)
    
    # Modify config for this tau value
    config_copy['sampling']['domain_params']['tau'] = tau_str
    
    # Get device
    device = torch.device('cpu')
    
    # DEBUG_TAU_SWEEP: Remove after tau sweep verification
    print(f"Supervised pretraining config: {config_copy['model']['supervised_pretraining']}")
    print(f"Use residual: {config_copy['model'].get('use_residual', False)}")
    print(f"Domain: {config_copy['sampling']['domain']}")
    
    # Create model with supervised pretraining (skip_init=False to enable training)
    model = create_embedding_model(config_copy, device, skip_init=False)
    
    # Save model
    model_path = os.path.join(output_dir, f'model_tau_{tau_label.replace("+", "_").replace(".", "_")}.pt')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config_copy,
        'tau': tau_str,
        'tau_label': tau_label
    }
    torch.save(checkpoint, model_path)
    print(f"Saved model to {model_path}\n")
    
    return model_path


def load_model_from_checkpoint(checkpoint_path: str, device=torch.device('cpu')):
    """Load a trained model from a checkpoint file."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get tau from checkpoint
    tau_str = checkpoint.get('tau', '1j')
    tau_label = checkpoint.get('tau_label', tau_str)
    if isinstance(tau_str, str):
        tau = complex(tau_str.replace(' ', ''))
    else:
        tau = tau_str
    
    config = checkpoint['config']
    
    # Create model
    model = create_embedding_model(config, device, skip_init=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tau, tau_label


def plot_torus_3d(ax, model, num_points=3000, domain='torus', device=torch.device('cpu'), 
                  title="", color='viridis', alpha=0.7):
    """Plot a torus in 3D."""
    # Sample points
    uv = sample_parameters(num_points, domain, device, torch.float32)
    
    # Get embedding
    with torch.no_grad():
        xyz = model(uv).cpu().numpy()
    
    # Color by position in fundamental domain (periodic)
    # Corners (0,0), (2π,0), (0,2π), (2π,2π) are blue
    # Center (π,π) is yellow
    uv_np = uv.cpu().numpy()
    u = uv_np[:, 0]
    v = uv_np[:, 1]
    
    # Use sine waves for periodic coloring
    # sin²(u/2) and sin²(v/2) are 0 at corners, 1 at center
    u_factor = np.sin(u / 2) ** 2
    v_factor = np.sin(v / 2) ** 2
    brightness = u_factor * v_factor  # Peak at center, 0 at corners
    
    # Map to blue (corners) → yellow (center)
    # Blue = (0, 0, 1), Yellow = (1, 1, 0)
    colors = np.stack([
        brightness,           # R: 0 at corners, 1 at center
        brightness,           # G: 0 at corners, 1 at center  
        1 - 0.5 * brightness  # B: 1 at corners, 0.5 at center
    ], axis=1)
    
    scatter = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=colors, alpha=alpha, s=2
    )
    
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
    
    return scatter


def visualize_tau_sweep(run_numbers, tau_labels, config_path='hyperparameters.yaml', 
                        num_points=3000, output_dir='logs'):
    """
    Visualize multiple torus embeddings with different tau values.
    
    Args:
        run_numbers: List of run numbers to visualize
        tau_labels: List of tau string labels for display
        config_path: Path to config file
        num_points: Number of points to sample
        output_dir: Directory to save outputs
    """
    device = torch.device('cpu')
def visualize_tau_sweep(model_paths, num_points=1000, output_dir='logs'):
    """
    Visualize multiple torus embeddings with different tau values.
    
    Args:
        model_paths: List of paths to model checkpoints
        num_points: Number of points to sample
        output_dir: Directory to save outputs
    """
    device = torch.device('cpu')
    
    # Load all models
    models_data = []
    for model_path in model_paths:
        try:
            model, tau, tau_label = load_model_from_checkpoint(model_path, device)
            models_data.append((model, tau, tau_label))
            print(f"Loaded {os.path.basename(model_path)}: τ = {tau_label}")
        except Exception as e:
            print(f"Could not load {model_path}: {e}")
    
    if not models_data:
        print("No models loaded!")
        return
    
    # Output goes directly to the specified directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    n_models = len(models_data)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig = plt.figure(figsize=(6*cols, 5*rows))
    
    for idx, (model, tau, tau_label) in enumerate(models_data):
        ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
        
        # Create title with tau value
        title = f"τ = {tau_label}"
        
        plot_torus_3d(ax, model, num_points, 'torus', device, title)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'tau_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to {output_path}")
    
    # Print summary
    print("\nTau Sweep Summary:")
    print("=" * 60)
    for model, tau, tau_label in models_data:
        print(f"τ = {tau_label:15s} | {tau}")
    print("=" * 60)


def main():
    """Main function to train and visualize tau sweep."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and visualize tori with different tau values")
    parser.add_argument('--config', type=str, default='hyperparameters.yaml',
                       help='Path to config file')
    parser.add_argument('--points', type=int, default=20000,
                       help='Number of points to sample for visualization')
    parser.add_argument('--output-dir', type=str, default='logs/tau_sweep',
                       help='Output directory for models and visualizations')
    
    args = parser.parse_args()
    
    # Define tau values to train
    # Keep |τ| close to 1 to avoid self-intersection
    # Re(τ) creates vertical tilt, Im(τ) controls minor radius
    # Keep |Re(τ)| < |Im(τ)| to ensure tilt doesn't overwhelm geometry
    tau_values = [
        ("1j", "i"),                    # Standard square torus (no tilt)
        ("0.3+0.95j", "0.3+0.95i"),    # Slight tilt
        ("0.5+0.87j", "0.5+0.87i"),    # Moderate tilt (60° angle)
        ("0.7+0.7j", "0.7+0.7i"),      # Balanced tilt (45° angle)
        ("-0.4+0.9j", "-0.4+0.9i"),    # Negative tilt (opposite direction)
        ("0.6+0.65j", "0.6+0.65i"),    # Strong tilt, slightly thin
    ]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("TAU SWEEP TRAINING")
    print("="*60)
    print(f"\nTraining {len(tau_values)} different tau values...")
    print("Each will be trained with supervised pretraining only (fast).\n")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for training
    config['model']['supervised_pretraining']['num_epochs'] = 100
    config['model']['supervised_pretraining']['num_points_per_epoch'] = 5000
    config['model']['supervised_pretraining']['batch_size'] = 512
    
    # Train each tau value and collect model paths
    model_paths = []
    for tau_str, tau_label in tau_values:
        model_path = train_tau_value(tau_str, tau_label, args.output_dir, config)
        if model_path:
            model_paths.append(model_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    if not model_paths:
        print("No models to visualize!")
        return
    
    # Visualize results
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60 + "\n")
    
    visualize_tau_sweep(
        model_paths,
        num_points=args.points,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()
