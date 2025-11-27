"""
Visualize the evolution of learned embeddings through training.

Plots 3D embeddings at different checkpoints to show how the surface
evolves toward the Willmore-optimal geometry.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os
import glob
from pathlib import Path

from model import create_embedding_model
from sampling import sample_parameters


def load_checkpoint_model(checkpoint_path, config, device):
    """Load model from checkpoint."""
    # Skip reference initialization when loading from checkpoint
    model = create_embedding_model(config, device, skip_init=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('epoch', 0), checkpoint.get('loss', 0)


def plot_embedding_3d(ax, xyz, title, color='viridis', alpha=0.6):
    """Plot a 3D embedding."""
    xyz_np = xyz.detach().cpu().numpy()
    
    # Color by z-coordinate for better visualization
    colors = xyz_np[:, 2]
    
    scatter = ax.scatter(
        xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2],
        c=colors, cmap=color, alpha=alpha, s=1
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
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
    
    return scatter


def visualize_training_evolution(
    config_path='hyperparameters.yaml',
    checkpoint_dir='checkpoints',
    num_test_points=5000,
    output_path='logs/embedding_evolution.png',
    number_of_models=None
):
    """
    Visualize how the embedding evolves during training.
    
    Args:
        config_path: Path to configuration file
        checkpoint_dir: Directory containing checkpoints
        num_test_points: Number of points to sample for visualization
        output_path: Where to save the visualization
        number_of_models: Number of models to plot. If None or >= total models, plots all.
                         If a positive integer < total models, selects that many with even spacing,
                         always including first and last. Must be >= 2 if specified.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    domain = config['sampling']['domain']
    
    print(f"Visualizing training evolution for {domain}")
    print(f"Using {num_test_points} test points")
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    # Sort checkpoint files by epoch number (extract epoch from filename)
    def extract_epoch(filepath):
        filename = os.path.basename(filepath)
        # Extract epoch number from 'checkpoint_epoch_N.pt'
        try:
            epoch_str = filename.replace('checkpoint_epoch_', '').replace('.pt', '')
            return int(epoch_str)
        except ValueError:
            return 0
    
    checkpoint_files = sorted(checkpoint_files, key=extract_epoch)
    
    # Validate and apply number_of_models parameter
    total_checkpoints = len(checkpoint_files)
    
    if number_of_models is not None:
        # Validate that number_of_models is a positive integer
        if not isinstance(number_of_models, int) or number_of_models < 1:
            raise ValueError(f"number_of_models must be a positive integer, got: {number_of_models}")
        
        # If number_of_models is less than total and at least 2, select with even spacing
        if number_of_models < total_checkpoints:
            if number_of_models < 2:
                raise ValueError(f"number_of_models must be at least 2 to include first and last models, got: {number_of_models}")
            
            # Always include first and last, then select evenly spaced models in between
            if number_of_models == 2:
                selected_indices = [0, total_checkpoints - 1]
            else:
                # Use linspace to get evenly spaced indices including first and last
                selected_indices = np.linspace(0, total_checkpoints - 1, number_of_models, dtype=int)
                # Remove duplicates while preserving order
                selected_indices = sorted(list(dict.fromkeys(selected_indices)))
            
            selected_checkpoints = [checkpoint_files[i] for i in selected_indices]
        else:
            # Use all checkpoint files if number_of_models >= total
            selected_checkpoints = checkpoint_files
    else:
        # Use all checkpoint files if number_of_models is None
        selected_checkpoints = checkpoint_files
    
    # Add best model if it exists (will appear last)
    best_checkpoint = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_checkpoint):
        selected_checkpoints.append(best_checkpoint)
    
    print(f"\nVisualizing {len(selected_checkpoints)} checkpoints:")
    
    # Generate test data (same for all models)
    uv_test = sample_parameters(num_test_points, domain, device)
    
    # Create figure with subplots
    n_plots = len(selected_checkpoints)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    
    # Plot each checkpoint
    for idx, checkpoint_path in enumerate(selected_checkpoints):
        print(f"  Loading {Path(checkpoint_path).name}...")
        
        model, epoch, loss = load_checkpoint_model(checkpoint_path, config, device)
        
        # Compute embedding
        with torch.no_grad():
            xyz = model(uv_test)
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        # Determine title
        if 'best' in checkpoint_path:
            title = f'Best Model\nEpoch {epoch}, W={loss:.2f}'
        elif 'latest' in checkpoint_path:
            title = f'Latest Model\nEpoch {epoch}, W={loss:.2f}'
        else:
            title = f'Epoch {epoch}\nW={loss:.2f}'
        
        plot_embedding_3d(ax, xyz, title)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")
    
    plt.close()


def visualize_single_model(
    checkpoint_path='checkpoints/best_model.pt',
    config_path='hyperparameters.yaml',
    num_test_points=10000,
    output_path='logs/best_embedding.png'
):
    """
    Visualize a single model from multiple angles.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to configuration file
        num_test_points: Number of points to sample
        output_path: Where to save the visualization
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    domain = config['sampling']['domain']
    
    print(f"Visualizing model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    model, epoch, loss = load_checkpoint_model(checkpoint_path, config, device)
    
    # Generate test data
    uv_test = sample_parameters(num_test_points, domain, device)
    
    # Compute embedding
    with torch.no_grad():
        xyz = model(uv_test)
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(18, 6))
    
    angles = [(20, 45), (20, 135), (20, 225)]
    view_names = ['Front', 'Side', 'Back']
    
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        plot_embedding_3d(ax, xyz, f'{view_names[idx]} View', alpha=0.7)
        ax.view_init(elev=elev, azim=azim)
    
    fig.suptitle(f'Best Model - Epoch {epoch}, Willmore Energy = {loss:.4f}', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    plt.close()


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize learned embeddings")
    parser.add_argument('--config', type=str, default='hyperparameters.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoints', type=str, default='checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--mode', type=str, choices=['evolution', 'best', 'both'], default='both',
                       help='Visualization mode')
    parser.add_argument('--points', type=int, default=5000,
                       help='Number of test points')
    parser.add_argument('--num-models', type=int, default=None,
                       help='Number of models to plot (default: all). Must be >= 2 if specified.')
    
    args = parser.parse_args()
    
    if args.mode in ['evolution', 'both']:
        print("=" * 60)
        print("Visualizing Training Evolution")
        print("=" * 60)
        visualize_training_evolution(
            config_path=args.config,
            checkpoint_dir=args.checkpoints,
            num_test_points=args.points,
            number_of_models=args.num_models
        )
    
    if args.mode in ['best', 'both']:
        print("\n" + "=" * 60)
        print("Visualizing Best Model")
        print("=" * 60)
        best_path = os.path.join(args.checkpoints, 'best_model.pt')
        latest_path = os.path.join(args.checkpoints, 'latest_model.pt')
        
        if os.path.exists(best_path):
            visualize_single_model(
                checkpoint_path=best_path,
                config_path=args.config,
                num_test_points=args.points * 2
            )
        elif os.path.exists(latest_path):
            print(f"Best model not found, using latest model instead")
            visualize_single_model(
                checkpoint_path=latest_path,
                config_path=args.config,
                num_test_points=args.points * 2
            )
        else:
            print(f"No models found at {best_path} or {latest_path}")


if __name__ == '__main__':
    main()
