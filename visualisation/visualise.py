"""
Visualize the evolution of learned embeddings through training.

Plots 3D embeddings at different checkpoints to show how the surface
evolves toward the Willmore-optimal geometry.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import glob
from pathlib import Path

from model import create_embedding_model
from sampling import sample_parameters, get_domain_for_genus
from utils import get_next_run_number, plot_fundamental_domain_coloring, plot_surface_3d


def load_checkpoint_model(checkpoint_path, config, device):
    """Load model from checkpoint."""
    # Skip reference initialization when loading from checkpoint
    model = create_embedding_model(config, device, skip_init=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint.get('epoch', 0), checkpoint.get('loss', 0)


def get_highest_run_number(base_checkpoint_dir: str) -> int:
    """Get the highest run number from run directories."""
    if not os.path.exists(base_checkpoint_dir):
        return None
    
    # Find all run_* directories
    run_dirs = [d for d in os.listdir(base_checkpoint_dir) 
                if os.path.isdir(os.path.join(base_checkpoint_dir, d)) and d.startswith('run_')]
    
    if not run_dirs:
        return None
    
    # Extract numbers and find max
    run_numbers = []
    for d in run_dirs:
        try:
            num = int(d.split('_')[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(run_numbers) if run_numbers else None


def plot_fundamental_domain_image(output_path, num_points=200, genus=None, config_path='hyperparameters.yaml'):
    """
    Plot and save the fundamental domain with rainbow coloring.
    
    Args:
        output_path: Path to save the plot
        num_points: Resolution of the grid
        genus: Surface genus (0, 1, or 2). If None, read from config.
        config_path: Path to config file to determine genus if not specified
    """
    # Get genus from config if not provided
    if genus is None:
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            genus = config.get('topology', {}).get('genus', 1)
        except Exception:
            genus = 1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_fundamental_domain_coloring(ax, genus=genus, num_points=num_points)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved fundamental domain to {output_path}")
    
    plt.close()


def visualise_training_evolution(
    config_path='hyperparameters.yaml',
    checkpoint_dir='checkpoints',
    num_test_points=5000,
    output_path='logs/embedding_evolution.png',
    number_of_models=20
):
    """
    Visualize how the embedding evolves during training.
    
    Args:
        config_path: Path to configuration file
        checkpoint_dir: Directory containing checkpoints (can be base dir or specific run dir)
        num_test_points: Number of points to sample for visualization
        output_path: Where to save the visualization
        number_of_models: Number of models to plot (default: 20). If >= total models, plots all.
                         If a positive integer < total models, selects that many with even spacing,
                         always including first and last. Must be >= 2 if specified.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # Get topology configuration
    topology_config = config.get("topology", {})
    genus = topology_config.get("genus", 1)
    
    domain = get_domain_for_genus(genus)
    
    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}
    print(f"Visualizing training evolution for genus {genus} ({genus_names.get(genus, 'unknown')})")
    print(f"Domain: {domain}")
    print(f"Using {num_test_points} test points")
    print(f"Looking in: {checkpoint_dir}")
    
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
    
    # Verify checkpoint files contain valid epochs
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
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
    
    # Always use genus from checkpoint config for correct sampling
    # (Assume all checkpoints in a run have the same genus)
    first_checkpoint = torch.load(selected_checkpoints[0], map_location='cpu')
    config_ckpt = first_checkpoint.get('config', config)
    genus_ckpt = config_ckpt['topology']['genus']
    domain_ckpt = get_domain_for_genus(genus_ckpt)
    uv_test = sample_parameters(num_test_points, domain_ckpt, device, genus=genus_ckpt)
    
    # Create figure with subplots
    n_plots = len(selected_checkpoints)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}

    # First pass: load all models and collect xyz + title
    _plot_data = []
    for checkpoint_path in selected_checkpoints:
        checkpoint_name = Path(checkpoint_path).name
        print(f"  Loading {checkpoint_name}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config_ckpt = checkpoint.get('config', config)
        genus_ckpt = config_ckpt['topology']['genus']
        domain_ckpt = get_domain_for_genus(genus_ckpt)
        print(f"Embedding model created for genus {genus_ckpt} ({genus_names.get(genus_ckpt, 'unknown')})")
        print(f"  Domain: {domain_ckpt}")
        print(f"  Parameters: {sum(p.numel() for p in create_embedding_model(config_ckpt, device, skip_init=True).parameters())} trainable")
        model, epoch, loss = load_checkpoint_model(checkpoint_path, config_ckpt, device)
        with torch.no_grad():
            xyz = model(uv_test)
        if 'best' in checkpoint_path:
            title = f'Best Model\nEpoch {epoch}, W={loss:.2f}'
        elif 'latest' in checkpoint_path:
            title = f'Latest Model\nEpoch {epoch}, W={loss:.2f}'
        else:
            expected_epoch = extract_epoch(checkpoint_path) if checkpoint_name.startswith('checkpoint_epoch_') else epoch
            if expected_epoch != epoch:
                print(f"  WARNING: Checkpoint {checkpoint_name} has epoch {epoch} but filename suggests {expected_epoch}")
            title = f'Epoch {epoch}\nW={loss:.2f}'
        _plot_data.append((xyz, title, genus_ckpt))

    # Shared scale across all subplots
    _all_xyz = np.concatenate([d[0].detach().cpu().numpy() for d in _plot_data])
    _global_range = np.array([
        _all_xyz[:, 0].max() - _all_xyz[:, 0].min(),
        _all_xyz[:, 1].max() - _all_xyz[:, 1].min(),
        _all_xyz[:, 2].max() - _all_xyz[:, 2].min()
    ]).max() / 2.0

    for idx, (xyz, title, genus_ckpt) in enumerate(_plot_data):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        plot_surface_3d(ax, xyz, uv_test, title, genus=genus_ckpt, global_range=_global_range)
        ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")
    
    plt.close()


def visualise_single_model(
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
    # Load checkpoint and config
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', None)
    if config is None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    device = torch.device('cpu')
    genus = config['topology']['genus']
    domain = get_domain_for_genus(genus)
    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}
    print(f"Visualizing model from {checkpoint_path}")
    print(f"Embedding model created for genus {genus} ({genus_names.get(genus, 'unknown')})")
    print(f"  Domain: {domain}")
    # Load model
    model, epoch, loss = load_checkpoint_model(checkpoint_path, config, device)
    # Generate test data
    uv_test = sample_parameters(num_test_points, domain, device)
    # Compute embedding
    with torch.no_grad():
        xyz = model(uv_test)
    # Shared scale across all three views
    _xyz_np = xyz.detach().cpu().numpy()
    _global_range = np.array([
        _xyz_np[:, 0].max() - _xyz_np[:, 0].min(),
        _xyz_np[:, 1].max() - _xyz_np[:, 1].min(),
        _xyz_np[:, 2].max() - _xyz_np[:, 2].min()
    ]).max() / 2.0

    # Create figure with multiple views
    fig = plt.figure(figsize=(18, 6))
    angles = [(20, 45), (20, 135), (20, 225)]
    view_names = ['Front', 'Side', 'Back']
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        plot_surface_3d(ax, xyz, uv_test, f'{view_names[idx]} View', genus=genus, alpha=0.7, global_range=_global_range)
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(f'Best Model - Epoch {epoch}, Willmore Energy = {loss:.4f}', fontsize=14, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    """Main visualization function."""
    import argparse
    
    # Default paths relative to parent directory (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, 'hyperparameters.yaml')
    default_checkpoints = os.path.join(project_root, 'checkpoints')
    
    parser = argparse.ArgumentParser(description="Visualize learned embeddings")
    parser.add_argument('--config', type=str, default=default_config,
                       help='Path to configuration file')
    parser.add_argument('--checkpoints', type=str, default=default_checkpoints,
                       help='Base directory containing checkpoint runs')
    parser.add_argument('--run-number', type=int, default=None,
                       help='Specific run number to visualise (default: highest run number)')
    parser.add_argument('--mode', type=str, choices=['evolution', 'best', 'both'], default='both',
                       help='Visualisation mode')
    parser.add_argument('--points', type=int, default=5000,
                       help='Number of test points')
    parser.add_argument('--num-models', type=int, default=20,
                       help='Number of models to plot (default: 20). Must be >= 2 if specified.')
    
    args = parser.parse_args()
    
    # Determine which run to visualise
    if args.run_number is not None:
        run_number = args.run_number
    else:
        # Use highest run number
        run_number = get_highest_run_number(args.checkpoints)
        if run_number is None:
            print(f"Error: No run directories found in {args.checkpoints}")
            print(f"Expected directory structure: {args.checkpoints}/run_N/")
            return
    
    checkpoint_dir = os.path.join(args.checkpoints, f'run_{run_number}')
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Run {run_number} not found at {checkpoint_dir}")
        return
    
    print(f"Using run #{run_number}")
    
    # Set output directory to logs/run_# to match the run being visualised
    log_dir = os.path.join(project_root, 'logs', f'run_{run_number}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Load genus from checkpoint (try best model first, then latest, then first checkpoint)
    genus_from_checkpoint = None
    for checkpoint_name in ['best_model.pt', 'latest_model.pt']:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                genus_from_checkpoint = checkpoint.get('config', {}).get('topology', {}).get('genus')
                if genus_from_checkpoint is not None:
                    break
            except Exception:
                pass
    
    # If not found in best/latest, check first checkpoint
    if genus_from_checkpoint is None:
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')))
        if checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_files[0], map_location='cpu')
                genus_from_checkpoint = checkpoint.get('config', {}).get('topology', {}).get('genus')
            except Exception:
                pass
    
    # Generate fundamental domain coloring plot
    print("=" * 60)
    print("Generating Fundamental Domain Coloring")
    print("=" * 60)
    plot_fundamental_domain_image(
        output_path=os.path.join(log_dir, 'fundamental_domain.png'),
        genus=genus_from_checkpoint,
        config_path=args.config
    )
    
    if args.mode in ['evolution', 'both']:
        print("=" * 60)
        print("Visualizing Training Evolution")
        print("=" * 60)
        visualise_training_evolution(
            config_path=args.config,
            checkpoint_dir=checkpoint_dir,
            num_test_points=args.points,
            output_path=os.path.join(log_dir, 'embedding_evolution.png'),
            number_of_models=args.num_models
        )
    
    if args.mode in ['best', 'both']:
        print("\n" + "=" * 60)
        print("Visualizing Best Model")
        print("=" * 60)
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        latest_path = os.path.join(checkpoint_dir, 'latest_model.pt')
        
        out_name = 'best_embedding.png'
        out_name_latest = 'latest_embedding.png'
        if os.path.exists(best_path):
            visualise_single_model(
                checkpoint_path=best_path,
                config_path=args.config,
                num_test_points=args.points * 2,
                output_path=os.path.join(log_dir, out_name)
            )
        elif os.path.exists(latest_path):
            print(f"Best model not found, using latest model instead")
            visualise_single_model(
                checkpoint_path=latest_path,
                config_path=args.config,
                num_test_points=args.points * 2,
                output_path=os.path.join(log_dir, out_name_latest)
            )
        else:
            print(f"No models found at {best_path} or {latest_path}")


if __name__ == '__main__':
    main()
