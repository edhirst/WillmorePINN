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

from model import create_embedding_model
from sampling import sample_parameters, compute_reference_willmore_energy


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
    config_path: str = 'hyperparameters.yaml',
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
    parser.add_argument("--config", type=str, default='hyperparameters.yaml',
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
