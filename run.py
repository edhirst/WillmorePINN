"""
Training Script for Embedding-Based Willmore Energy Minimization

This script trains a neural network to learn an embedding φ: (u,v) → (x,y,z)
that minimizes the Willmore energy functional.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import yaml
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional

from model import create_embedding_model
from losses import create_embedding_loss
from sampling import sample_parameters, compute_reference_willmore_energy
from utils import plot_loss_curves


def parse_tau(tau_value) -> complex:
    """
    Parse tau parameter from config, which may be a string or complex number.
    
    Args:
        tau_value: Can be complex (1j), string ("1j", "0.5+0.866j"), or dict
    
    Returns:
        Complex number
    """
    if isinstance(tau_value, complex):
        return tau_value
    elif isinstance(tau_value, str):
        # Handle strings like "1j", "0.5+0.866j", etc.
        return complex(tau_value.replace(' ', ''))
    elif isinstance(tau_value, (int, float)):
        # Pure real number
        return complex(tau_value, 0)
    elif isinstance(tau_value, dict):
        # Handle dict format: {real: 0.5, imag: 0.866}
        return complex(tau_value.get('real', 0), tau_value.get('imag', 1))
    else:
        # Default to 1j
        return 1j


def get_device(config: dict) -> torch.device:
    """Determine which device to use."""
    device_config = config.get("device", "auto")
    
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using device: cuda")
        elif torch.backends.mps.is_available():
            print("Note: Using CPU. MPS is available but autodiff for fundamental")
            print("      forms requires full linalg support. For experiments, CPU is recommended.")
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            print("Using device: cpu")
    else:
        device = torch.device(device_config)
        print(f"Using device: {device_config}")
    
    return device


def get_next_run_number(base_checkpoint_dir: str) -> int:
    """Get the next run number by finding the highest existing run number."""
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    
    # Find all run_* directories
    run_dirs = [d for d in os.listdir(base_checkpoint_dir) 
                if os.path.isdir(os.path.join(base_checkpoint_dir, d)) and d.startswith('run_')]
    
    if not run_dirs:
        return 1
    
    # Extract numbers and find max
    run_numbers = []
    for d in run_dirs:
        try:
            num = int(d.split('_')[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(run_numbers) + 1 if run_numbers else 1


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    config: dict,
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"[Epoch {epoch}] Saved best model with Willmore energy: {loss:.6f}")
    
    # Save latest model
    latest_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    torch.save(checkpoint, latest_path)


def train_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    num_points: int,
    batch_size: int,
    domain: str,
    device: torch.device,
    dtype: torch.dtype,
    gradient_clip: Optional[float] = None,
    use_rotation_augmentation: bool = True
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    # Sample parameter space points
    uv = sample_parameters(num_points, domain, device, dtype)
    
    # Apply random z-axis rotation augmentation if enabled
    if use_rotation_augmentation and domain == 'torus':
        # Random rotation angle in [0, 2π)
        theta = torch.rand(1, device=device, dtype=dtype).item() * 2 * 3.14159265359
        # Shift u coordinate by theta (rotates around z-axis)
        uv[:, 0] = (uv[:, 0] + theta) % (2 * 3.14159265359)
    
    # Split into batches
    num_batches = (num_points + batch_size - 1) // batch_size
    epoch_losses = {
        'total': 0.0,
        'willmore': 0.0,
        'regularity': 0.0
    }
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_points)
        uv_batch = uv[start_idx:end_idx]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = loss_fn(model, uv_batch)
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses.keys():
            if key == 'total':
                epoch_losses[key] += loss_dict[key].item()
            elif key in loss_dict:
                epoch_losses[key] += loss_dict[key]
            else:
                epoch_losses[key] += loss_dict.get(key, 0.0)
    
    # Average losses over batches
    for key in epoch_losses.keys():
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def train(config_path: str = "hyperparameters.yaml", resume_from: Optional[str] = None):
    """Main training loop."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get device
    device = get_device(config)
    dtype = torch.float32 if config.get("dtype", "float32") == "float32" else torch.float64
    
    # Create model
    model = create_embedding_model(config, device)
    
    # Create loss function
    loss_fn = create_embedding_loss(config)
    loss_fn = loss_fn.to(device)
    
    # Create optimizer
    optimizer_config = config.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "adam").lower()
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    
    if optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get("betas", [0.9, 0.999])
        )
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=optimizer_config.get("momentum", 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Create scheduler
    scheduler_config = config["training"].get("scheduler", "cosine")
    if scheduler_config == "cosine":
        scheduler_params = config["training"].get("scheduler_params", {})
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("T_max", config["training"]["num_epochs"]),
            eta_min=scheduler_params.get("eta_min", 1e-5)
        )
    elif scheduler_config == "none":
        scheduler = None
    else:
        print(f"Warning: Scheduler '{scheduler_config}' not fully implemented, using cosine")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])
    
    # Training parameters
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    num_points = config["sampling"]["num_points"]
    domain = config["sampling"]["domain"]
    log_frequency = config["training"].get("log_frequency", 10)
    save_frequency = config["training"].get("save_frequency", 50)
    gradient_clip = config["training"].get("gradient_clip", None)
    
    # Output directories with run numbering
    base_checkpoint_dir = config["output"]["checkpoint_dir"]
    base_log_dir = config["output"]["log_dir"]
    
    # Get next run number and create run-specific directories
    run_number = get_next_run_number(base_checkpoint_dir)
    checkpoint_dir = os.path.join(base_checkpoint_dir, f'run_{run_number}')
    log_dir = os.path.join(base_log_dir, f'run_{run_number}')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting Training Run #{run_number}")
    print(f"{'='*60}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    
    # Load checkpoint if resuming
    start_epoch = 1
    best_willmore = float('inf')
    
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_willmore = checkpoint.get('loss', float('inf'))
    
    # Compute reference Willmore energy (only meaningful in residual mode)
    use_residual = config['model'].get('use_residual', True)
    ref_willmore = None
    
    if use_residual:
        try:
            domain_params = config['sampling'].get('domain_params', {})
            tau = parse_tau(domain_params.get('tau', 1j))
            uv_ref = sample_parameters(100, domain, device, dtype)
            ref_willmore = compute_reference_willmore_energy(uv_ref, domain, tau)
            print(f"\nReference surface Willmore energy: {ref_willmore:.6f}")
            print(f"Mode: Learning residual corrections from reference")
            if domain == 'torus':
                print(f"Torus modulus τ = {tau:.4f}")
        except Exception as e:
            print(f"Warning: Could not compute reference Willmore energy: {e}")
            pass
    else:
        domain_params = config['sampling'].get('domain_params', {})
        tau = parse_tau(domain_params.get('tau', 1j))
        uv_ref = sample_parameters(100, domain, device, dtype)
        ref_willmore = compute_reference_willmore_energy(uv_ref, domain, tau)
        print(f"\nMode: Learning full embedding from scratch (no reference)")
        print(f"Starting from reference geometry: Willmore energy = {ref_willmore:.6f}")
        if domain == 'torus':
            print(f"Torus modulus τ = {tau:.4f}")
    
    # Compute theoretical minimum for comparison
    if domain == 'torus':
        clifford_willmore = 2 * np.pi ** 2  # Clifford torus: W = 2π² ≈ 19.74
        print(f"Optimization target: {clifford_willmore:.6f} (Clifford minimum)")
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Number of points: {num_points}")
    print(f"Domain: {domain}, Parameter space: [0, 2π] × [0, 2π]\n")
    
    # Training history
    history = {
        'epoch': [],
        'total_loss': [],
        'willmore_energy': [],
        'regularity': [],
        'learning_rate': []
    }
    
    # Initialize epoch_losses in case num_epochs is 0
    epoch_losses = {
        'total': 0.0,
        'willmore': ref_willmore if ref_willmore else 0.0,
        'regularity': 0.0
    }
    
    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        # Update loss weights progressively (annealing)
        loss_fn.update_weights(epoch, num_epochs)
        
        # Train one epoch
        use_rotation_aug = config["sampling"].get("use_rotation_augmentation", True)
        epoch_losses = train_epoch(
            model, loss_fn, optimizer,
            num_points, batch_size, domain,
            device, dtype, gradient_clip,
            use_rotation_augmentation=use_rotation_aug
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['epoch'].append(epoch)
        history['total_loss'].append(epoch_losses['total'])
        history['willmore_energy'].append(epoch_losses['willmore'])
        history['regularity'].append(epoch_losses['regularity'])
        history['learning_rate'].append(current_lr)
        
        # Check if best model using the actual Willmore energy
        current_willmore = epoch_losses['willmore']
        is_best = current_willmore < best_willmore
        if is_best:
            best_willmore = current_willmore
        
        # Log progress
        if epoch % log_frequency == 0:
            print(f"Epoch [{epoch}/{num_epochs}] - LR: {current_lr:.6f}")
            print(f"  Loss Weights - Willmore:{loss_fn.willmore_weight:.3f} Regularity:{loss_fn.regularity_weight:.3f}")
            print(f"  Total Loss: {epoch_losses['total']:.6f}")
            print(f"  Willmore Energy: {epoch_losses['willmore']:.6f}")
            print(f"  Regularity: {epoch_losses['regularity']:.6f}")
            if domain == 'torus':
                ratio_to_optimal = epoch_losses['willmore'] / clifford_willmore
                print(f"  Ratio to Clifford minimum: {ratio_to_optimal:.4f}x")
            elif ref_willmore:
                ratio = epoch_losses['willmore'] / ref_willmore
                print(f"  Ratio to reference: {ratio:.4f}x")
        
        # Save checkpoint - save at regular intervals AND when we find a new best
        # Use separate condition to save regular checkpoints vs best model
        should_save_regular = (epoch % save_frequency == 0)
        if should_save_regular or is_best:
            save_checkpoint(
                model, optimizer, epoch,
                epoch_losses['willmore'], config,
                checkpoint_dir, is_best
            )
    
    # Save final model (only if we actually trained)
    if num_epochs > 0:
        save_checkpoint(
            model, optimizer, num_epochs,
            epoch_losses['willmore'], config,
            checkpoint_dir, is_best=False
        )
    else:
        # For num_epochs=0, save the pretrained model
        save_checkpoint(
            model, optimizer, 0,
            epoch_losses['willmore'], config,
            checkpoint_dir, is_best=False
        )
    
    # Save training history
    history_path = os.path.join(log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to {history_path}")
    
    # Plot loss curves
    loss_curves_path = os.path.join(log_dir, 'loss_curves.png')
    plot_loss_curves(history, loss_curves_path)
    
    print("\nTraining completed!")
    print(f"Best Willmore energy: {best_willmore:.6f}")
    if domain == 'torus':
        print(f"Clifford torus minimum: {clifford_willmore:.6f}")
        print(f"Ratio to optimal: {best_willmore / clifford_willmore:.4f}x")
        if use_residual and ref_willmore:
            print(f"Reference surface: {ref_willmore:.6f}")
    elif ref_willmore:
        print(f"Reference Willmore energy: {ref_willmore:.6f}")
        print(f"Ratio: {best_willmore / ref_willmore:.4f}x")
    print(f"Final epoch - Total Loss: {epoch_losses['total']:.6f}")
    print(f"             Willmore: {epoch_losses['willmore']:.6f}")
    print(f"             Regularity: {epoch_losses['regularity']:.6f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train an embedding network to minimize the Willmore energy functional"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hyperparameters.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Run training
    train(config_path=args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()
