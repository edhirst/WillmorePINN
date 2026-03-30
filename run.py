"""
Training Script for Embedding-Based Willmore Energy Minimization

This script trains a neural network to learn an embedding φ: (u,v) → (x,y,z)
that minimizes the Willmore energy functional.

Supported topologies:
- Genus 0 (sphere/ellipsoid): Minimizes to round sphere (W = 4π)
- Genus 1 (torus): Minimizes to Clifford torus (W = 2π²)
- Genus 2 (double torus): Minimizes toward Lawson surface (W ≈ 4π²)
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
from sampling import (
    sample_parameters, compute_reference_willmore_energy,
    get_domain_for_genus, get_theoretical_minimum_willmore,
    sample_torus_excluding_disk,
)
from utils import plot_loss_curves
import functools
print = functools.partial(print, flush=True)


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
    is_best: bool = False,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> None:
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    # Save scheduler state if provided (needed for checkpoint rollback)
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    
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
    use_rotation_augmentation: bool = True,
    genus: Optional[int] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    # --- Genus 2: multi-chart sampling & loss ---
    if genus == 2:
        return _train_epoch_genus2(
            model, loss_fn, optimizer, num_points, batch_size,
            device, dtype, gradient_clip,
        )

    # --- Genus 0 / 1: single-domain path ---
    # Sample parameter space points according to the topology
    uv = sample_parameters(num_points, domain, device, dtype, genus=genus)
    
    # Apply random z-axis rotation augmentation if enabled
    # Only applicable for torus (not ellipsoid due to poles)
    if use_rotation_augmentation and domain in ['torus']:
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
        optimizer.zero_grad(set_to_none=True)
        
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


def _train_epoch_genus2(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    num_points: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    gradient_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Train one epoch for genus 2 (two-chart)."""
    # Distribute points equally across T₁ and T₂.
    n_T1 = num_points // 2
    n_T2 = num_points - n_T1

    uv_T1 = sample_torus_excluding_disk(
        n_T1,
        disk_center=model.disk_center_T1,
        disk_radius=model.disk_radius,
        device=device, dtype=dtype,
    )
    uv_T2 = sample_torus_excluding_disk(
        n_T2,
        disk_center=model.disk_center_T2,
        disk_radius=model.disk_radius,
        device=device, dtype=dtype,
    )

    # The two-chart loss handles both charts in one forward call.
    # Batch by splitting each chart proportionally.
    num_batches = max(1, (num_points + batch_size - 1) // batch_size)
    b_T1 = max(1, n_T1 // num_batches)
    b_T2 = max(1, n_T2 // num_batches)

    epoch_losses = {'total': 0.0, 'willmore': 0.0, 'regularity': 0.0, 'gluing': 0.0,
                    'junction_r1': 0.0, 'junction_r2': 0.0}

    for batch_idx in range(num_batches):
        s1, e1 = batch_idx * b_T1, min((batch_idx + 1) * b_T1, n_T1)
        s2, e2 = batch_idx * b_T2, min((batch_idx + 1) * b_T2, n_T2)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = loss_fn.forward(model, uv_T1[s1:e1], uv_T2[s2:e2])
        loss = loss_dict['total']
        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        for key in epoch_losses:
            val = loss_dict.get(key, 0.0)
            if val is None:
                val = 0.0
            epoch_losses[key] += val.item() if hasattr(val, 'item') else float(val)

    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


def train(config_path: str = "hyperparameters.yaml", resume_from: Optional[str] = None, config_dict: Optional[dict] = None):
    """Main training loop."""
    # Load configuration
    if config_dict is not None:
        config = config_dict
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get device
    device = get_device(config)
    dtype = torch.float32 if config.get("dtype", "float32") == "float32" else torch.float64

    # Get topology configuration
    topology_config = config.get("topology", {})
    genus = topology_config.get("genus", 1)  # Default to torus for backward compatibility
    # If genus is overridden (e.g., via command line), ensure config is updated
    config['topology']['genus'] = genus

    # Print genus being used
    print(f"[INFO] Using genus: {genus}")

    # Validate genus
    if genus < 0:
        raise ValueError(f"Genus must be non-negative, got {genus}")
    if genus > 2:
        raise NotImplementedError(f"Genus {genus} is not supported. Only genus 0, 1, 2 are implemented.")

    # Get domain from genus
    domain = get_domain_for_genus(genus)
    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}

    print(f"\n{'='*60}")
    print(f"Willmore Energy Minimization - Genus {genus} ({genus_names.get(genus, 'unknown')})")
    print(f"{'='*60}")

    # Create model (pretraining runs inside create_embedding_model)
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
    
    # Print training configuration
    genus = config.get('topology', {}).get('genus', 1)
    adaptive_config = config['training'].get('adaptive_training', {})
    if adaptive_config.get('enabled', False):
        print(f"\nAdaptive training enabled for genus {genus}:")
        print(f"  - Regularity monitoring with threshold {adaptive_config.get('regularity_threshold', 0.5)}")
        if adaptive_config.get('checkpoint_rollback', False):
            print(f"  - Automatic checkpoint rollback on severe degradation (>{adaptive_config.get('severe_degradation_threshold', 2.0)}x)")
        print(f"  - Conservative learning: LR={config['training']['learning_rate']}, clip={config['training']['gradient_clip']}")
    
    # Create scheduler
    scheduler_config = config["training"].get("scheduler", "cosine")
    scheduler_params = config["training"].get("scheduler_params", {})
    if scheduler_config == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("T_max", config["training"]["num_epochs"]),
            eta_min=scheduler_params.get("eta_min", 1e-7)
        )
    elif scheduler_config == "cosine_warm_restarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params.get("T_0", 100),
            T_mult=scheduler_params.get("T_mult", 2),
            eta_min=scheduler_params.get("eta_min", 1e-7)
        )
    elif scheduler_config == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get("mode", "min"),
            factor=scheduler_params.get("factor", 0.5),
            patience=scheduler_params.get("patience", 10),
            min_lr=scheduler_params.get("min_lr", 1e-8)
        )
    elif scheduler_config == "none":
        scheduler = None
    else:
        print(f"Warning: Scheduler '{scheduler_config}' not recognised, using cosine")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"], eta_min=1e-7)
    
    # Training parameters
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    num_points = config["sampling"]["num_points"]
    log_frequency = config["training"].get("log_frequency", 10)
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
    
    print(f"\nStarting Training Run #{run_number}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    
    # Load checkpoint if resuming
    start_epoch = 1
    best_willmore = float('inf')
    
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_willmore = checkpoint.get('loss', float('inf'))
    
    # Compute reference Willmore energy
    use_residual = config['model'].get('use_residual', True)
    ref_willmore = None
    
    if genus == 2:
        # Multi-chart: no single-domain reference embedding
        print(f"\nMode: Two-chart genus-2 embedding (T₁ + T₂, direct gluing)")
        ref_willmore = 4 * np.pi**2
        print(f"Reference Willmore energy (Lawson ξ_{{2,1}}): {ref_willmore:.6f}")
    else:
        try:
            # Get topology-specific parameters for reference computation
            tau = None
            if genus == 1:
                torus_params = topology_config.get('torus', {})
                tau = parse_tau(torus_params.get('tau', '1j'))
            
            uv_ref = sample_parameters(100, domain, device, dtype, genus=genus)
            ref_willmore = compute_reference_willmore_energy(
                uv_ref, domain, tau=tau if tau else 1j, 
                genus=genus, topology_params=topology_config
            )
            
            if use_residual:
                print(f"\nReference surface Willmore energy: {ref_willmore:.6f}")
                print(f"Mode: Learning residual corrections from reference")
            else:
                print(f"\nMode: Learning full embedding from scratch")
                print(f"Initial reference geometry Willmore energy: {ref_willmore:.6f}")
            
            if genus == 1 and tau:
                print(f"Torus modulus τ = {tau:.4f}")
        except Exception as e:
            print(f"Warning: Could not compute reference Willmore energy: {e}")
    
    # Compute theoretical minimum for comparison
    try:
        theoretical_min = get_theoretical_minimum_willmore(genus)
        target_names = {
            0: "round sphere",
            1: "Clifford torus", 
            2: "Lawson surface ξ_{2,1}"
        }
        print(f"Optimization target: {theoretical_min:.6f} ({target_names.get(genus, 'optimal surface')})")
    except Exception as e:
        theoretical_min = None
        print(f"Note: Theoretical minimum not available for genus {genus}")
    
    # Print domain information
    domain_ranges = {
        "ellipsoid": "[0, 2π] × [0, π]",
        "torus": "[0, 2π] × [0, 2π]",
        "double_torus": "multi-chart (T₁ + T₂, direct gluing)"
    }
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Number of points: {num_points}")
    print(f"Domain: {domain}, Parameter space: {domain_ranges.get(domain, 'custom')}")
    
    print()
    
    # Training history
    history = {
        'epoch': [],
        'total_loss': [],
        'willmore_energy': [],
        'regularity': [],
        'gluing': [],          # gluing loss (genus 2 only; 0.0 for genus 0/1)
        'junction_r1': [],     # ℝ³ radius of T₁ gluing circle (genus 2 only)
        'junction_r2': [],     # ℝ³ radius of T₂ gluing circle (genus 2 only)
        'learning_rate': [],
        'genus': genus
    }
    
    # Initialize epoch_losses in case num_epochs is 0
    epoch_losses = {
        'total': 0.0,
        'willmore': ref_willmore if ref_willmore else 0.0,
        'regularity': 0.0
    }
    
    # Track rollbacks to prevent infinite loops
    rollback_count = 0
    max_rollbacks = config['training'].get('adaptive_training', {}).get('max_rollbacks', 3)

    # EMA of uncapped Willmore energy for rollback decisions.
    # Raw single-epoch MC estimates have high variance when sampling is junction-focused,
    # causing false-positive rollbacks.  The EMA smooths over ~5 epochs so only genuine
    # multi-epoch degradation triggers a rollback.
    _ema_alpha = config['training'].get('adaptive_training', {}).get('willmore_ema_alpha', 0.2)
    ema_willmore: Optional[float] = None
    
    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        # Adaptive weight adjustment based on regularity health
        adaptive_config = config['training'].get('adaptive_training', {})
        if epoch > 1 and len(history['regularity']) > 0:
            current_regularity = history['regularity'][-1]
            loss_fn.update_weights(epoch, num_epochs, current_regularity, adaptive_config)
        else:
            loss_fn.update_weights(epoch, num_epochs, adaptive_config=adaptive_config)

        # Frequency curriculum (NeRF-style coarse-to-fine).
        # Progressively activates higher Fourier bands so the network first
        # learns the global shape before fitting high-curvature details at
        # the bridge junction.  Only meaningful when use_spectral_features=True
        # and the periodic_layer supports set_active_frequencies (PeriodicEmbedding).
        freq_curriculum = config.get('model', {}).get('freq_curriculum', {})
        freq_warmup_epochs = freq_curriculum.get('warmup_epochs', 0)
        _pl = getattr(model, 'periodic_layer', None)
        if freq_warmup_epochs > 0 and _pl is not None and hasattr(_pl, 'set_active_frequencies'):
            total_freqs = _pl.num_frequencies
            freq_start = freq_curriculum.get('start_freqs', 1)
            if epoch <= freq_warmup_epochs:
                t = (epoch - 1) / max(freq_warmup_epochs - 1, 1)  # 0 at e=1, 1 at e=warmup
                num_active = freq_start + round((total_freqs - freq_start) * t)
                _pl.set_active_frequencies(num_active)
            else:
                _pl.set_active_frequencies(total_freqs)

        # Train one epoch
        use_rotation_aug = config["sampling"].get("use_rotation_augmentation", True)

        epoch_losses = train_epoch(
            model, loss_fn, optimizer,
            num_points, batch_size, domain,
            device, dtype, gradient_clip,
            use_rotation_augmentation=use_rotation_aug,
            genus=genus,
        )

        # --- Unbiased Willmore evaluation ---
        # epoch_losses['willmore'] was computed from junction-focused training samples,
        # which inflate the MC estimate when H is large at the junctions.  Evaluate on
        # a separate uniform sample so that best-model selection, rollback decisions,
        # and training history all reflect the true Willmore energy.
        #
        # When print_eval=False, the full eval only runs on logging/saving epochs;
        # other epochs use the training-sample willmore as a cheap proxy.
        eval_num_points = config["sampling"].get("eval_num_points", 5000)
        print_eval = config["sampling"].get("print_eval", True)
        should_eval = print_eval or (epoch % log_frequency == 0)

        if should_eval:
            if genus == 2:
                # Two-chart eval: sample each torus chart uniformly, sum Willmore energies
                n_eval_T1 = eval_num_points // 2
                n_eval_T2 = eval_num_points - n_eval_T1
                eval_uv_T1 = sample_torus_excluding_disk(
                    n_eval_T1, disk_center=model.disk_center_T1,
                    disk_radius=model.disk_radius, device=device, dtype=dtype)
                eval_uv_T2 = sample_torus_excluding_disk(
                    n_eval_T2, disk_center=model.disk_center_T2,
                    disk_radius=model.disk_radius, device=device, dtype=dtype)
                model.eval()
                eval_willmore = loss_fn.eval_willmore_batched(
                    model, eval_uv_T1, eval_uv_T2,
                    chunk_size=batch_size,
                )
                model.train()
            else:
                eval_uv = sample_parameters(eval_num_points, domain, device, dtype, genus=genus)
                model.eval()
                # Evaluate in 10 chunks to reduce peak autograd graph size
                chunk = (eval_num_points + 9) // 10
                total_weighted = 0.0
                for _start in range(0, eval_num_points, chunk):
                    _end = min(_start + chunk, eval_num_points)
                    total_weighted += loss_fn.willmore_loss(model, eval_uv[_start:_end])[1] * (_end - _start)
                eval_willmore = total_weighted / eval_num_points
                model.train()
        else:
            eval_willmore = epoch_losses['willmore']

        # Helper to execute a rollback: loads a checkpoint, halves LR, rescales scheduler.
        def _do_rollback(checkpoint_path: str, lr_scale: float = 0.5) -> bool:
            """Load checkpoint, apply lr_scale to LR and scheduler base_lrs. Returns True on success."""
            if not os.path.exists(checkpoint_path):
                return False
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if scheduler is not None and 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            for pg in optimizer.param_groups:
                pg['lr'] *= lr_scale
            if scheduler is not None and hasattr(scheduler, 'base_lrs'):
                scheduler.base_lrs = [lr * lr_scale for lr in scheduler.base_lrs]
            return True

        rolled_back = False

        # --- Check 1: Regularity degradation ---
        # Guard with an absolute minimum so near-zero regularity never spuriously fires.
        # Regularity ≈ 1e-9 (numerically inactive) must not trigger this.
        if adaptive_config.get('checkpoint_rollback', False) and epoch > 5 and len(history['regularity']) >= 5:
            reg_abs_min = adaptive_config.get('regularity_abs_threshold', 1e-6)
            threshold = adaptive_config.get('severe_degradation_threshold', 4.0)
            recent_window = min(5, len(history['regularity']))
            baseline_regularity = min(history['regularity'][-recent_window:])
            current_regularity = epoch_losses['regularity']

            if (current_regularity > reg_abs_min
                    and current_regularity > baseline_regularity * threshold):
                if rollback_count < max_rollbacks:
                    rollback_count += 1
                    print(f"\n🚨 REGULARITY DEGRADATION (Rollback {rollback_count}/{max_rollbacks})")
                    print(f"   Current: {current_regularity:.2e}, Baseline: {baseline_regularity:.2e}")
                    ok = _do_rollback(os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch-1}.pt"))
                    if ok:
                        loss_fn.regularity_weight *= 2.0
                        print(f"   LR → {optimizer.param_groups[0]['lr']:.2e}, regularity_weight → {loss_fn.regularity_weight:.2f}")
                        rolled_back = True
                    else:
                        print(f"   Warning: rollback checkpoint not found, continuing")
                else:
                    print(f"\n⚠️  Max regularity rollbacks reached. Permanently reducing LR.")
                    for pg in optimizer.param_groups:
                        pg['lr'] *= 0.1
                    if scheduler is not None and hasattr(scheduler, 'base_lrs'):
                        scheduler.base_lrs = [lr * 0.1 for lr in scheduler.base_lrs]
                    loss_fn.regularity_weight *= 3.0
                    rollback_count = 0

        # --- EMA update (unconditional) ---
        # Keep a smoothed estimate of W before any rollback checks so that all
        # downstream guards see a consistent, low-variance signal.
        ema_willmore = (eval_willmore if ema_willmore is None
                        else _ema_alpha * eval_willmore + (1.0 - _ema_alpha) * ema_willmore)

        # --- Check 2: Willmore spike rollback ---
        # If W spikes far above the running best, revert to the best checkpoint.
        willmore_spike_threshold = adaptive_config.get('willmore_spike_threshold', 0.0)
        if (not rolled_back
                and adaptive_config.get('checkpoint_rollback', False)
                and willmore_spike_threshold > 0
                and epoch > 5
                and best_willmore < float('inf')
                and rollback_count < max_rollbacks):
            if ema_willmore > best_willmore * willmore_spike_threshold:
                rollback_count += 1
                print(f"\n🚨 WILLMORE SPIKE (Rollback {rollback_count}/{max_rollbacks})")
                print(f"   EMA W={ema_willmore:.1f}, eval W={eval_willmore:.1f}, best W={best_willmore:.1f}")
                ok = _do_rollback(os.path.join(checkpoint_dir, 'best_model.pt'), lr_scale=0.5)
                if ok:
                    ema_willmore = best_willmore  # reset EMA to best after rollback
                    print(f"   Reverted to best model. LR \u2192 {optimizer.param_groups[0]['lr']:.2e}")
                    rolled_back = True
                else:
                    print(f"   Warning: best_model.pt not found, continuing")

        # --- Check 3: Topology collapse (genus-2 only) ---
        # The Willmore conjecture (Marques-Neves 2012) guarantees W ≥ 4π² ≈ 39.48
        # for any smooth embedded genus-2 surface.  When the EMA of W falls below
        # this threshold after the warmup period, the surface has lost genus-2
        # topology (degenerated to a sphere-like dumbbell).  Roll back to the
        # last good checkpoint and boost gluing + regularity weights to stabilise
        # the topology before the Willmore gradient dominates again.
        warmup_epochs_topo = adaptive_config.get('willmore_warmup_epochs', 0)
        topo_floor = adaptive_config.get('willmore_topology_floor', 4.0 * np.pi ** 2)
        if (not rolled_back
                and genus == 2
                and adaptive_config.get('checkpoint_rollback', False)
                and epoch > max(5, warmup_epochs_topo)
                and ema_willmore < topo_floor
                and rollback_count < max_rollbacks):
            rollback_count += 1
            print(f"\n\U0001f6a8 TOPOLOGY COLLAPSE (Rollback {rollback_count}/{max_rollbacks})")
            print(f"   EMA W={ema_willmore:.2f} < genus-2 floor {topo_floor:.2f}")
            ckpt_path = os.path.join(checkpoint_dir, 'best_model.pt')
            ok = _do_rollback(ckpt_path, lr_scale=0.5)
            if ok:
                ema_willmore = topo_floor  # reset EMA so guard doesn't immediately re-fire
                loss_fn.regularity_weight = min(loss_fn.regularity_weight * 2.0,
                                                loss_fn.initial_regularity_weight * 8.0)
                loss_fn.gluing_weight = min(loss_fn.gluing_weight * 2.0,
                                            loss_fn.initial_willmore_weight * 40.0)
                print(f"   Reverted. regularity_weight\u2192{loss_fn.regularity_weight:.1f}, "
                      f"gluing_weight\u2192{loss_fn.gluing_weight:.1f}")
                print(f"   LR \u2192 {optimizer.param_groups[0]['lr']:.2e}")
                rolled_back = True
            else:
                print(f"   Warning: best_model.pt not found, continuing")

        # Skip recording this failed epoch if we rolled back
        if rolled_back:
            continue
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_losses['total'])
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history — use eval_willmore (unbiased) for the logged energy
        history['epoch'].append(epoch)
        history['total_loss'].append(epoch_losses['total'])
        history['willmore_energy'].append(eval_willmore)
        history['regularity'].append(epoch_losses['regularity'])
        history['gluing'].append(epoch_losses.get('gluing', 0.0))
        history['junction_r1'].append(epoch_losses.get('junction_r1', 0.0))
        history['junction_r2'].append(epoch_losses.get('junction_r2', 0.0))
        history['learning_rate'].append(current_lr)

        # Best-model check: only update on epochs where a proper eval was run
        is_best = should_eval and (eval_willmore < best_willmore)
        if is_best:
            best_willmore = eval_willmore
        
        # Log progress
        if epoch % log_frequency == 0:
            print(f"Epoch [{epoch}/{num_epochs}] - LR: {current_lr:.6f}")
            print(f"  Loss Weights - Willmore:{loss_fn.willmore_weight:.3f} Regularity:{loss_fn.regularity_weight:.3f}")
            print(f"  Total Loss: {epoch_losses['total']:.6f}")
            print(f"  Willmore Energy (eval): {eval_willmore:.6f}")
            print(f"  Willmore Energy (train sample): {epoch_losses['willmore']:.6f}")
            print(f"  Regularity: {epoch_losses['regularity']:.6f}")
            if genus == 2:
                print(f"  Gluing loss: {epoch_losses.get('gluing', 0.0):.6f}")
                if epoch_losses.get('junction_r1', 0.0) > 0 or epoch_losses.get('junction_r2', 0.0) > 0:
                    print(f"  Junction radii: r1={epoch_losses.get('junction_r1', 0.0):.4f}  r2={epoch_losses.get('junction_r2', 0.0):.4f}")

            # Gradient-norm balance diagnostic (GradNorm-inspired, Chen et al. 2018).
            # Only for genus 0/1 (single-chart); genus 2 uses multi-chart loss.
            if genus != 2:
                try:
                    model.zero_grad(set_to_none=True)
                    _diag_uv = sample_parameters(min(1000, num_points), domain, device, dtype, genus=genus)
                    _w_tensor, _ = loss_fn.willmore_loss(model, _diag_uv)
                    _w_tensor = loss_fn.willmore_weight * _w_tensor
                    _w_tensor.backward()
                    _gnorm_w = sum(p.grad.detach().norm(2).item() ** 2
                                   for p in model.parameters() if p.grad is not None) ** 0.5
                    model.zero_grad(set_to_none=True)
                    _r_tensor = loss_fn.regularity_weight * loss_fn.regularity_loss(model, _diag_uv)
                    _r_tensor.backward()
                    _gnorm_r = sum(p.grad.detach().norm(2).item() ** 2
                                   for p in model.parameters() if p.grad is not None) ** 0.5
                    model.zero_grad(set_to_none=True)
                    print(f"  Grad norms — Willmore: {_gnorm_w:.3e}  Regularity: {_gnorm_r:.3e}  "
                          f"ratio W/R: {(_gnorm_w / (_gnorm_r + 1e-12)):.1f}")
                except Exception:
                    pass  # diagnostics must never abort training

            # Print topology-specific info
            if theoretical_min:
                ratio_to_optimal = eval_willmore / theoretical_min
                print(f"  Ratio to theoretical minimum: {ratio_to_optimal:.4f}x")

            # Report active frequencies for frequency curriculum
            if freq_warmup_epochs > 0 and _pl is not None and hasattr(_pl, 'set_active_frequencies'):
                n_active = int(_pl.freq_alphas.sum().item())
                print(f"  Active Fourier bands: {n_active}/{_pl.num_frequencies}")

        # Save checkpoint at log frequency and when a new best is found
        should_save_regular = (epoch % log_frequency == 0)
        if should_save_regular or is_best:
            save_checkpoint(
                model, optimizer, epoch,
                eval_willmore, config,
                checkpoint_dir, is_best, scheduler
            )
    
    # Save final model (only if we actually trained)
    if num_epochs > 0:
        save_checkpoint(
            model, optimizer, num_epochs,
            eval_willmore, config,
            checkpoint_dir, is_best=False, scheduler=scheduler
        )
    else:
        # For num_epochs=0, save the pretrained model
        save_checkpoint(
            model, optimizer, 0,
            eval_willmore, config,
            checkpoint_dir, is_best=False, scheduler=scheduler
        )
    
    # Save training history
    history_path = os.path.join(log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to {history_path}")
    
    # Plot loss curves
    loss_curves_path = os.path.join(log_dir, 'loss_curves.png')
    plot_loss_curves(history, loss_curves_path)
    
    # Final summary
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Genus: {genus} ({genus_names.get(genus, 'unknown')})")
    print(f"Best Willmore energy: {best_willmore:.6f}")
    
    if theoretical_min:
        print(f"Theoretical minimum: {theoretical_min:.6f}")
        print(f"Ratio to optimal: {best_willmore / theoretical_min:.4f}x")
    
    if ref_willmore:
        print(f"Initial reference energy: {ref_willmore:.6f}")
        print(f"Improvement ratio: {ref_willmore / best_willmore:.4f}x")
    
    print(f"\nFinal epoch metrics:")
    print(f"  Total Loss: {epoch_losses['total']:.6f}")
    print(f"  Willmore: {epoch_losses['willmore']:.6f}")
    print(f"  Regularity: {epoch_losses['regularity']:.6f}")

    return history


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
    parser.add_argument(
        "--genus",
        type=int,
        default=None,
        help="Override genus from config file (0=sphere, 1=torus, 2=double torus)"
    )

    args = parser.parse_args()

    # If genus is provided, override genus in config before training
    if args.genus is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        if 'topology' not in config:
            config['topology'] = {}
        config['topology']['genus'] = args.genus
        print(f"[INFO] Overriding genus from command line: genus = {args.genus}")
        train(config_dict=config, resume_from=args.resume)
    else:
        train(config_path=args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()
