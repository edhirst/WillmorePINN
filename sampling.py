"""
Parameter Space Sampling for Embedding Learning

This module provides functions to sample points in parameter space (u,v)
for various topologies. The neural network will learn the embedding to R³.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def sample_rectangular_domain(
    num_points: int,
    u_range: Tuple[float, float] = (0, 2*np.pi),
    v_range: Tuple[float, float] = (0, 2*np.pi),
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Sample points uniformly from a rectangular domain [u_min, u_max] × [v_min, v_max].
    
    This is appropriate for surfaces with doubly-periodic parameter spaces like
    torus and Klein bottle, both with (u,v) ∈ [0, 2π] × [0, 2π].
    
    Args:
        num_points: Number of points to sample
        u_range: Range for u parameter (min, max)
        v_range: Range for v parameter (min, max)
        device: Device to place tensor on
        dtype: Data type for tensor
    
    Returns:
        Parameter coordinates of shape (num_points, 2)
    """
    u_min, u_max = u_range
    v_min, v_max = v_range
    
    u = torch.rand(num_points, device=device, dtype=dtype) * (u_max - u_min) + u_min
    v = torch.rand(num_points, device=device, dtype=dtype) * (v_max - v_min) + v_min
    
    return torch.stack([u, v], dim=1)


def sample_sphere_parameters(
    num_points: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Sample points in parameter space for a sphere with area-weighting.
    
    For a sphere: u ∈ [0, 2π] (azimuthal), v ∈ [0, π] (polar)
    Uses area-weighted sampling: cos(v) uniform in [-1, 1] ensures uniform
    point distribution on the sphere surface (avoids pole clustering).
    
    Args:
        num_points: Number of points to sample
        device: Device to place tensor on
        dtype: Data type for tensor
    
    Returns:
        Parameter coordinates of shape (num_points, 2)
    """
    u = torch.rand(num_points, device=device, dtype=dtype) * 2 * np.pi
    cos_v = torch.rand(num_points, device=device, dtype=dtype) * 2 - 1
    v = torch.acos(cos_v)
    
    return torch.stack([u, v], dim=1)


def sample_parameters(
    num_points: int,
    domain: str = "torus",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Sample points in parameter space for the specified domain.
    
    Args:
        num_points: Number of points to sample
        domain: Type of surface ('torus', 'sphere', 'klein_bottle')
        device: Device to place tensor on
        dtype: Data type for tensor
    
    Returns:
        Parameter coordinates of shape (num_points, 2)
    """
    if domain.lower() == "torus":
        return sample_rectangular_domain(num_points, (0, 2*np.pi), (0, 2*np.pi), device, dtype)
    elif domain.lower() == "sphere":
        return sample_sphere_parameters(num_points, device, dtype)
    elif domain.lower() == "klein_bottle":
        return sample_rectangular_domain(num_points, (0, 2*np.pi), (0, 2*np.pi), device, dtype)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def transform_square_to_parallelogram(
    uv: torch.Tensor,
    tau: complex = 1j,
    max_height: Optional[float] = None
) -> torch.Tensor:
    """
    Transform uniform samples from [0, 2π]×[0, 2π] to a parallelogram fundamental domain.
    
    The parallelogram is defined by:
    - One edge along [0, 2π] on the real axis
    - Other edge from 0 to 2π*τ (scaled if max_height is set)
    
    Args:
        uv: Parameter coordinates on [0, 2π]×[0, 2π] (batch_size, 2)
        tau: Complex modulus defining the torus shape (default: 1j for square torus)
        max_height: Optional maximum height to scale tau (clips Im(τ) if needed)
    
    Returns:
        Transformed coordinates (batch_size, 2) in parallelogram domain
    """
    u, v = uv[:, 0], uv[:, 1]
    
    # Normalize to [0, 1] × [0, 1]
    u_norm = u / (2 * np.pi)
    v_norm = v / (2 * np.pi)
    
    # Apply max_height constraint if specified
    if max_height is not None:
        tau_height = abs(tau.imag)
        if tau_height > max_height:
            # Scale tau to have the specified max height while preserving aspect ratio
            scale = max_height / tau_height
            tau = complex(tau.real * scale, tau.imag * scale)
    
    # Affine transformation: (u, v) -> u * (2π, 0) + v * 2π * τ
    # This maps [0, 1]×[0, 1] to parallelogram with vertices at 0, 2π, 2πτ, 2π(1+τ)
    tau_real = tau.real
    tau_imag = tau.imag
    
    u_new = u_norm * 2 * np.pi + v_norm * 2 * np.pi * tau_real
    v_new = v_norm * 2 * np.pi * tau_imag
    
    return torch.stack([u_new, v_new], dim=1)


def get_flat_torus_embedding(
    uv: torch.Tensor,
    tau: complex = 1j
) -> torch.Tensor:
    """
    Create a flat torus embedding in R³ with complex modulus τ.
    
    For a flat torus, we embed the parallelogram [0, 2π] × [0, 2π*τ] into R³
    by wrapping it around to form a torus. The embedding maps:
    - The u-direction (along [0, 2π]) wraps around the major circle
    - The v-direction (along [0, 2π*τ]) wraps around the minor circle with twist
    - Re(τ) creates a helical twist in the minor circle as we go around the major circle
    - Im(τ) controls the minor radius
    
    Args:
        uv: Parameter coordinates (batch_size, 2) on the parallelogram domain
        tau: Complex modulus (default: 1j gives standard circular cross-section)
             Re(τ) controls twist/shear, Im(τ) controls minor radius
    
    Returns:
        Embedding coordinates (batch_size, 3) in R³
    """
    u, v = uv[:, 0], uv[:, 1]
    
    # Extract real and imaginary parts of tau
    tau_real = tau.real
    tau_imag = tau.imag
    
    # Compute radii from the fundamental domain
    # Major radius: based on the horizontal period (2π)
    # Minor radius: based on the vertical period (2π * |Im(τ)|)
    R = 1.0 / (2 * np.pi)  # Major radius scales inversely with period
    r = abs(tau_imag) / (2 * np.pi)  # Minor radius from imaginary part of τ
    
    # Renormalize v to [0, 2π] for standard torus parametrization
    v_normalized = v / abs(tau_imag) if abs(tau_imag) > 1e-10 else v
    
    # Scale to reasonable size
    scale = 3.0  # Scaling factor for visualization
    R_scaled = scale * R
    r_scaled = scale * r
    
    # Standard torus embedding
    x = (R_scaled + r_scaled * torch.cos(v_normalized)) * torch.cos(u)
    y = (R_scaled + r_scaled * torch.cos(v_normalized)) * torch.sin(u)
    z = r_scaled * torch.sin(v_normalized)
    
    # Add distortion from xy-plane based on Re(τ)
    # Creates a "tilted" or "wavy" torus that deviates from horizontal
    # The distortion is proportional to Re(τ) and varies around the major circle
    distortion_strength = tau_real * 0.3  # Scale factor to keep distortion reasonable
    z_distortion = distortion_strength * torch.cos(u) * (R_scaled + r_scaled * torch.cos(v_normalized))
    z = z + z_distortion
    
    return torch.stack([x, y, z], dim=1)


def get_reference_embedding(
    uv: torch.Tensor,
    domain: str = "torus",
    tau: complex = 1j,
    max_height: Optional[float] = None
) -> torch.Tensor:
    """
    Get reference embedding for comparison (analytical parametrization).
    
    Args:
        uv: Parameter coordinates (batch_size, 2) on [0, 2π]×[0, 2π]
        domain: Type of surface ('torus' or 'sphere')
        tau: Complex modulus for torus (defines shape). Default 1j gives standard embedding.
        max_height: Optional maximum height constraint for tau
    
    Returns:
        Reference embedding coordinates (batch_size, 3)
    """
    if domain.lower() == "torus":
        # Transform square domain to parallelogram domain
        uv_transformed = transform_square_to_parallelogram(uv, tau, max_height)
        # Create flat torus embedding
        return get_flat_torus_embedding(uv_transformed, tau)
    
    elif domain.lower() == "sphere":
        u, v = uv[:, 0], uv[:, 1]
        # Sphere parametrization (radius = 1)
        x = torch.sin(v) * torch.cos(u)
        y = torch.sin(v) * torch.sin(u)
        z = torch.cos(v)
        return torch.stack([x, y, z], dim=1)
    
    else:
        raise ValueError(f"Reference embedding not implemented for: {domain}")


def compute_reference_willmore_energy(
    uv: torch.Tensor,
    domain: str = "torus",
    tau: complex = 1j
) -> float:
    """
    Compute the analytical Willmore energy for reference embeddings.
    
    Args:
        uv: Parameter coordinates (batch_size, 2)
        domain: Type of surface
        tau: Complex modulus for torus
    
    Returns:
        Willmore energy (scalar)
    """
    if domain.lower() == "torus":
        # For a flat torus with modulus τ, compute radii
        scale = 3.0
        R = scale * 1.0 / (2 * np.pi)
        r = scale * abs(tau.imag) / (2 * np.pi)
        
        # For a torus: W = 2π² (R² + r²) / (Rr)
        W = 2 * np.pi**2 * (R**2 + r**2) / (R * r)
        return W
    elif domain.lower() == "sphere":
        # For a sphere: W = 4π (independent of radius)
        return 4 * np.pi
    else:
        raise ValueError(f"Reference Willmore energy not known for: {domain}")
