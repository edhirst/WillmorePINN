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


def get_reference_embedding(
    uv: torch.Tensor,
    domain: str = "torus",
    major_radius: float = 2.0,
    minor_radius: float = 1.0
) -> torch.Tensor:
    """
    Get reference embedding for comparison (analytical parametrization).
    
    Args:
        uv: Parameter coordinates (batch_size, 2)
        domain: Type of surface
        major_radius: Major radius for torus
        minor_radius: Minor radius for torus
    
    Returns:
        Reference embedding coordinates (batch_size, 3)
    """
    u, v = uv[:, 0], uv[:, 1]
    
    if domain.lower() == "torus":
        # Standard torus parametrization
        x = (major_radius + minor_radius * torch.cos(v)) * torch.cos(u)
        y = (major_radius + minor_radius * torch.cos(v)) * torch.sin(u)
        z = minor_radius * torch.sin(v)
    elif domain.lower() == "sphere":
        # Sphere parametrization (radius = 1)
        x = torch.sin(v) * torch.cos(u)
        y = torch.sin(v) * torch.sin(u)
        z = torch.cos(v)
    else:
        raise ValueError(f"Reference embedding not implemented for: {domain}")
    
    return torch.stack([x, y, z], dim=1)


def compute_reference_willmore_energy(
    uv: torch.Tensor,
    domain: str = "torus",
    major_radius: float = 2.0,
    minor_radius: float = 1.0
) -> float:
    """
    Compute the analytical Willmore energy for reference embeddings.
    
    Args:
        uv: Parameter coordinates (batch_size, 2)
        domain: Type of surface
        major_radius: Major radius for torus
        minor_radius: Minor radius for torus
    
    Returns:
        Willmore energy (scalar)
    """
    if domain.lower() == "torus":
        # For a torus: W = 2π² (R² + r²) / (Rr)
        W = 2 * np.pi**2 * (major_radius**2 + minor_radius**2) / (major_radius * minor_radius)
        return W
    elif domain.lower() == "sphere":
        # For a sphere: W = 4π (independent of radius)
        return 4 * np.pi
    else:
        raise ValueError(f"Reference Willmore energy not known for: {domain}")
