"""
Parameter Space Sampling for Embedding Learning

This module provides functions to sample points in parameter space (u,v)
for various topologies. The neural network will learn the embedding to R³.

Supported topologies:
- Genus 0 (sphere/ellipsoid): polar coordinates on [0, π] × [0, 2π]
- Genus 1 (torus): doubly-periodic coordinates on [0, 2π] × [0, 2π]  
- Genus 2 (double torus): two-chart sampling (torus with disk exclusion, no bridge)
"""

import torch
import numpy as np
import functools
from typing import Tuple, Optional, Dict
print = functools.partial(print, flush=True)

# Track which tau values have been warned about for self-intersection
_warned_tau_values = set()


def get_domain_for_genus(genus: int) -> str:
    """
    Get the appropriate domain type for a given genus.
    
    Args:
        genus: Surface genus (0, 1, or 2)
    
    Returns:
        Domain string identifier
    """
    if genus == 0:
        return "ellipsoid"
    elif genus == 1:
        return "torus"
    elif genus == 2:
        return "double_torus"
    else:
        raise NotImplementedError(f"Genus {genus} is not supported. Only genus 0, 1, 2 are implemented.")


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
    Sample points uniformly in parameter space for a sphere/ellipsoid.

    For a sphere: u ∈ [0, 2π] (azimuthal), v ∈ [0, π] (polar)
    Both coordinates are sampled uniformly in parameter space. The surface
    area element √(EG-F²) is part of the integrand in the Monte Carlo
    estimator and naturally accounts for the metric distortion, so no
    importance correction is needed or wanted.

    Args:
        num_points: Number of points to sample
        device: Device to place tensor on
        dtype: Data type for tensor

    Returns:
        Parameter coordinates of shape (num_points, 2)
        u ∈ [0, 2π], v ∈ [0, π]
    """
    u = torch.rand(num_points, device=device, dtype=dtype) * 2 * np.pi
    v = torch.rand(num_points, device=device, dtype=dtype) * np.pi

    return torch.stack([u, v], dim=1)


def sample_ellipsoid_parameters(
    num_points: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Sample points for an ellipsoid parametrization.
    Same as sphere sampling since we use the same polar parametrization.
    
    Args:
        num_points: Number of points to sample
        device: Device to place tensor on
        dtype: Data type for tensor
    
    Returns:
        Parameter coordinates of shape (num_points, 2)
        u ∈ [0, 2π] (azimuthal), v ∈ [0, π] (polar)
    """
    return sample_sphere_parameters(num_points, device, dtype)



def sample_parameters(
    num_points: int,
    domain: str = "torus",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    genus: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample points in parameter space for the specified domain/topology.

    For genus 2 (multi-chart), use sample_torus_excluding_disk directly.
    
    Args:
        num_points: Number of points to sample
        domain: Type of surface ('torus', 'sphere', 'ellipsoid')
        device: Device to place tensor on
        dtype: Data type for tensor
        genus: If provided, overrides domain selection (0=ellipsoid, 1=torus)
    
    Returns:
        Parameter coordinates of shape (num_points, 2)
    """
    # If genus is provided, derive domain from it
    if genus is not None:
        domain = get_domain_for_genus(genus)
    
    domain_lower = domain.lower()
    
    if domain_lower == "torus":
        return sample_rectangular_domain(num_points, (0, 2*np.pi), (0, 2*np.pi), device, dtype)
    elif domain_lower in ["sphere", "ellipsoid"]:
        return sample_ellipsoid_parameters(num_points, device, dtype)
    elif domain_lower == "double_torus":
        raise ValueError(
            "Genus 2 uses multi-chart sampling. Call sample_torus_excluding_disk / "
            "sample_bridge_domain directly."
        )
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
    - The v-direction (along [0, 2π*τ]) wraps around the minor circle
    - Re(τ) creates a helical twist: the minor circle rotates as we go around the major circle
    - Im(τ) controls the minor/major radius ratio
    
    Args:
        uv: Parameter coordinates (batch_size, 2) on the parallelogram domain
        tau: Complex modulus (default: 1j gives standard circular cross-section)
             Re(τ) controls helical twist, Im(τ) controls minor/major ratio
    
    Returns:
        Embedding coordinates (batch_size, 3) in R³
    """
    u, v = uv[:, 0], uv[:, 1]
    
    # Extract real and imaginary parts of tau
    tau_real = tau.real
    tau_imag = abs(tau.imag)
    
    # Set major/minor radii so that r/R = Im(tau) (standard aspect ratio for rectangular domain)
    # R = 1.0 (major radius fixed), r = Im(tau) (minor radius)
    R = 1.0
    r = abs(tau.imag)
    
    # Warn if self-intersection will occur (only once per tau value)
    if r >= 1.0:
        tau_key = (round(tau.real, 6), round(tau.imag, 6))
        if tau_key not in _warned_tau_values:
            _warned_tau_values.add(tau_key)
            print(f"Warning: Im(τ) = {r:.2f} >= 1.0 will cause torus self-intersection (r >= R)")
    
    # Wrap u to [0, 2π] since it parametrizes the major circle (period 2π)
    # When Re(τ) ≠ 0, the parallelogram transformation can produce u outside [0, 2π]
    u = u % (2 * np.pi)
    
    # Normalize v to [0, 2π] for the minor circle parametrization
    # v comes in as the parallelogram v-coordinate, which has range [0, 2π*Im(τ)]
    v_normalized = v / tau_imag if tau_imag > 1e-10 else v
    
    # Add helical twist from Re(τ): as we go around in u, rotate the v angle
    # The twist rate is Re(τ)/Im(τ) rotations per u-period
    twist_rate = tau_real / tau_imag if tau_imag > 1e-10 else 0.0
    v_twisted = v_normalized + twist_rate * u
    
    # Twisted torus embedding (symmetric about z=0)
    # The twist causes the minor circle to rotate around as we traverse the major circle
    x = (R + r * torch.cos(v_twisted)) * torch.cos(u)
    y = (R + r * torch.cos(v_twisted)) * torch.sin(u)
    z = r * torch.sin(v_twisted)
    
    return torch.stack([x, y, z], dim=1)


def get_ellipsoid_embedding(
    uv: torch.Tensor,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0
) -> torch.Tensor:
    """
    Create an ellipsoid embedding in R³.
    
    Parametrization using spherical-like coordinates:
    x = a * sin(v) * cos(u)
    y = b * sin(v) * sin(u)
    z = c * cos(v)
    
    where u ∈ [0, 2π] (azimuthal) and v ∈ [0, π] (polar).
    For a = b = c = r, this gives a sphere of radius r.
    
    Args:
        uv: Parameter coordinates (batch_size, 2)
            u ∈ [0, 2π], v ∈ [0, π]
        a: Semi-axis along x
        b: Semi-axis along y
        c: Semi-axis along z
    
    Returns:
        Embedding coordinates (batch_size, 3) in R³
    """
    u, v = uv[:, 0], uv[:, 1]
    
    x = a * torch.sin(v) * torch.cos(u)
    y = b * torch.sin(v) * torch.sin(u)
    z = c * torch.cos(v)
    
    return torch.stack([x, y, z], dim=1)


def get_reference_embedding(
    uv: torch.Tensor,
    domain: str = "torus",
    tau: complex = 1j,
    max_height: Optional[float] = None,
    genus: Optional[int] = None,
    topology_params: Optional[Dict] = None
) -> torch.Tensor:
    """
    Get reference embedding for comparison (analytical parametrization).
    
    For genus 2, reference embeddings are handled per-chart by Genus2MultiChartNetwork.
    
    Args:
        uv: Parameter coordinates (batch_size, 2)
        domain: Type of surface ('torus', 'sphere', 'ellipsoid')
        tau: Complex modulus for torus (defines shape). Default 1j gives standard embedding.
        max_height: Optional maximum height constraint for tau
        genus: If provided, overrides domain selection (0=ellipsoid, 1=torus)
        topology_params: Dictionary of topology-specific parameters from config
    
    Returns:
        Reference embedding coordinates (batch_size, 3)
    """
    # If genus is provided, derive domain from it
    if genus is not None:
        domain = get_domain_for_genus(genus)
    
    topology_params = topology_params or {}
    domain_lower = domain.lower()
    
    if domain_lower == "torus":
        # Transform square domain to parallelogram domain
        uv_transformed = transform_square_to_parallelogram(uv, tau, max_height)
        # Create flat torus embedding
        return get_flat_torus_embedding(uv_transformed, tau)
    
    elif domain_lower in ["sphere", "ellipsoid"]:
        # Get ellipsoid parameters
        ellipsoid_params = topology_params.get('ellipsoid', {})
        a = ellipsoid_params.get('a', 1.0)
        b = ellipsoid_params.get('b', 1.0)
        c = ellipsoid_params.get('c', 1.0)
        return get_ellipsoid_embedding(uv, a, b, c)
    
    elif domain_lower == "double_torus":
        raise ValueError(
            "Genus 2 uses per-chart reference embeddings via Genus2MultiChartNetwork."
        )
    
    else:
        raise ValueError(f"Reference embedding not implemented for: {domain}")


def compute_reference_willmore_energy(
    uv: torch.Tensor,
    domain: str = "torus",
    tau: complex = 1j,
    genus: Optional[int] = None,
    topology_params: Optional[Dict] = None
) -> float:
    """
    Compute the analytical Willmore energy for reference embeddings.
    
    Args:
        uv: Parameter coordinates (batch_size, 2)
        domain: Type of surface
        tau: Complex modulus for torus
        genus: If provided, overrides domain selection
        topology_params: Dictionary of topology-specific parameters
    
    Returns:
        Willmore energy (scalar)
    """
    # If genus is provided, derive domain from it
    if genus is not None:
        domain = get_domain_for_genus(genus)
    
    domain_lower = domain.lower()
    topology_params = topology_params or {}
    
    if domain_lower == "torus":
        # For a flat torus with modulus τ, compute radii
        scale = 3.0
        R = scale * 1.0 / (2 * np.pi)
        r = scale * abs(tau.imag) / (2 * np.pi)
        
        # For a torus: W = 2π² (R² + r²) / (Rr)
        # Note: This assumes a standard round torus, not Clifford
        W = 2 * np.pi**2 * (R**2 + r**2) / (R * r)
        return W
    
    elif domain_lower in ["sphere", "ellipsoid"]:
        # For a sphere: W = 4π (independent of radius)
        # This is the theoretical minimum for genus 0
        # For an ellipsoid, the Willmore energy is generally higher
        ellipsoid_params = topology_params.get('ellipsoid', {})
        a = ellipsoid_params.get('a', 1.0)
        b = ellipsoid_params.get('b', 1.0)
        c = ellipsoid_params.get('c', 1.0)
        
        if abs(a - b) < 1e-6 and abs(b - c) < 1e-6:
            # Sphere case: W = 4π
            return 4 * np.pi
        else:
            # Ellipsoid case: approximate Willmore energy
            # For a general ellipsoid, W > 4π
            # Use numerical approximation or return an estimate
            # The Willmore energy increases with eccentricity
            # Simple estimate based on semi-axes ratios
            mean_radius = (a + b + c) / 3
            variance = ((a - mean_radius)**2 + (b - mean_radius)**2 + (c - mean_radius)**2) / 3
            eccentricity_factor = 1.0 + variance / (mean_radius**2)
            return 4 * np.pi * eccentricity_factor
    
    elif domain_lower == "double_torus":
        # Genus 2: reference energy is just the theoretical minimum (no analytical
        # reference surface in the multi-chart architecture).
        return 4 * np.pi**2  # Lawson ξ_{2,1} ≈ 39.48
    
    else:
        raise ValueError(f"Reference Willmore energy not known for: {domain}")


def get_theoretical_minimum_willmore(genus: int) -> float:
    """
    Get the theoretical minimum Willmore energy for a given genus.
    
    These are the conjectured/proven minimizers:
    - Genus 0: Round sphere, W = 4π ≈ 12.566
    - Genus 1: Clifford torus, W = 2π² ≈ 19.739
    - Genus 2: Lawson surface ξ_{2,1}, W ≈ 4π² ≈ 39.478
    
    Args:
        genus: Surface genus
    
    Returns:
        Theoretical minimum Willmore energy
    """
    if genus == 0:
        return 4 * np.pi  # Round sphere
    elif genus == 1:
        return 2 * np.pi**2  # Clifford torus
    elif genus == 2:
        return 4 * np.pi**2  # Lawson minimal surface (approximate)
    else:
        raise NotImplementedError(f"Theoretical minimum not known for genus {genus}")


# ============================================================================
# GENUS 2 MULTI-CHART SAMPLING
# ============================================================================


def sample_torus_excluding_disk(
    num_points: int,
    disk_center: Tuple[float, float] = (0.0, 0.0),
    disk_radius: float = 0.3,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample uniformly on [0, 2π]² excluding a disk of parameter-space radius δ.

    Uses rejection sampling: points within δ of the disk centre (with periodic
    wrap-around) are discarded.

    Args:
        num_points: Target number of accepted points.
        disk_center: (u₀, v₀) centre of the excluded disk.
        disk_radius: δ, parameter-space radius to exclude.
        device, dtype: Torch device and data type.

    Returns:
        (num_points, 2) tensor of (u, v) ∈ [0, 2π]².
    """
    u0, v0 = disk_center
    collected = []
    # Acceptance ratio ≈ 1 − πδ² / (2π)² = 1 − δ² / (4π)
    oversample = max(2, int(1.2 / max(1e-6, 1.0 - disk_radius ** 2 / (4 * np.pi))))
    while sum(c.shape[0] for c in collected) < num_points:
        n = num_points * oversample
        u = torch.rand(n, device=device, dtype=dtype) * 2 * np.pi
        v = torch.rand(n, device=device, dtype=dtype) * 2 * np.pi
        # Periodic distance to disk centre
        du = torch.abs(u - u0)
        du = torch.min(du, 2 * np.pi - du)
        dv = torch.abs(v - v0)
        dv = torch.min(dv, 2 * np.pi - dv)
        dist2 = du * du + dv * dv
        mask = dist2 > disk_radius * disk_radius
        collected.append(torch.stack([u[mask], v[mask]], dim=1))
    return torch.cat(collected, dim=0)[:num_points]


def sample_bridge_domain(
    num_points: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample uniformly on the bridge chart domain [0, 2π] × [0, 1].

    Args:
        num_points: Number of points.
        device, dtype: Torch device and data type.

    Returns:
        (num_points, 2) tensor with col 0 = u ∈ [0, 2π], col 1 = t ∈ [0, 1].
    """
    u = torch.rand(num_points, device=device, dtype=dtype) * 2 * np.pi
    t = torch.rand(num_points, device=device, dtype=dtype)
    return torch.stack([u, t], dim=1)


def sample_disk_boundary(
    num_points: int,
    disk_center: Tuple[float, float] = (0.0, 0.0),
    disk_radius: float = 0.3,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points on the boundary ∂D of a parametric disk on [0, 2π]².

    Returns both the angle s ∈ [0, 2π] parametrising the boundary and the
    corresponding torus-chart coordinates (u, v).

    Args:
        num_points: Number of boundary samples.
        disk_center: (u₀, v₀).
        disk_radius: δ.
        device, dtype: Torch device and data type.

    Returns:
        s:  (num_points,) angles in [0, 2π]
        uv: (num_points, 2) boundary coordinates on the torus chart (periodic wrap)
    """
    s = torch.linspace(0, 2 * np.pi, num_points + 1, device=device, dtype=dtype)[:-1]
    u0, v0 = disk_center
    u = (u0 + disk_radius * torch.cos(s)) % (2 * np.pi)
    v = (v0 + disk_radius * torch.sin(s)) % (2 * np.pi)
    uv = torch.stack([u, v], dim=1)
    return s, uv
