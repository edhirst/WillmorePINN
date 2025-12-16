"""
Parameter Space Sampling for Embedding Learning

This module provides functions to sample points in parameter space (u,v)
for various topologies. The neural network will learn the embedding to R³.

Supported topologies:
- Genus 0 (sphere/ellipsoid): polar coordinates on [0, π] × [0, 2π]
- Genus 1 (torus): doubly-periodic coordinates on [0, 2π] × [0, 2π]  
- Genus 2 (double torus): custom parametrization with catenoid bridge
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict


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
    Sample points in parameter space for a sphere/ellipsoid with area-weighting.
    
    For a sphere: u ∈ [0, 2π] (azimuthal), v ∈ [0, π] (polar)
    Uses area-weighted sampling: cos(v) uniform in [-1, 1] ensures uniform
    point distribution on the sphere surface (avoids pole clustering).
    
    Args:
        num_points: Number of points to sample
        device: Device to place tensor on
        dtype: Data type for tensor
    
    Returns:
        Parameter coordinates of shape (num_points, 2)
        u ∈ [0, 2π], v ∈ [0, π]
    """
    u = torch.rand(num_points, device=device, dtype=dtype) * 2 * np.pi
    cos_v = torch.rand(num_points, device=device, dtype=dtype) * 2 - 1
    v = torch.acos(cos_v)
    
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


def sample_double_torus_parameters(
    num_points: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Sample points in parameter space for a genus-2 double torus.
    
    The double torus is constructed from two tori connected by a catenoid bridge.
    We use a composite parametrization:
    - First coordinate u ∈ [0, 2π] wraps around the combined tube
    - Second coordinate v encodes position along the double torus structure
    
    The domain is [0, 2π] × [0, 4π] where:
    - v ∈ [0, 2π]: first torus
    - v ∈ [2π, 4π]: second torus (with bridge connection implicit)
    
    Args:
        num_points: Number of points to sample
        device: Device to place tensor on
        dtype: Data type for tensor
    
    Returns:
        Parameter coordinates of shape (num_points, 2)
    """
    u = torch.rand(num_points, device=device, dtype=dtype) * 2 * np.pi
    v = torch.rand(num_points, device=device, dtype=dtype) * 4 * np.pi
    
    return torch.stack([u, v], dim=1)


def sample_parameters(
    num_points: int,
    domain: str = "torus",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    genus: Optional[int] = None
) -> torch.Tensor:
    """
    Sample points in parameter space for the specified domain/topology.
    
    Args:
        num_points: Number of points to sample
        domain: Type of surface ('torus', 'sphere', 'ellipsoid', 'double_torus', 'klein_bottle')
        device: Device to place tensor on
        dtype: Data type for tensor
        genus: If provided, overrides domain selection (0=ellipsoid, 1=torus, 2=double_torus)
    
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
        return sample_double_torus_parameters(num_points, device, dtype)
    elif domain_lower == "klein_bottle":
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
    - The v-direction (along [0, 2π*τ]) wraps around the minor circle
    - Re(τ) creates a helical twist: the minor circle rotates as we go around the major circle
    - Im(τ) controls the minor radius
    
    Args:
        uv: Parameter coordinates (batch_size, 2) on the parallelogram domain
        tau: Complex modulus (default: 1j gives standard circular cross-section)
             Re(τ) controls helical twist, Im(τ) controls minor radius
    
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
    
    # Add helical twist from Re(τ): as we go around in u, rotate the v angle
    # This respects the parallelogram identification geometrically
    twist_angle = tau_real * u if abs(tau_imag) > 1e-10 else 0.0
    v_twisted = v_normalized + twist_angle
    
    # Scale to reasonable size
    scale = 3.0  # Scaling factor for visualization
    R_scaled = scale * R
    r_scaled = scale * r
    
    # Twisted torus embedding (symmetric about z=0)
    # The twist causes the minor circle to rotate around as we traverse the major circle
    x = (R_scaled + r_scaled * torch.cos(v_twisted)) * torch.cos(u)
    y = (R_scaled + r_scaled * torch.cos(v_twisted)) * torch.sin(u)
    z = r_scaled * torch.sin(v_twisted)
    
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


def get_double_torus_embedding(
    uv: torch.Tensor,
    torus_separation: float = 3.0,
    major_radius: float = 1.5,
    minor_radius: float = 0.6,
    bridge_radius: float = 0.4,
    bridge_length: float = 1.0
) -> torch.Tensor:
    """
    Create a genus-2 double torus embedding using catenoid bridge construction.
    
    The double torus is constructed from two tori positioned along the x-axis
    and conceptually connected via a catenoid-like bridge. The parametrization
    smoothly transitions between the two tori.
    
    Domain: u ∈ [0, 2π], v ∈ [0, 4π]
    - v ∈ [0, 2π]: wraps around first torus
    - v ∈ [2π, 4π]: wraps around second torus
    
    Args:
        uv: Parameter coordinates (batch_size, 2)
        torus_separation: Distance between torus centers along x-axis
        major_radius: R - distance from center of tube to center of each torus
        minor_radius: r - radius of the tube
        bridge_radius: Radius of the connecting bridge region
        bridge_length: Length parameter for bridge smoothing
    
    Returns:
        Embedding coordinates (batch_size, 3) in R³
    """
    u, v = uv[:, 0], uv[:, 1]
    
    # Normalize v to [0, 2] for determining which torus and position
    v_normalized = v / (2 * np.pi)  # Now in [0, 2]
    
    # Determine which torus (0 = first, 1 = second) using smooth transition
    torus_index = torch.floor(v_normalized).clamp(0, 1)
    v_local = (v_normalized - torus_index) * 2 * np.pi  # Local v in [0, 2π]
    
    # X offset for each torus
    x_offset = (torus_index - 0.5) * torus_separation
    
    # Smooth blending factor for bridge region
    # Bridge connects at v_local ≈ 0 and v_local ≈ 2π (the touching points)
    bridge_blend = torch.exp(-((v_local - np.pi) ** 2) / (bridge_length ** 2))
    
    # Effective radii with bridge modification
    # At the bridge points, squeeze the tori together
    R_effective = major_radius
    r_effective = minor_radius * (1 - 0.3 * bridge_blend)  # Slightly thinner at bridge
    
    # Standard torus parametrization centered at x_offset
    x = (R_effective + r_effective * torch.cos(v_local)) * torch.cos(u) + x_offset
    y = (R_effective + r_effective * torch.cos(v_local)) * torch.sin(u)
    z = r_effective * torch.sin(v_local)
    
    # Add bridge connection: at transition points, blend the two tori
    # This creates a smooth genus-2 surface
    transition_width = 0.3
    transition_at_0 = torch.exp(-(v_local ** 2) / (transition_width ** 2))
    transition_at_2pi = torch.exp(-((v_local - 2 * np.pi) ** 2) / (transition_width ** 2))
    transition = transition_at_0 + transition_at_2pi
    
    # At transition regions, move points inward to create the bridge effect
    bridge_factor = 1.0 - 0.5 * transition * bridge_blend
    
    # Apply bridge modification
    x = x * bridge_factor + x_offset * (1 - bridge_factor)
    
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
    
    Args:
        uv: Parameter coordinates (batch_size, 2)
        domain: Type of surface ('torus', 'sphere', 'ellipsoid', 'double_torus')
        tau: Complex modulus for torus (defines shape). Default 1j gives standard embedding.
        max_height: Optional maximum height constraint for tau
        genus: If provided, overrides domain selection (0=ellipsoid, 1=torus, 2=double_torus)
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
        # Get double torus parameters
        dt_params = topology_params.get('double_torus', {})
        return get_double_torus_embedding(
            uv,
            torus_separation=dt_params.get('torus_separation', 3.0),
            major_radius=dt_params.get('torus_major_radius', 1.5),
            minor_radius=dt_params.get('torus_minor_radius', 0.6),
            bridge_radius=dt_params.get('bridge_radius', 0.4),
            bridge_length=dt_params.get('bridge_length', 1.0)
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
        # For a genus-2 surface, the theoretical minimum Willmore energy
        # is achieved by Lawson's minimal surface ξ_{2,1}
        # W_min ≈ 2π² × 2 = 4π² ≈ 39.48 for the optimal genus-2 surface
        # For our initial double torus configuration, energy will be higher
        dt_params = topology_params.get('double_torus', {})
        R = dt_params.get('torus_major_radius', 1.5)
        r = dt_params.get('torus_minor_radius', 0.6)
        
        # Approximate as sum of two tori (overestimate)
        single_torus_W = 2 * np.pi**2 * (R**2 + r**2) / (R * r)
        return 2 * single_torus_W * 0.8  # Factor to account for bridge connection
    
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
