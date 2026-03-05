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


def sample_hyperbolic_octagon(
    num_points:  int,
    device:      torch.device = torch.device("cpu"),
    dtype:       torch.dtype  = torch.float32,
    edge_margin: float        = 0.0,
) -> torch.Tensor:
    """
    Sample points uniformly (by Euclidean area) inside the regular hyperbolic octagon
    in the Poincaré disk. These (x, y) coordinates serve as the parameter domain for
    the genus-2 spectral embedding.

    Rejection sampling within the axis-aligned bounding box [−r, r] × [−r, r],
    where r = tanh(arccosh(cot(π/8)) / 2) ≈ 0.644 is the octagon circumradius.

    Args:
        num_points:  Number of points to sample
        device:      Device to place tensor on
        dtype:       Data type for tensor
        edge_margin: Minimum perpendicular distance from each octagon edge.
                     Excludes a strip near the identified boundary, reducing
                     gradient noise from the C¹ discontinuity of the spectral
                     basis at identified edges.

    Returns:
        xy: (num_points, 2) coordinates in the Poincaré disk octagon
    """
    from spectral import build_octagon_vertices, is_inside_octagon, _octagon_disk_radius
    r     = _octagon_disk_radius()
    verts = build_octagon_vertices()

    batches: list = []
    total = 0
    while total < num_points:
        n_try = max(int((num_points - total) * 3.0) + 200, 500)
        x   = np.random.uniform(-r, r, n_try)
        y   = np.random.uniform(-r, r, n_try)
        pts = np.column_stack([x, y])
        mask  = is_inside_octagon(pts, verts, margin=edge_margin)
        valid = pts[mask]
        if len(valid) > 0:
            batches.append(valid)
            total += len(valid)

    xy_np = np.concatenate(batches, axis=0)[:num_points]
    return torch.tensor(xy_np, dtype=dtype, device=device)


def sample_parameters(
    num_points:  int,
    domain:      str               = "torus",
    device:      torch.device      = torch.device("cpu"),
    dtype:       torch.dtype       = torch.float32,
    genus:       Optional[int]     = None,
    edge_margin: float             = 0.0,
) -> torch.Tensor:
    """
    Sample points in parameter space for the specified domain/topology.
    
    Args:
        num_points: Number of points to sample
        domain: Type of surface ('torus', 'sphere', 'ellipsoid', 'double_torus')
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
        # Get tau_imag from config if available, else default to 1.0
        tau_imag = 1.0
        import inspect
        frame = inspect.currentframe()
        tau = None
        # Try to get tau from the calling context if possible
        if 'tau' in frame.f_back.f_locals:
            tau = frame.f_back.f_locals['tau']
        elif 'tau' in frame.f_back.f_globals:
            tau = frame.f_back.f_globals['tau']
        if tau is not None:
            try:
                tau_imag = abs(complex(tau).imag)
            except Exception:
                tau_imag = 1.0
        return sample_rectangular_domain(num_points, (0, 2*np.pi), (0, 2*np.pi * tau_imag), device, dtype)
    elif domain_lower in ["sphere", "ellipsoid"]:
        return sample_ellipsoid_parameters(num_points, device, dtype)
    elif domain_lower == "double_torus":
        return sample_hyperbolic_octagon(num_points, device, dtype, edge_margin=edge_margin)
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
    
    # Scale to reasonable size for visualization
    scale = 1.5
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


# ============================================================================
# GENUS 2: HYPERBOLIC OCTAGON FUNDAMENTAL DOMAIN
# ============================================================================
#
# The genus-2 surface is realised as the regular hyperbolic octagon in the
# Poincaré disk with opposite-edge identification:
#
#   Word: a₁ b₁ a₁⁻¹ b₁⁻¹ a₂ b₂ a₂⁻¹ b₂⁻¹
#
#   Edge 0 (a₁)  ↔  Edge 2 (a₁⁻¹) reversed
#   Edge 1 (b₁)  ↔  Edge 3 (b₁⁻¹) reversed
#   Edge 4 (a₂)  ↔  Edge 6 (a₂⁻¹) reversed
#   Edge 5 (b₂)  ↔  Edge 7 (b₂⁻¹) reversed
#
# All 8 corner vertices are identified to a single point.
# The resulting surface Σ₂ has constant curvature K = −1.
#
# See spectral.py for the mesh construction, cotangent Laplacian, and
# Laplace-Beltrami eigenfunctions used as network input features.
# Sampling functions are provided by sample_hyperbolic_octagon (above)
# and HyperbolicOctagonSpectral.sample_uniform (in spectral.py).
# ============================================================================

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
        # Genus-2 now uses the hyperbolic octagon spectral parameterisation.
        # There is no analytical reference embedding in (x, y) octagon coordinates;
        # supervised pretraining is disabled for genus 2.
        raise NotImplementedError(
            "No analytical reference embedding for genus-2 hyperbolic octagon "
            "parametrisation. Disable supervised pretraining for genus 2."
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
        # Genus-2 uses the hyperbolic octagon parameterisation; there is no
        # analytical reference embedding.  Return the conjectured minimum as
        # a nominal reference value.
        return 4 * np.pi ** 2  # Lawson surface ξ_{2,1}: W_min ≈ 4π² ≈ 39.48

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
