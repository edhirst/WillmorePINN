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


def sample_double_torus_parameters(
    num_points: int,
    bridge_oversample_factor: float = 1.0,
    junction_focus: float = 0.0,
    junction_sigma: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Sample points in parameter space for a genus-2 double torus.
    
    The double torus is constructed from two tori connected by a catenoid bridge.
    We use a composite parametrization:
    - First coordinate u ∈ [0, 2π] wraps around the tube/catenoid cross-section
    - Second coordinate v encodes position along the double torus structure
    
    The domain is [0, 2π] × [0, 5π] where:
    - v ∈ [0, 2π): first torus (with cut at φ=0 for bridge)
    - v ∈ [2π, 3π): catenoid bridge
    - v ∈ [3π, 5π): second torus (with cut at φ=π for bridge)
    
    Args:
        num_points: Number of points to sample
        bridge_oversample_factor: Ratio of bridge samples relative to its uniform share.
            The bridge occupies π out of 5π (1/5) of the v domain.  With factor f,
            the bridge receives f/(f+4) of total points instead of 1/5.  This
            concentrates gradient signal on the catenoid region without reweighting,
            intentionally biasing the optimizer to improve bridge geometry.
            Set to 1.0 for uniform sampling.
        junction_focus: Fraction of points drawn from Gaussians centred on the two
            junctions (v=2π and v=3π).  0 = pure uniform/bridge-oversample;
            1 = all points at junctions.  Annealed from high → 0 during training
            to prioritise smoothing high-curvature regions first.
        junction_sigma: Standard deviation (radians) of each junction Gaussian.
            Controls how tightly samples cluster around v=2π and v=3π.
        device: Device to place tensor on
        dtype: Data type for tensor
    
    Returns:
        Parameter coordinates of shape (num_points, 2)
    """
    v_max = 5 * np.pi

    # --- Junction-focused component -----------------------------------------
    # When junction_focus > 0, a fraction of points are drawn from Gaussians
    # centred on the T1/catenoid junction (v=2π) and the catenoid/T2 junction
    # (v=3π).  u is always uniform in [0, 2π].
    n_junction = max(0, min(num_points - 1, int(round(junction_focus * num_points))))
    n_uniform = num_points - n_junction

    parts_u, parts_v = [], []

    if n_junction > 0:
        n_j1 = n_junction // 2          # points near v = 2π
        n_j2 = n_junction - n_j1        # points near v = 3π
        v_j1 = torch.clamp(
            torch.randn(n_j1, device=device, dtype=dtype) * junction_sigma + 2 * np.pi,
            0.0, v_max)
        v_j2 = torch.clamp(
            torch.randn(n_j2, device=device, dtype=dtype) * junction_sigma + 3 * np.pi,
            0.0, v_max)
        u_j = torch.rand(n_junction, device=device, dtype=dtype) * 2 * np.pi
        parts_u.append(u_j)
        parts_v.append(torch.cat([v_j1, v_j2]))

    # --- Uniform / bridge-oversampled component -----------------------------
    if n_uniform > 0:
        if bridge_oversample_factor <= 1.0:
            u_uni = torch.rand(n_uniform, device=device, dtype=dtype) * 2 * np.pi
            v_uni = torch.rand(n_uniform, device=device, dtype=dtype) * v_max
            parts_u.append(u_uni)
            parts_v.append(v_uni)
        else:
            f = bridge_oversample_factor
            n_bridge = int(round(n_uniform * f / (f + 4)))
            n_other = n_uniform - n_bridge

            parts_u.append(torch.rand(n_bridge, device=device, dtype=dtype) * 2 * np.pi)
            parts_v.append(torch.rand(n_bridge, device=device, dtype=dtype) * np.pi + 2 * np.pi)

            u_other = torch.rand(n_other, device=device, dtype=dtype) * 2 * np.pi
            v_raw = torch.rand(n_other, device=device, dtype=dtype) * 4 * np.pi
            v_other = torch.where(v_raw < 2 * np.pi, v_raw, v_raw + np.pi)
            parts_u.append(u_other)
            parts_v.append(v_other)

    u = torch.cat(parts_u) if parts_u else torch.empty(0, device=device, dtype=dtype)
    v = torch.cat(parts_v) if parts_v else torch.empty(0, device=device, dtype=dtype)
    return torch.stack([u, v], dim=1)


def sample_parameters(
    num_points: int,
    domain: str = "torus",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    genus: Optional[int] = None,
    bridge_oversample_factor: float = 1.0,
    junction_focus: float = 0.0,
    junction_sigma: float = 1.0
) -> torch.Tensor:
    """
    Sample points in parameter space for the specified domain/topology.
    
    Args:
        num_points: Number of points to sample
        domain: Type of surface ('torus', 'sphere', 'ellipsoid', 'double_torus')
        device: Device to place tensor on
        dtype: Data type for tensor
        genus: If provided, overrides domain selection (0=ellipsoid, 1=torus, 2=double_torus)
        bridge_oversample_factor: For double_torus only — oversample catenoid bridge region.
            See sample_double_torus_parameters for details.
        junction_focus: For double_torus only — fraction of points near junctions.
            See sample_double_torus_parameters for details.
        junction_sigma: For double_torus only — Gaussian std dev around junctions (radians).
    
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
        return sample_double_torus_parameters(
            num_points,
            bridge_oversample_factor=bridge_oversample_factor,
            junction_focus=junction_focus,
            junction_sigma=junction_sigma,
            device=device,
            dtype=dtype
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


# ============================================================================
# GENUS 2: TWO TORI + CATENOID BRIDGE
# ============================================================================
#
# A genus-2 surface constructed as two separate tori connected by a catenoid
# (capillary bridge).
#
# GEOMETRY:
# - Two tori with major radius axes parallel to z-axis (standard orientation)
# - Centers at y=0, z=0, separated along x-axis
# - Catenoid bridge parallel to x-axis, centered at y=0, z=0
#
# THREE FUNDAMENTAL DOMAINS:
# 1. Torus 1 parallelogram (with gluing disk excluded)
# 2. Catenoid rectangle
# 3. Torus 2 parallelogram (with gluing disk excluded)
#
# ============================================================================


def sample_genus2_surface(
    num_points: int,
    tau1: complex = 1j,
    tau2: complex = 1j,
    R1: float = 1.0,
    R2: float = 1.0,
    r1: float = 0.35,
    r2: float = 0.35,
    neck_length: float = 1.0,
    glue_angle: float = 0.5,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample points from a genus-2 surface (two tori + catenoid bridge).
    
    Returns points from all three regions with appropriate exclusions.
    
    Args:
        num_points: Total number of points to sample
        tau1, tau2: Complex modular parameters for each torus
        R1, R2: Major radii of each torus
        r1, r2: Minor radii of each torus
        neck_length: Length of catenoid bridge
        glue_angle: Angular width of gluing region (in radians)
        device, dtype: Torch device and dtype
    
    Returns:
        Tuple of (uv_coords, xyz_coords, region_labels)
        region_labels: 0 = Torus 1, 1 = Catenoid, 2 = Torus 2
    """
    # Distribute points across regions based on approximate surface areas
    # Torus area ≈ 4π²Rr, Catenoid area depends on length and radius
    area_T1 = 4 * np.pi**2 * R1 * r1 * (1 - glue_angle / (2 * np.pi))
    area_T2 = 4 * np.pi**2 * R2 * r2 * (1 - glue_angle / (2 * np.pi))
    catenoid_radius = min(r1, r2)
    area_cat = 2 * np.pi * catenoid_radius * neck_length  # Approximate
    
    total_area = area_T1 + area_T2 + area_cat
    n_T1 = int(num_points * area_T1 / total_area)
    n_cat = int(num_points * area_cat / total_area)
    n_T2 = num_points - n_T1 - n_cat
    
    # Sample Torus 1 (excluding gluing region around u=0)
    uv_T1 = sample_torus_with_exclusion(n_T1, tau1, glue_angle, exclude_at_u=0.0, 
                                         device=device, dtype=dtype)
    
    # Sample Catenoid
    uv_cat = sample_catenoid_domain(n_cat, device=device, dtype=dtype)
    
    # Sample Torus 2 (excluding gluing region around u=π)
    uv_T2 = sample_torus_with_exclusion(n_T2, tau2, glue_angle, exclude_at_u=np.pi,
                                         device=device, dtype=dtype)
    
    return uv_T1, uv_cat, uv_T2


def sample_torus_with_exclusion(
    num_points: int,
    tau: complex,
    glue_angle: float,
    exclude_at_u: float,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Sample points on a torus, excluding a disk around the gluing region.
    
    The torus is parametrized by (u, v) ∈ [0, 2π] × [0, 2π].
    We exclude points where |u - exclude_at_u| < glue_angle/2.
    
    Args:
        num_points: Number of points to sample
        tau: Complex modular parameter
        glue_angle: Angular width to exclude
        exclude_at_u: Center of excluded region (0 or π typically)
        device, dtype: Torch device and dtype
    
    Returns:
        uv coordinates (num_points, 2)
    """
    # Sample with rejection
    points_collected = []
    batch_size = num_points * 2  # Oversample to handle rejections
    
    while len(points_collected) < num_points:
        # Sample uniformly on [0, 2π] × [0, 2π]
        u = torch.rand(batch_size, device=device, dtype=dtype) * 2 * np.pi
        v = torch.rand(batch_size, device=device, dtype=dtype) * 2 * np.pi
        
        # Compute distance from exclusion center (handling wrap-around)
        du = torch.abs(u - exclude_at_u)
        du = torch.min(du, 2 * np.pi - du)  # Wrap-around distance
        
        # Keep points outside exclusion region
        mask = du > glue_angle / 2
        
        valid_u = u[mask]
        valid_v = v[mask]
        
        if len(valid_u) > 0:
            points_collected.append(torch.stack([valid_u, valid_v], dim=1))
    
    # Concatenate and trim to exact count
    all_points = torch.cat(points_collected, dim=0)[:num_points]
    return all_points


def sample_catenoid_domain(
    num_points: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Sample points on the catenoid domain.
    
    Catenoid parametrization: (θ, s) where
    - θ ∈ [0, 2π]: angle around the catenoid
    - s ∈ [-1, 1]: normalized position along the catenoid axis
    
    Args:
        num_points: Number of points to sample
        device, dtype: Torch device and dtype
    
    Returns:
        (θ, s) coordinates (num_points, 2)
    """
    theta = torch.rand(num_points, device=device, dtype=dtype) * 2 * np.pi
    s = torch.rand(num_points, device=device, dtype=dtype) * 2 - 1  # [-1, 1]
    
    return torch.stack([theta, s], dim=1)


def embed_torus(
    uv: torch.Tensor,
    R: float,
    r: float,
    tau: complex,
    center_x: float = 0.0
) -> torch.Tensor:
    """
    Embed a torus in R³ with major radius axis along z-axis.
    
    Standard torus parametrization:
        x = (R + r·cos(v)) · cos(u)
        y = (R + r·cos(v)) · sin(u)
        z = r · sin(v)
    
    The hole (central empty region) is visible from above/below (along z-axis).
    
    Args:
        uv: Parameter coordinates (N, 2), u and v in [0, 2π]
        R: Major radius (center of torus to center of tube)
        r: Minor radius (tube radius)
        tau: Complex modular parameter (Re = shear, Im = aspect scaling)
        center_x: x-coordinate of torus center
    
    Returns:
        xyz coordinates (N, 3)
    """
    u, v = uv[:, 0], uv[:, 1]
    
    # Apply shear from Re(τ)
    tau_re = float(tau.real) if hasattr(tau, 'real') else 0.0
    u_sheared = u + tau_re * v / (2 * np.pi)
    
    cos_u = torch.cos(u_sheared)
    sin_u = torch.sin(u_sheared)
    cos_v = torch.cos(v)
    sin_v = torch.sin(v)
    
    rho = R + r * cos_v
    x = rho * cos_u + center_x
    y = rho * sin_u
    z = r * sin_v
    
    return torch.stack([x, y, z], dim=1)


def embed_catenoid(
    params: torch.Tensor,
    neck_length: float,
    waist_radius: float,
    end_radius: float
) -> torch.Tensor:
    """
    Embed a catenoid (capillary bridge) in R³ along the x-axis.
    
    Catenoid parametrization:
        x = s · (neck_length/2)  (s ∈ [-1, 1])
        y = ρ(s) · cos(θ)
        z = ρ(s) · sin(θ)
    
    where ρ(s) interpolates from end_radius at |s|=1 to waist_radius at s=0.
    
    For a true catenoid: ρ(s) = a · cosh(s·L/(2a)) where a is the waist radius.
    
    Args:
        params: (θ, s) coordinates (N, 2), θ ∈ [0, 2π], s ∈ [-1, 1]
        neck_length: Total length of the catenoid
        waist_radius: Radius at the narrowest point (center)
        end_radius: Radius at the ends (where it meets the tori)
    
    Returns:
        xyz coordinates (N, 3)
    """
    theta, s = params[:, 0], params[:, 1]
    
    # Catenoid profile: ρ(s) = a · cosh(s · L / (2a))
    # We want ρ(0) = waist_radius (a = waist_radius)
    # And ρ(±1) = end_radius = a · cosh(L/(2a))
    # 
    # Given waist_radius and end_radius, solve for L:
    # end_radius = waist_radius · cosh(L/(2·waist_radius))
    # L = 2·waist_radius · arccosh(end_radius/waist_radius)
    
    a = waist_radius
    
    # Clamp the ratio to avoid numerical issues
    ratio = max(end_radius / waist_radius, 1.0001)
    effective_L = 2 * a * np.arccosh(ratio)
    
    # Scale s to the effective catenoid parameter
    s_scaled = s * effective_L / 2  # s_scaled ∈ [-effective_L/2, effective_L/2]
    
    # Catenoid radius
    rho = a * torch.cosh(s_scaled / a)
    
    # Position along x-axis
    x = s * neck_length / 2
    
    # Cross-section
    y = rho * torch.cos(theta)
    z = rho * torch.sin(theta)
    
    return torch.stack([x, y, z], dim=1)


def _smooth_step_quintic(t: torch.Tensor) -> torch.Tensor:
    """Quintic smooth step: C² at both endpoints (f' = f'' = 0 at t=0 and t=1).

    Maps t=0 → 0 and t=1 → 1 with zero first and second derivatives at both ends.
    This ensures the blended surface has bounded second derivatives at the blend boundary,
    i.e. H is not artificially spiked by the blending kernel itself.
    """
    t = t.clamp(0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _bridge_radius_optimal(r1: float, r2: float) -> float:
    """
    Catenoid waist radius a that maximises the natural bridge span L₁+L₂.

    For a catenoid connecting tube radii r₁ and r₂, the natural bridge span is
    L₁+L₂ = a·arccosh(r₁/a) + a·arccosh(r₂/a).  This function has a unique
    maximum over a ∈ (0, min(r₁,r₂)), found by coarse grid search.  Using this
    a gives the smoothest possible (least-stretched) catenoid for the given tori.

    Args:
        r1, r2: Tube radii of the two tori

    Returns:
        Optimal waist radius a
    """
    r_min = min(r1, r2)
    a_vals = np.linspace(0.05 * r_min, 0.98 * r_min, 500)
    spans = a_vals * np.arccosh(r1 / a_vals) + a_vals * np.arccosh(r2 / a_vals)
    return float(a_vals[np.argmax(spans)])


def get_double_torus_embedding(
    uv: torch.Tensor,
    tau1: complex = 1j,
    tau2: complex = 1j,
    bridge_radius: Optional[float] = None,
    neck_twist: float = 0.0,
    neck_length: float = 0.0,
    s_blend: float = 0.15,
    **kwargs
) -> torch.Tensor:
    """
    Embed a genus-2 surface: two tori connected by a catenoid bridge.
    
    === GEOMETRY ===
    
    Two tori with major radius axes parallel to z-axis (hole along z):
    - Standard torus: x = (R + r·cos(θ))·cos(φ), y = (R + r·cos(θ))·sin(φ), z = r·sin(θ)
    - φ goes around the major circle (in xy-plane)
    - θ goes around the minor tube
    
    Each torus has a small angular cut where the catenoid attaches:
    - T1: cut near φ=0 (facing +x direction)
    - T2: cut near φ=π (facing -x direction)
    
    The catenoid connects these two cut circles with a proper catenoid profile:
    - Minimum radius a at the center (neck)
    - Radius grows as a·cosh(x/a) toward the ends
    - End radii match the torus tube radii r1, r2
    
    === PARAMETRIZATION ===
    
    Domain: u ∈ [0, 2π), v ∈ [0, 5π)
    
    We use (u, v) where:
    - u ∈ [0, 2π): angle around the tube/catenoid cross-section (θ)
    - v ∈ [0, 5π): position along the surface
    
    The v parameter is divided as:
    - v ∈ [0, 2π): Torus 1 (φ from gap to 2π-gap, avoiding φ=0)
    - v ∈ [2π, 3π): Catenoid bridge (with cosh profile)
    - v ∈ [3π, 5π): Torus 2 (φ from π+gap to 3π-gap, avoiding φ=π)
    
    Args:
        uv: Parameter coordinates (batch_size, 2)
        tau1: Complex modulus for torus 1 (Im controls thickness, Re controls twist)
        tau2: Complex modulus for torus 2
        bridge_radius: Waist radius of the catenoid neck.  If None (default), inferred
            automatically as the value that maximises the natural bridge span
            L₁+L₂ = a·arccosh(r₁/a) + a·arccosh(r₂/a) for the given tube radii.
            This is the smoothest possible catenoid for the current τ₁, τ₂.
            Set explicitly to override (must be < tube radii).
        neck_twist: Additional twist angle applied to torus 2
        neck_length: Extra physical gap added to (or subtracted from) the natural
            catenoid length.  The natural catenoid length L₁+L₂ = a·arccosh(r₁/a)
            + a·arccosh(r₂/a) is the minimum bridge extent needed for the
            cosh profile to reach both tube radii; it depends only on
            bridge_radius and the implicit tube radii.

            neck_length=0: tori positioned so the natural catenoid just fills
              the gap — the most compact valid reference geometry.
            neck_length>0: tori further apart; the catenoid is physically
              stretched (lower curvature from the orientation rotation).
            neck_length<0: tori closer together; the catenoid is compressed.
              Clamped automatically so total_sep ≥ R+r (no self-intersection).
              At neck_length ≈ -(L₁+L₂) the bridge degenerates to a flat disk
              (connected-sum limit).
        s_blend: Half-width (in v units) of the quintic smooth-step transition between
            regions.  The blend weight goes from 0 to 1 over the window
            [junction − s_blend, junction + s_blend].  Because the quintic has
            f' = f'' = 0 at both endpoints, ∂²xyz/∂v² from the blending kernel is zero
            at the window boundaries, suppressing curvature spikes that would otherwise
            appear in H² and destabilise the loss.
    """
    u, v = uv[:, 0], uv[:, 1]
    device = u.device
    dtype = u.dtype
    
    # Parse tau parameters
    if isinstance(tau1, (int, float)):
        tau1 = complex(0, float(tau1))
    if isinstance(tau2, (int, float)):
        tau2 = complex(0, float(tau2))
    
    tau1_im = max(float(tau1.imag), 0.3)
    tau2_im = max(float(tau2.imag), 0.3)
    
    # Compute minor radii from tau (larger Im(τ) → thinner tube)
    r_min, r_max = 0.2, 0.55
    k = 1.5
    r1 = r_min + (r_max - r_min) * np.exp(-k * (tau1_im - 0.3))
    r2 = r_min + (r_max - r_min) * np.exp(-k * (tau2_im - 0.3))
    r1 = np.clip(r1, r_min, r_max)
    r2 = np.clip(r2, r_min, r_max)
    
    R1 = 1.0
    R2 = 1.0

    # Bridge radius: infer from tube radii if not supplied, otherwise clamp to valid range
    if bridge_radius is None:
        a = _bridge_radius_optimal(r1, r2)
    else:
        a = float(np.clip(float(bridge_radius), 0.05 * min(r1, r2), min(r1, r2) * 0.95))
    
    # Catenoid half-lengths: a·arccosh(r/a) gives the x-extent on each side
    L1 = a * np.arccosh(r1 / a)
    L2 = a * np.arccosh(r2 / a)
    
    # Minimum total_sep that prevents torus-tube self-intersection: R1+r1+R2+r2
    # plus a small safety margin.  neck_length can be negative (tori closer
    # together, catenoid compressed) down to this floor.
    min_sep = R1 + r1 + R2 + r2 + 0.05
    total_sep = max(R1 + r1 + R2 + r2 + L1 + L2 + 0.1 + float(neck_length), min_sep)
    x1 = -total_sep / 2
    x2 = +total_sep / 2
    
    attach_x1 = x1 + R1  # x-center of T1 attachment circle (φ=0, lies in xz-plane)
    attach_x2 = x2 - R2  # x-center of T2 attachment circle (φ=π, lies in xz-plane)
    
    v_T1_end = 2 * np.pi   # T1 / catenoid boundary in parameter space
    v_cat_end = 3 * np.pi  # Catenoid / T2 boundary in parameter space
    
    # === Torus 1 formula (evaluated for ALL v) ===
    # φ = v, so at v=2π φ=2π≡0: T1 evaluates exactly on the catenoid attachment
    # circle (y=0 xz-plane), guaranteeing C0 continuity at the blend boundary.
    phi_T1 = v
    # Re(τ₁)/Im(τ₁) twist rate: tube cross-section rotates by one full turn per
    # Re(τ₁)/Im(τ₁) traversals of the major circle, matching the genus-1 convention.
    twist_rate1 = float(tau1.real) / max(abs(float(tau1.imag)), 1e-10)
    theta_T1 = u + twist_rate1 * phi_T1
    rho_T1  = R1 + r1 * torch.cos(theta_T1)
    x_T1 = rho_T1 * torch.cos(phi_T1) + x1
    y_T1 = rho_T1 * torch.sin(phi_T1)
    z_T1 = r1 * torch.sin(theta_T1)
    
    # === Catenoid formula (evaluated for ALL v) ===
    # t ∈ [0, 1]; cosh profile clamped to prevent overflow outside natural range.
    # u_cat interpolates the tube-rotation from T1's holonomy (twist_rate1·2π, the
    # rotation accumulated going once around T1's major circle) to neck_twist, so:
    #   t=0: u_cat = u + twist_rate1·2π  ← matches theta_T1 at v=2π (φ=2π≡0)
    #   t=1: u_cat = u + neck_twist       ← matches theta_T2 at v=3π (φ=π)
    holonomy_T1 = twist_rate1 * 2.0 * np.pi
    t_cat = (v - v_T1_end) / np.pi
    u_cat = u + holonomy_T1 + t_cat * (neck_twist - holonomy_T1)
    s_cat = torch.clamp(-L1 + t_cat * (L1 + L2), -10 * a, 10 * a)
    rho_cat = a * torch.cosh(s_cat / a)
    center_x_cat = attach_x1 + (attach_x2 - attach_x1) * t_cat
    orient_cat   = np.pi * t_cat  # 0 at T1 end, π at T2 end
    local_y_cat  = rho_cat * torch.cos(u_cat)
    local_z_cat  = rho_cat * torch.sin(u_cat)
    x_cat = center_x_cat + local_y_cat * torch.cos(orient_cat)
    y_cat = local_y_cat * torch.sin(orient_cat)
    z_cat = local_z_cat
    
    # === Torus 2 formula (evaluated for ALL v) ===
    # φ = π + v_local, so at v=3π φ=π: T2 evaluates exactly on the catenoid
    # attachment circle, guaranteeing C0 continuity at the blend boundary.
    # (phi_T2 - π) normalises the major angle to [0, 2π] for twist accumulation;
    # neck_twist offsets the tube angle at the bridge to match u_cat at t=1.
    v_local_T2 = v - v_cat_end
    phi_T2 = np.pi + v_local_T2
    twist_rate2 = float(tau2.real) / max(abs(float(tau2.imag)), 1e-10)
    theta_T2 = u + twist_rate2 * (phi_T2 - np.pi) + neck_twist
    rho_T2  = R2 + r2 * torch.cos(theta_T2)
    x_T2 = rho_T2 * torch.cos(phi_T2) + x2
    y_T2 = rho_T2 * torch.sin(phi_T2)
    z_T2 = r2 * torch.sin(theta_T2)
    
    # === Smooth sigmoid blending ===
    # Each formula is evaluated everywhere; sigmoid weights localise contributions to
    # their natural region. The C0 boundary match ensures no geometric seam; the
    # sigmoid provides C∞ smoothing across the transition zone.
    # Quintic smooth step: 0 → 1 over [junction − s_blend, junction + s_blend].
    # f' = f'' = 0 at both window endpoints → C² blending, no curvature spike from kernel.
    w_to_cat = _smooth_step_quintic((v - (v_T1_end - s_blend)) / (2.0 * s_blend))  # T1 → catenoid at v = 2π
    w_to_T2  = _smooth_step_quintic((v - (v_cat_end - s_blend)) / (2.0 * s_blend))  # catenoid → T2 at v = 3π
    
    w1 = (1.0 - w_to_cat).unsqueeze(1)              # T1 weight
    w2 = (w_to_cat * (1.0 - w_to_T2)).unsqueeze(1)  # Catenoid weight
    w3 = (w_to_cat * w_to_T2).unsqueeze(1)          # T2 weight
    
    xyz_T1  = torch.stack([x_T1,  y_T1,  z_T1],  dim=1)
    xyz_cat = torch.stack([x_cat, y_cat, z_cat], dim=1)
    xyz_T2  = torch.stack([x_T2,  y_T2,  z_T2],  dim=1)
    
    return w1 * xyz_T1 + w2 * xyz_cat + w3 * xyz_T2


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
        # Get double torus parameters (Two tori + catenoid bridge)
        dt_params = topology_params.get('double_torus', {})
        
        # Parse tau1 (complex modular parameter for torus 1)
        tau1_raw = dt_params.get('tau1', {'real': 0.0, 'imag': 1.0})
        if isinstance(tau1_raw, dict):
            tau1 = complex(tau1_raw.get('real', 0.0), tau1_raw.get('imag', 1.0))
        elif isinstance(tau1_raw, str):
            tau1 = complex(tau1_raw.replace('i', 'j'))
        else:
            tau1 = complex(tau1_raw)
        
        # Parse tau2 (complex modular parameter for torus 2)
        tau2_raw = dt_params.get('tau2', {'real': 0.0, 'imag': 1.0})
        if isinstance(tau2_raw, dict):
            tau2 = complex(tau2_raw.get('real', 0.0), tau2_raw.get('imag', 1.0))
        elif isinstance(tau2_raw, str):
            tau2 = complex(tau2_raw.replace('i', 'j'))
        else:
            tau2 = complex(tau2_raw)
        
        # Catenoid bridge parameters
        neck_length = float(dt_params.get('neck_length', 0.0))
        neck_twist = float(dt_params.get('neck_twist', 0.0))
        bridge_radius_raw = dt_params.get('bridge_radius', None)
        bridge_radius = None if bridge_radius_raw is None else float(bridge_radius_raw)
        s_blend = float(dt_params.get('s_blend', 0.15))
        return get_double_torus_embedding(
            uv,
            tau1=tau1,
            tau2=tau2,
            bridge_radius=bridge_radius,
            neck_twist=neck_twist,
            neck_length=neck_length,
            s_blend=s_blend,
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
