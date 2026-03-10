"""
Loss Functions for Embedding-Based Willmore Energy Minimization

This module implements the Willmore functional and related loss functions
for training a neural network to learn an embedding φ: (u,v) → (x,y,z) 
that minimizes the Willmore energy.

Supports different topologies:
- Genus 0 (sphere/ellipsoid): Standard Willmore minimization
- Genus 1 (torus): Standard Willmore minimization
- Genus 2 (double torus): Extended domain parametrization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class EmbeddingWillmoreLoss(nn.Module):
    """
    Computes the true Willmore energy functional for embedded surfaces.
    
    The Willmore energy is:
    W = ∫∫ H² dA
    
    where H is the mean curvature and dA is the area element.
    
    Supports different domain types:
    - ellipsoid: [0, 2π] × [0, π] domain
    - torus: [0, 2π] × [0, 2π] domain
    - double_torus: [0, 2π] × [0, 5π] domain
    """
    
    def __init__(self, epsilon: float = 1e-8, domain: str = "torus", genus: Optional[int] = None,
                 h2_clip: Optional[float] = None):
        """
        Args:
            epsilon: Small constant for numerical stability
            domain: Surface domain type
            genus: Surface genus (overrides domain if provided)
            h2_clip: If set, per-point H² values are clamped to this ceiling before averaging.
                This bounds the gradient contribution from high-curvature junction regions
                without biasing the minimum (the Willmore minimiser has bounded H everywhere).
        """
        super().__init__()
        self.epsilon = epsilon
        self.h2_clip = h2_clip
        
        # Determine domain from genus if provided
        if genus is not None:
            if genus == 0:
                domain = "ellipsoid"
            elif genus == 1:
                domain = "torus"
            elif genus == 2:
                domain = "double_torus"
        
        self.domain = domain.lower()
        self.genus = genus
        
        # Compute domain area for Monte Carlo integration
        if self.domain in ["ellipsoid", "sphere"]:
            # [0, 2π] × [0, π]
            self.domain_area = 2 * np.pi * np.pi
        elif self.domain == "torus":
            # [0, 2π] × [0, 2π]
            self.domain_area = (2 * np.pi) ** 2
        elif self.domain == "double_torus":
            # [0, 2π] × [0, 5π]
            self.domain_area = 2 * np.pi * 5 * np.pi
        else:
            # Default to torus domain
            self.domain_area = (2 * np.pi) ** 2
    
    def forward(self, model: nn.Module, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute the Willmore energy for the embedded surface.
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Willmore energy (scalar)
        """
        # Ensure gradients are enabled
        uv = uv.requires_grad_(True)
        
        # Compute first fundamental form
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv)
        
        # Compute second fundamental form
        L, M, N, normal = model.compute_second_fundamental_form(uv, phi_u, phi_v)
        
        # Compute mean curvature (this is already the geometric mean curvature)
        H = model.compute_mean_curvature(E, F, G, L, M, N, self.epsilon)
        
        # Compute area element: dA = √(EG - F²) du dv
        # clamp: EG-F² ≥ 0 for any Gram matrix; consistent with compute_mean_curvature
        area_element = torch.sqrt(torch.clamp(E * G - F * F, min=self.epsilon))

        # UNIFORM SAMPLING in parameter space for all topologies.
        # Samples are drawn uniformly over the parameter domain; the area element
        # √(EG-F²) is explicit in the integrand so no importance correction is needed.
        #
        # Monte Carlo estimator:
        # W = ∫∫ H² √(EG-F²) du dv ≈ domain_area × mean(H² √(EG-F²))
        #
        # domain_area:
        #   ellipsoid/sphere : 2π × π  = 2π²
        #   torus            : 2π × 2π = 4π²
        #   double_torus     : 2π × 5π = 10π²
        integrand = H * H * area_element

        # Always compute uncapped energy for honest reporting (detached — no gradient path)
        uncapped_willmore = (torch.mean(integrand.detach()) * self.domain_area).item()

        if self.h2_clip is not None:
            # Huber loss on H²: matches exact Willmore (H²) for |H| ≤ √h2_clip, then
            # switches to linear in |H| above the threshold.
            # Gradient: 2H below threshold (exact Willmore gradient);
            #           2√h2_clip · sign(H) above — constant non-zero magnitude, always
            #           pushing toward H=0.  Unlike a hard clamp, the gradient never
            #           vanishes at high-curvature junctions.  The Willmore minimum is
            #           unaffected: the minimiser has bounded H, so H² < h2_clip there.
            sqrt_clip = self.h2_clip ** 0.5
            h2 = H * H
            h2_huber = torch.where(h2 <= self.h2_clip,
                                   h2,
                                   2.0 * sqrt_clip * H.abs() - self.h2_clip)
            integrand = h2_huber * area_element

        training_energy = torch.mean(integrand) * self.domain_area

        # Return (training_tensor_for_backward, uncapped_scalar_for_reporting)
        return training_energy, uncapped_willmore


class RegularityLoss(nn.Module):
    """
    Regularization loss to maintain well-conditioned metric and smooth parametrization.
    Ensures the embedding stays non-degenerate with bounded derivatives.
    Includes area element preservation, orientation preservation, metric positivity, and smoothness.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        min_area_element: float = 0.01,
        area_element_weight: float = 1.0,
        orientation_weight: float = 1.0,
        metric_positivity_weight: float = 1.0,
        smoothness_weight: float = 1.0,
        max_metric_value: float = 10.0
    ):
        """
        Args:
            epsilon: Small constant for numerical stability
            min_area_element: Minimum allowed area element (prevents collapse)
            area_element_weight: Weight for area element loss term
            orientation_weight: Weight for orientation preservation term
            metric_positivity_weight: Weight for metric positivity term
            smoothness_weight: Weight for smoothness term
            max_metric_value: Upper threshold for E and G; only excess beyond this is penalised
        """
        super().__init__()
        self.epsilon = epsilon
        self.min_area_element = min_area_element
        self.max_metric_value = max_metric_value
        
        # Store initial weights
        self.area_element_weight = area_element_weight
        self.orientation_weight = orientation_weight
        self.metric_positivity_weight = metric_positivity_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(self, model: nn.Module, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute regularity loss.
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Regularity loss (scalar)
        """
        uv = uv.requires_grad_(True)
        
        # Compute first fundamental form (needed by multiple components)
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv)
        
        # Only compute loss components with non-zero weights
        total_loss = torch.tensor(0.0, device=uv.device)
        weight_sum = 0.0
        
        # Penalize extremely small area elements (prevents local collapse)
        if self.area_element_weight > 0:
            det = torch.clamp(E * G - F * F, min=self.epsilon)
            area_element = torch.sqrt(det)
            area_element_loss = torch.mean(torch.nn.functional.relu(self.min_area_element - area_element) ** 2)
            total_loss += self.area_element_weight * area_element_loss
            weight_sum += self.area_element_weight
        
        # Check orientation is preserved (normal should point consistently)
        # Cross product magnitude should stay bounded away from zero
        if self.orientation_weight > 0:
            cross_magnitude = torch.norm(torch.cross(phi_u, phi_v, dim=1), dim=1)
            orientation_loss = torch.mean(torch.nn.functional.relu(self.min_area_element - cross_magnitude) ** 2)
            total_loss += self.orientation_weight * orientation_loss
            weight_sum += self.orientation_weight
        
        # Penalize extreme metric distortion (prevents degeneration)
        # E and G should be positive and not too different (isotropy encouragement)
        if self.metric_positivity_weight > 0:
            metric_positivity = torch.mean(torch.nn.functional.relu(0.001 - E) ** 2) + \
                               torch.mean(torch.nn.functional.relu(0.001 - G) ** 2)
            total_loss += self.metric_positivity_weight * metric_positivity
            weight_sum += self.metric_positivity_weight
        
        # Smoothness: penalise only when E or G exceeds the permissible limit
        # (ReLU gate — zero gradient within the acceptable band)
        if self.smoothness_weight > 0:
            smoothness_loss = torch.mean(
                torch.nn.functional.relu(E - self.max_metric_value) ** 2
                + torch.nn.functional.relu(G - self.max_metric_value) ** 2
            )
            total_loss += self.smoothness_weight * smoothness_loss
            weight_sum += self.smoothness_weight
        
        # Normalize by weight sum if any components are active
        if weight_sum > 0:
            total_loss = total_loss / weight_sum
        
        return total_loss


class CombinedEmbeddingLoss(nn.Module):
    """
    Combined loss function for embedding-based Willmore minimization.
    
    Supports different topologies with appropriate constraints:
    - Genus 0: Willmore + regularity
    - Genus 1: Willmore + regularity
    - Genus 2: Willmore + regularity
    """
    
    def __init__(
        self,
        willmore_weight: float = 1.0,
        regularity_weight: float = 0.1,
        target_area: Optional[float] = None,
        epsilon: float = 1e-8,
        regularity_area_element_weight: float = 1.0,
        regularity_orientation_weight: float = 1.0,
        regularity_metric_positivity_weight: float = 1.0,
        regularity_smoothness_weight: float = 1.0,
        regularity_max_metric_value: float = 10.0,
        regularity_min_area_element: float = 0.01,
        genus: Optional[int] = None,
        domain: str = "torus",
        max_willmore_weight: float = 0.5,
        h2_clip: Optional[float] = None,
        topology_guard_weight: float = 0.0,
        topology_guard_num_probes: int = 64
    ):
        """
        Args:
            willmore_weight: Weight for Willmore energy term
            regularity_weight: Weight for metric regularity preservation
            target_area: Target surface area (None for adaptive)
            epsilon: Small constant for numerical stability
            regularity_area_element_weight: Weight for area element term within regularity loss
            regularity_orientation_weight: Weight for orientation term within regularity loss
            regularity_metric_positivity_weight: Weight for metric positivity term within regularity loss
            regularity_smoothness_weight: Weight for smoothness term within regularity loss
            regularity_max_metric_value: Upper threshold for E and G in the smoothness term
            regularity_min_area_element: Minimum allowed area element √(EG−F²); collapse below this is penalised
            genus: Surface genus (0, 1, or 2)
            domain: Surface domain type
            max_willmore_weight: Ceiling for willmore_weight annealing schedule
            h2_clip: Per-point H² ceiling passed to EmbeddingWillmoreLoss (None = no clipping)
            topology_guard_weight: Weight for the topology guard loss (genus 2 only). Evaluates
                regularity at fixed probe locations deep in each torus body, guaranteeing gradient
                signal that prevents handle collapse regardless of main sampling distribution.
            topology_guard_num_probes: Number of random-u probe points per torus body per forward pass.
        """
        super().__init__()
        
        self.willmore_weight = willmore_weight
        self.regularity_weight = regularity_weight
        self.genus = genus
        self.domain = domain
        self.topology_guard_weight = topology_guard_weight
        self.topology_guard_num_probes = topology_guard_num_probes
        # Fixed v₀ values for genus 2 topology guard probes:
        #   T1 body midpoint  : v = π    (T1 occupies v ∈ [0, 2π))
        #   Bridge midpoint   : v = 2.5π (bridge occupies v ∈ [2π, 3π))
        #   T2 body midpoint  : v = 4π   (T2 occupies v ∈ [3π, 5π])
        self._topology_guard_v_values = [np.pi, 2.5 * np.pi, 4 * np.pi]
        
        self.willmore_loss = EmbeddingWillmoreLoss(epsilon=epsilon, domain=domain, genus=genus,
                                                   h2_clip=h2_clip)
        self.regularity_loss = RegularityLoss(
            epsilon=epsilon,
            min_area_element=regularity_min_area_element,
            area_element_weight=regularity_area_element_weight,
            orientation_weight=regularity_orientation_weight,
            metric_positivity_weight=regularity_metric_positivity_weight,
            smoothness_weight=regularity_smoothness_weight,
            max_metric_value=regularity_max_metric_value
        )
        
        # Store initial weights for reference
        self.initial_willmore_weight = willmore_weight
        self.initial_regularity_weight = regularity_weight
        self.initial_weight_sum = willmore_weight + regularity_weight
        self.max_willmore_weight = max_willmore_weight
    
    def update_weights(self, epoch: int, total_epochs: int, regularity_value: Optional[float] = None, 
                      adaptive_config: Optional[dict] = None):
        """
        Progressively adjust loss weights during training with adaptive safeguards.
        Early training: High regularity weight, low Willmore weight
        Late training: Lower regularity weight, moderate Willmore weight
        
        ADAPTIVE SAFEGUARD: If regularity loss increases (surface becoming degenerate),
        automatically boost regularity weight to prevent collapse.
        
        Args:
            epoch: Current epoch number (1-indexed)
            total_epochs: Total number of training epochs
            regularity_value: Current regularity loss value (for adaptive adjustment)
            adaptive_config: Configuration for adaptive training safeguards
        """
        adaptive_enabled = adaptive_config and adaptive_config.get('enabled', False)

        if not adaptive_enabled:
            # Freeze weights at initial values; no scheduling
            self.willmore_weight = self.initial_willmore_weight
            self.regularity_weight = self.initial_regularity_weight
            return

        progress = min(1.0, (epoch - 1) / max(1, total_epochs))
        
        # Base schedule: Willmore gradually increases to max_willmore_weight, regularity decreases
        base_willmore = self.initial_willmore_weight + (self.max_willmore_weight - self.initial_willmore_weight) * progress
        base_regularity = self.initial_regularity_weight * (1.0 - 0.5 * progress)
        
        # Adaptive adjustment based on regularity health
        if adaptive_enabled and regularity_value is not None:
            threshold = adaptive_config.get('regularity_threshold', 0.5)
            boost_factor = adaptive_config.get('regularity_boost_factor', 2.0)
            
            # If regularity loss is high, boost its weight to stabilize
            if regularity_value > threshold:
                regularity_multiplier = boost_factor
                print(f"\n⚠️  Regularity loss {regularity_value:.4f} exceeds threshold {threshold:.4f}")
                print(f"    Boosting regularity weight by {boost_factor}x to prevent collapse")
            else:
                regularity_multiplier = 1.0
            
            self.regularity_weight = base_regularity * regularity_multiplier
        else:
            self.regularity_weight = base_regularity
        
        self.willmore_weight = base_willmore
    
    def forward(self, model: nn.Module, uv: torch.Tensor) -> dict:
        """
        Compute combined loss.
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Dictionary with total loss and individual components
        """
        # Only compute loss components with non-zero weights
        total_loss = torch.tensor(0.0, device=uv.device)
        willmore_value = 0.0
        regularity_value = 0.0
        
        # Compute Willmore loss if weight > 0
        if self.willmore_weight > 0:
            willmore, willmore_value = self.willmore_loss(model, uv)
            # willmore      — capped training tensor (stable backward pass)
            # willmore_value — uncapped true W (honest metric for logging and rollback)
            total_loss += self.willmore_weight * willmore
        
        # Compute regularity loss if weight > 0
        if self.regularity_weight > 0:
            regularity = self.regularity_loss(model, uv)
            regularity_value = regularity.item()
            total_loss += self.regularity_weight * regularity

        # Topology guard (genus 2 only): evaluate regularity at fixed probe locations
        # deep in each torus body (v = π for T1, v = 4π for T2), using randomly sampled u.
        # This fires regardless of the main sampling distribution, preventing handle collapse.
        if self.genus == 2 and self.topology_guard_weight > 0:
            guard_loss = torch.tensor(0.0, device=uv.device)
            n = self.topology_guard_num_probes
            for v0 in self._topology_guard_v_values:
                u_probe = torch.rand(n, device=uv.device, dtype=uv.dtype) * 2 * np.pi
                v_probe = torch.full((n,), v0, device=uv.device, dtype=uv.dtype)
                probe_uv = torch.stack([u_probe, v_probe], dim=1)
                guard_loss = guard_loss + self.regularity_loss(model, probe_uv)
            guard_loss = guard_loss / len(self._topology_guard_v_values)
            total_loss = total_loss + self.topology_guard_weight * guard_loss

        # Normalise by fixed initial weight sum so scale is stable as weights anneal
        if self.initial_weight_sum > 0:
            total_loss = total_loss / self.initial_weight_sum

        return {
            'total': total_loss,
            'willmore': willmore_value,
            'regularity': regularity_value
        }


def create_embedding_loss(config: dict) -> nn.Module:
    """
    Factory function to create loss from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Combined loss function
    """
    loss_config = config.get("loss", {})
    topology_config = config.get("topology", {})
    
    # Get genus from topology config
    genus = topology_config.get("genus", 1)
    
    # Validate genus
    if genus < 0:
        raise ValueError(f"Genus must be non-negative, got {genus}")
    if genus > 2:
        raise NotImplementedError(f"Genus {genus} is not supported. Only genus 0, 1, 2 are implemented.")
    
    # Determine domain from genus
    from sampling import get_domain_for_genus
    domain = get_domain_for_genus(genus)
    
    return CombinedEmbeddingLoss(
        willmore_weight=loss_config.get("willmore_weight", 1.0),
        regularity_weight=loss_config.get("regularity_weight", 0.1),
        target_area=loss_config.get("target_area", None),
        epsilon=loss_config.get("epsilon", 1e-8),
        regularity_area_element_weight=loss_config.get("regularity_area_element_weight", 1.0),
        regularity_orientation_weight=loss_config.get("regularity_orientation_weight", 1.0),
        regularity_metric_positivity_weight=loss_config.get("regularity_metric_positivity_weight", 1.0),
        regularity_smoothness_weight=loss_config.get("regularity_smoothness_weight", 1.0),
        regularity_max_metric_value=loss_config.get("regularity_max_metric_value", 10.0),
        regularity_min_area_element=loss_config.get("regularity_min_area_element", 0.01),
        genus=genus,
        domain=domain,
        max_willmore_weight=loss_config.get("max_willmore_weight", 0.5),
        h2_clip=loss_config.get("h2_clip", None),
        topology_guard_weight=loss_config.get("topology_guard_weight", 0.0),
        topology_guard_num_probes=loss_config.get("topology_guard_num_probes", 64),
    )


# ============================================================================
# Additional Loss Functions (Currently Disabled)
# ============================================================================


class AreaConstraintLoss(nn.Module):
    """
    Constraint to maintain appropriate surface area.
    Prevents collapse or explosion of the surface.
    """
    
    def __init__(
        self, 
        target_area: Optional[float] = None,
        epsilon: float = 1e-8
    ):
        """
        Args:
            target_area: Target total surface area (None for adaptive)
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.target_area = target_area
        self.epsilon = epsilon
    
    def forward(self, model: nn.Module, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute area constraint loss.
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Area constraint loss (scalar)
        """
        uv = uv.requires_grad_(True)
        
        # Compute first fundamental form
        E, F, G, _, _ = model.compute_first_fundamental_form(uv)
        
        # Compute area element
        area_element = torch.sqrt(torch.abs(E * G - F * F) + self.epsilon)
        
        # Total area using proper Monte Carlo integration
        # Integral ≈ (domain_volume / N) * sum = domain_volume * mean
        domain_area = (2 * 3.14159265359) ** 2  # (2π)² for [0,2π] × [0,2π]
        total_area = domain_area * torch.mean(area_element)
        
        if self.target_area is not None:
            # Penalize deviation from target area
            loss = ((total_area - self.target_area) / self.target_area) ** 2
        else:
            # For torus with R, r: area = 4π²Rr
            # Starting torus (R=3, r=0.5): area ≈ 59.22
            # Clifford torus (R=√2, r=1): area ≈ 55.55
            # Use tight bounds to prevent collapse
            min_area = 30.0  # Much tighter lower bound
            max_area = 150.0  # Reasonable upper bound
            
            # Strong penalty for going below minimum (prevents collapse)
            collapse_penalty = torch.nn.functional.relu(min_area - total_area) ** 2
            # Moderate penalty for excessive area
            explosion_penalty = torch.nn.functional.relu(total_area - max_area) ** 2
            
            loss = 10.0 * collapse_penalty + explosion_penalty  # Weight collapse prevention heavily
        
        return loss


class SymmetryLoss(nn.Module):
    """
    Encourages rotational symmetry around the z-axis for the torus.
    Penalizes deviation from axisymmetric configuration.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, model: nn.Module, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetry loss.
        
        For a torus symmetric around z-axis:
        1. Distance from z-axis should only depend on v (not u)
        2. z-coordinate should only depend on v (not u)
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Symmetry loss (scalar)
        """
        # Get embedding
        xyz = model.forward(uv)
        
        # Distance from z-axis: r = sqrt(x² + y²)
        r_xy = torch.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + self.epsilon)
        z = xyz[:, 2]
        
        # For perfect symmetry, r and z should be functions of v only
        # Sample pairs with same v but different u
        batch_size = uv.shape[0]
        if batch_size >= 2:
            # Split batch in half and create pairs
            half = batch_size // 2
            
            # First half and second half
            uv1 = uv[:half]
            uv2 = uv[half:2*half]
            
            # Create pairs with same v, different u
            uv_paired = torch.stack([
                torch.stack([uv1[:, 0], uv2[:, 1]], dim=1),  # Different u, v from uv2
                torch.stack([uv2[:, 0], uv2[:, 1]], dim=1),  # Different u, same v
            ], dim=0)
            
            xyz_paired = torch.stack([
                model.forward(uv_paired[0]),
                model.forward(uv_paired[1])
            ], dim=0)
            
            r_xy_paired = torch.sqrt(xyz_paired[:, :, 0]**2 + xyz_paired[:, :, 1]**2 + self.epsilon)
            z_paired = xyz_paired[:, :, 2]
            
            # Points with same v should have same r and z
            r_diff = torch.mean((r_xy_paired[0] - r_xy_paired[1]) ** 2)
            z_diff = torch.mean((z_paired[0] - z_paired[1]) ** 2)
            
            return r_diff + z_diff
        else:
            return torch.tensor(0.0, device=uv.device)
