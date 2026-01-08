"""
Loss Functions for Embedding-Based Willmore Energy Minimization

This module implements the Willmore functional and related loss functions
for training a neural network to learn an embedding φ: (u,v) → (x,y,z) 
that minimizes the Willmore energy.

Supports different topologies:
- Genus 0 (sphere/ellipsoid): Includes volume constraint to prevent collapse
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
    - double_torus: [0, 2π] × [0, 4π] domain
    """
    
    def __init__(self, epsilon: float = 1e-8, domain: str = "torus", genus: Optional[int] = None):
        """
        Args:
            epsilon: Small constant for numerical stability
            domain: Surface domain type
            genus: Surface genus (overrides domain if provided)
        """
        super().__init__()
        self.epsilon = epsilon
        
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
            # [0, 2π] × [0, 4π]
            self.domain_area = 2 * np.pi * 4 * np.pi
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
        
        # Compute area element: dA = sqrt(EG - F²) du dv
        # This is the surface area measure in parameter coordinates
        area_element = torch.sqrt(torch.abs(E * G - F * F) + self.epsilon)
        
        # IMPORTANT: For genus 0 (ellipsoid/sphere), we use area-weighted sampling
        # where points are distributed uniformly on the SURFACE (not parameter space).
        # This means the probability density is already proportional to the area element!
        # 
        # For area-weighted sampling: p(u,v) ∝ √(EG-F²)
        # Monte Carlo estimator: ∫ f dA ≈ (1/N) Σ f(u_i,v_i) / p(u_i,v_i)
        #                              ≈ (Surface_Area/N) Σ H²(u_i,v_i)
        # 
        # For uniform sampling in parameter space: p(u,v) = 1/domain_area
        # Monte Carlo estimator: ∫ f dA ≈ (domain_area/N) Σ f(u_i,v_i) * √(EG-F²)
        #
        # The code MUST match the sampling strategy!
        
        if self.domain in ["ellipsoid", "sphere"]:
            # AREA-WEIGHTED SAMPLING (genus 0):
            # Samples: u ~ Uniform[0,2π], cos(v) ~ Uniform[-1,1]
            # Joint density: p(u,v) = (1/(2π)) × (sin(v)/2) = sin(v)/(4π)
            #
            # Monte Carlo importance sampling estimator:
            # W = ∫₀^{2π} ∫₀^π H²(u,v) √(EG-F²)(u,v) du dv
            #   = 𝔼_p[f/p] where f = H² √(EG-F²), p = sin(v)/(4π)
            #   ≈ (1/N) Σ [f_i / p_i]
            #   = (1/N) Σ [H²_i × √(EG-F²)_i / (sin(v_i)/(4π))]
            #   = (4π/N) Σ [H²_i × √(EG-F²)_i / sin(v_i)]
            #
            # Extract v from uv samples (second column)
            v = uv[:, 1]  # v ∈ [0, π]
            sin_v = torch.sin(v) + self.epsilon  # Add epsilon to avoid division by zero
            
            # Compute importance sampling weights: 4π / sin(v)
            importance_weights = (4 * np.pi) / sin_v
            
            # Weighted integrand
            integrand = H * H * area_element * importance_weights
            willmore_energy = torch.mean(integrand)
        else:
            # UNIFORM SAMPLING in parameter space (genus 1: torus, genus 2: double torus)
            # Samples: u ~ Uniform, v ~ Uniform over respective domains
            # Sampling density: p(u,v) = 1/domain_area (constant)
            #
            # Monte Carlo estimator with uniform sampling:
            # W = ∫∫ H² √(EG-F²) du dv ≈ (domain_area/N) Σ [H²_i × √(EG-F²)_i]
            #   = domain_area × mean(H² × √(EG-F²))
            integrand = H * H * area_element
            willmore_energy = torch.mean(integrand) * self.domain_area
        
        return willmore_energy


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
        smoothness_weight: float = 1.0
    ):
        """
        Args:
            epsilon: Small constant for numerical stability
            min_area_element: Minimum allowed area element (prevents collapse)
            area_element_weight: Weight for area element loss term
            orientation_weight: Weight for orientation preservation term
            metric_positivity_weight: Weight for metric positivity term
            smoothness_weight: Weight for smoothness term
        """
        super().__init__()
        self.epsilon = epsilon
        self.min_area_element = min_area_element
        
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
        
        # Compute first fundamental form
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv)
        
        # Penalize extremely small area elements (prevents local collapse)
        det = E * G - F * F
        area_element = torch.sqrt(torch.abs(det) + self.epsilon)
        area_element_loss = torch.mean(torch.nn.functional.relu(self.min_area_element - area_element) ** 2)
        
        # Check orientation is preserved (normal should point consistently)
        # Cross product magnitude should stay bounded away from zero
        cross_magnitude = torch.norm(torch.cross(phi_u, phi_v, dim=1), dim=1)
        orientation_loss = torch.mean(torch.nn.functional.relu(self.min_area_element - cross_magnitude) ** 2)
        
        # Penalize extreme metric distortion (prevents degeneration)
        # E and G should be positive and not too different (isotropy encouragement)
        metric_positivity = torch.mean(torch.nn.functional.relu(0.001 - E) ** 2) + \
                           torch.mean(torch.nn.functional.relu(0.001 - G) ** 2)
        
        # Smoothness: penalize large derivatives (E = |φ_u|², G = |φ_v|²)
        smoothness_loss = torch.mean(E + G)
        
        # Normalize weights so they sum to 1.0
        weight_sum = (
            self.area_element_weight +
            self.orientation_weight +
            self.metric_positivity_weight +
            self.smoothness_weight
        )
        
        # Weighted combination with normalized weights
        total_loss = (
            (self.area_element_weight / weight_sum) * area_element_loss +
            (self.orientation_weight / weight_sum) * orientation_loss +
            (self.metric_positivity_weight / weight_sum) * metric_positivity +
            (self.smoothness_weight / weight_sum) * smoothness_loss
        )
        
        return total_loss


class VolumeMinimumConstraint(nn.Module):
    """
    ReLU-style constraint to maintain minimum enclosed volume.
    
    Critical for genus 0 surfaces (sphere/ellipsoid) to prevent collapse to a point.
    Only penalizes when volume falls below the minimum threshold.
    
    Loss = weight * relu(min_volume - current_volume)²
    """
    
    def __init__(
        self,
        min_volume: float = 1.0,
        weight: float = 10.0,
        epsilon: float = 1e-8,
        domain: str = "ellipsoid",
        genus: Optional[int] = None
    ):
        """
        Args:
            min_volume: Minimum allowed enclosed volume
            weight: Weight for the constraint loss
            epsilon: Small constant for numerical stability
            domain: Surface domain type
            genus: Surface genus (overrides domain if provided)
        """
        super().__init__()
        self.min_volume = min_volume
        self.weight = weight
        self.epsilon = epsilon
        
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
            self.domain_area = 2 * np.pi * np.pi  # [0, 2π] × [0, π]
        elif self.domain == "torus":
            self.domain_area = (2 * np.pi) ** 2  # [0, 2π] × [0, 2π]
        elif self.domain == "double_torus":
            self.domain_area = 2 * np.pi * 4 * np.pi  # [0, 2π] × [0, 4π]
        else:
            self.domain_area = (2 * np.pi) ** 2
    
    def forward(self, model: nn.Module, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute volume minimum constraint loss.
        
        Uses the divergence theorem to compute enclosed volume:
        Volume = (1/3) ∫∫ (x·n) dA
        
        where n is the outward unit normal.
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Volume constraint loss (scalar)
        """
        uv = uv.requires_grad_(True)
        
        # Get embedding and derivatives
        xyz = model.forward(uv)
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv)
        
        # Compute unit normal
        normal_unnorm = torch.cross(phi_u, phi_v, dim=1)
        normal_norm = torch.norm(normal_unnorm, dim=1, keepdim=True) + self.epsilon
        normal = normal_unnorm / normal_norm
        
        # Compute area element
        area_element = torch.sqrt(torch.abs(E * G - F * F) + self.epsilon)
        
        # Volume integrand using divergence theorem: (1/3) * (x·n) * dA
        position_normal_dot = torch.sum(xyz * normal, dim=1)
        volume_integrand = (1.0 / 3.0) * position_normal_dot * area_element
        
        # Monte Carlo integration
        enclosed_volume = torch.abs(self.domain_area * torch.mean(volume_integrand))
        
        # ReLU-style penalty: only penalize when below minimum
        volume_deficit = torch.nn.functional.relu(self.min_volume - enclosed_volume)
        loss = self.weight * volume_deficit ** 2
        
        return loss, enclosed_volume


class CombinedEmbeddingLoss(nn.Module):
    """
    Combined loss function for embedding-based Willmore minimization.
    
    Supports different topologies with appropriate constraints:
    - Genus 0: Willmore + regularity + volume minimum constraint
    - Genus 1: Willmore + regularity
    - Genus 2: Willmore + regularity
    """
    
    def __init__(
        self,
        willmore_weight: float = 1.0,
        regularity_weight: float = 0.1,
        target_area: Optional[float] = None,
        target_volume: Optional[float] = None,
        epsilon: float = 1e-8,
        regularity_area_element_weight: float = 1.0,
        regularity_orientation_weight: float = 1.0,
        regularity_metric_positivity_weight: float = 1.0,
        regularity_smoothness_weight: float = 1.0,
        genus: Optional[int] = None,
        domain: str = "torus",
        volume_constraint_enabled: bool = False,
        volume_constraint_min: float = 1.0,
        volume_constraint_weight: float = 10.0
    ):
        """
        Args:
            willmore_weight: Weight for Willmore energy term
            regularity_weight: Weight for metric regularity preservation
            target_area: Target surface area (None for adaptive)
            target_volume: Target enclosed volume (None for adaptive)
            epsilon: Small constant for numerical stability
            regularity_area_element_weight: Weight for area element term within regularity loss
            regularity_orientation_weight: Weight for orientation term within regularity loss
            regularity_metric_positivity_weight: Weight for metric positivity term within regularity loss
            regularity_smoothness_weight: Weight for smoothness term within regularity loss
            genus: Surface genus (0, 1, or 2)
            domain: Surface domain type
            volume_constraint_enabled: Whether to enable volume minimum constraint
            volume_constraint_min: Minimum allowed volume (for genus 0)
            volume_constraint_weight: Weight for volume constraint
        """
        super().__init__()
        
        self.willmore_weight = willmore_weight
        self.regularity_weight = regularity_weight
        self.genus = genus
        self.domain = domain
        
        # Determine if volume constraint should be enabled
        # By default, enable for genus 0 unless explicitly disabled
        if genus == 0 and volume_constraint_enabled:
            self.use_volume_constraint = True
        else:
            self.use_volume_constraint = volume_constraint_enabled
        
        self.willmore_loss = EmbeddingWillmoreLoss(epsilon=epsilon, domain=domain, genus=genus)
        self.regularity_loss = RegularityLoss(
            epsilon=epsilon,
            area_element_weight=regularity_area_element_weight,
            orientation_weight=regularity_orientation_weight,
            metric_positivity_weight=regularity_metric_positivity_weight,
            smoothness_weight=regularity_smoothness_weight
        )
        
        # Volume constraint for genus 0
        if self.use_volume_constraint:
            self.volume_constraint = VolumeMinimumConstraint(
                min_volume=volume_constraint_min,
                weight=volume_constraint_weight,
                epsilon=epsilon,
                domain=domain,
                genus=genus
            )
        else:
            self.volume_constraint = None
        
        # Store initial weights for reference
        self.initial_willmore_weight = willmore_weight
        self.initial_regularity_weight = regularity_weight
    
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
        progress = min(1.0, (epoch - 1) / max(1, total_epochs))
        
        # Base schedule: Willmore gradually increases, regularity gradually decreases
        base_willmore = self.initial_willmore_weight + (0.5 - self.initial_willmore_weight) * progress
        base_regularity = self.initial_regularity_weight * (1.0 - 0.5 * progress)
        
        # Adaptive adjustment based on regularity health
        if adaptive_config and adaptive_config.get('enabled', False) and regularity_value is not None:
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
        # Compute individual losses
        willmore = self.willmore_loss(model, uv)
        regularity = self.regularity_loss(model, uv)
        
        # Compute volume constraint if enabled (for genus 0)
        volume_loss = torch.tensor(0.0, device=uv.device)
        current_volume = 0.0
        if self.use_volume_constraint and self.volume_constraint is not None:
            volume_loss, current_volume = self.volume_constraint(model, uv)
            current_volume = current_volume.item()
        
        # Normalize weights so they sum to 1.0
        weight_sum = self.willmore_weight + self.regularity_weight
        
        # Weighted combination with normalized weights
        total_loss = (
            (self.willmore_weight / weight_sum) * willmore +
            (self.regularity_weight / weight_sum) * regularity
        )
        
        # Add volume constraint (not normalized, applied separately)
        if self.use_volume_constraint:
            total_loss = total_loss + volume_loss
        
        result = {
            'total': total_loss,
            'willmore': willmore.item(),
            'regularity': regularity.item()
        }
        
        if self.use_volume_constraint:
            result['volume_loss'] = volume_loss.item()
            result['current_volume'] = current_volume
        
        return result


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
    
    # Get volume constraint settings
    volume_config = loss_config.get("volume_constraint", {})
    volume_constraint_enabled = volume_config.get("enabled", genus == 0)  # Default enabled for genus 0
    volume_constraint_min = volume_config.get("min_volume", 1.0)
    volume_constraint_weight = volume_config.get("weight", 10.0)
    
    return CombinedEmbeddingLoss(
        willmore_weight=loss_config.get("willmore_weight", 1.0),
        regularity_weight=loss_config.get("regularity_weight", 0.1),
        target_area=loss_config.get("target_area", None),
        target_volume=loss_config.get("target_volume", None),
        epsilon=loss_config.get("epsilon", 1e-8),
        regularity_area_element_weight=loss_config.get("regularity_area_element_weight", 1.0),
        regularity_orientation_weight=loss_config.get("regularity_orientation_weight", 1.0),
        regularity_metric_positivity_weight=loss_config.get("regularity_metric_positivity_weight", 1.0),
        regularity_smoothness_weight=loss_config.get("regularity_smoothness_weight", 1.0),
        genus=genus,
        domain=domain,
        volume_constraint_enabled=volume_constraint_enabled,
        volume_constraint_min=volume_constraint_min,
        volume_constraint_weight=volume_constraint_weight
    )


# ============================================================================
# Additional Loss Functions (Currently Disabled)
# ============================================================================


class VolumeConstraintLoss(nn.Module):
    """
    Constraint to maintain enclosed volume of the surface.
    Critical for preventing collapse to a point or line.
    """
    
    def __init__(
        self, 
        target_volume: Optional[float] = None,
        epsilon: float = 1e-8
    ):
        """
        Args:
            target_volume: Target enclosed volume (None for adaptive bounds)
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.target_volume = target_volume
        self.epsilon = epsilon
    
    def forward(self, model: nn.Module, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute volume constraint loss.
        
        For a torus, we approximate the enclosed volume using the divergence theorem.
        Volume = (1/3) ∫∫ (x * n_x + y * n_y + z * n_z) dA
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Volume constraint loss (scalar)
        """
        uv = uv.requires_grad_(True)
        
        # Get embedding and derivatives
        xyz = model.forward(uv)
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv)
        
        # Compute unit normal
        normal_unnorm = torch.cross(phi_u, phi_v, dim=1)
        normal_norm = torch.norm(normal_unnorm, dim=1, keepdim=True) + self.epsilon
        normal = normal_unnorm / normal_norm
        
        # Compute area element
        area_element = torch.sqrt(torch.abs(E * G - F * F) + self.epsilon)
        
        # Volume integrand: (1/3) * (x*n_x + y*n_y + z*n_z) * dA
        position_normal_dot = torch.sum(xyz * normal, dim=1)
        volume_integrand = (1.0 / 3.0) * position_normal_dot * area_element
        
        # Monte Carlo integration over [0,2π]×[0,2π]
        domain_area = (2 * 3.14159265359) ** 2
        enclosed_volume = torch.abs(domain_area * torch.mean(volume_integrand))
        
        if self.target_volume is not None:
            # Penalize deviation from target volume
            loss = ((enclosed_volume - self.target_volume) / self.target_volume) ** 2
        else:
            # For torus (R, r): volume ≈ 2π²Rr²
            # Starting (R=3, r=0.5): volume ≈ 7.40
            # Clifford (R=√2, r=1): volume ≈ 27.87
            # Prevent collapse with strong lower bound
            min_volume = 5.0  # Must stay above this
            max_volume = 100.0  # Reasonable upper bound
            
            # Very strong penalty for volume collapse
            collapse_penalty = torch.nn.functional.relu(min_volume - enclosed_volume) ** 2
            explosion_penalty = torch.nn.functional.relu(enclosed_volume - max_volume) ** 2
            
            loss = 100.0 * collapse_penalty + explosion_penalty  # Extremely strong collapse prevention
        
        return loss


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
