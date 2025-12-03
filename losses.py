"""
Loss Functions for Embedding-Based Willmore Energy Minimization

This module implements the Willmore functional and related loss functions
for training a neural network to learn an embedding φ: (u,v) → (x,y,z) 
that minimizes the Willmore energy.
"""

import torch
import torch.nn as nn
from typing import Optional


class EmbeddingWillmoreLoss(nn.Module):
    """
    Computes the true Willmore energy functional for embedded surfaces.
    
    The Willmore energy is:
    W = ∫∫ H² dA
    
    where H is the mean curvature and dA is the area element.
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
        
        # Compute mean curvature
        H = model.compute_mean_curvature(E, F, G, L, M, N, self.epsilon)
        
        # Compute area element: dA = sqrt(EG - F²)
        area_element = torch.sqrt(torch.abs(E * G - F * F) + self.epsilon)
        
        # Willmore energy: ∫∫ H² √(EG-F²) du dv over domain [0,2π]×[0,2π]
        # Monte Carlo: integral ≈ (volume/N) * sum = volume * mean
        domain_area = (2 * 3.14159265359) ** 2  # (2π)²
        
        integrand = H * H * area_element
        willmore_energy = domain_area * torch.mean(integrand)
        
        return willmore_energy


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


class EmbeddingSmoothnessLoss(nn.Module):
    """
    Smoothness loss to encourage smooth embeddings.
    Penalizes large gradients in the embedding.
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
        Compute smoothness loss.
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Smoothness loss (scalar)
        """
        uv = uv.requires_grad_(True)
        
        # Compute first fundamental form (which gives us derivatives)
        E, F, G, _, _ = model.compute_first_fundamental_form(uv)
        
        # Penalize large derivatives
        # E = |φ_u|², G = |φ_v|²
        # Want these to be moderate, not too large
        smoothness = torch.mean(E + G)
        
        return smoothness


class TopologyPreservationLoss(nn.Module):
    """
    Loss to help preserve topological properties.
    For a torus, ensures the mapping doesn't create self-intersections or collapse.
    """
    
    def __init__(self, epsilon: float = 1e-8, min_area_element: float = 0.01):
        """
        Args:
            epsilon: Small constant for numerical stability
            min_area_element: Minimum allowed area element (prevents collapse)
        """
        super().__init__()
        self.epsilon = epsilon
        self.min_area_element = min_area_element
    
    def forward(self, model: nn.Module, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute topology preservation loss.
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Topology loss (scalar)
        """
        uv = uv.requires_grad_(True)
        
        # Compute first fundamental form
        E, F, G, phi_u, phi_v = model.compute_first_fundamental_form(uv)
        
        # Check that tangent vectors remain linearly independent
        # det(I) = EG - F² should stay positive and bounded below
        det = E * G - F * F
        
        # Strong penalty for negative or near-zero determinant
        # Use exponential penalty for stronger enforcement
        degeneracy_loss = torch.mean(torch.nn.functional.relu(self.min_area_element - det) ** 2)
        
        # Penalize extremely small area elements (prevents local collapse)
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
        
        return degeneracy_loss + area_element_loss + orientation_loss + metric_positivity


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


class CombinedEmbeddingLoss(nn.Module):
    """
    Combined loss function for embedding-based Willmore minimization.
    """
    
    def __init__(
        self,
        willmore_weight: float = 1.0,
        area_weight: float = 0.1,
        smoothness_weight: float = 0.001,
        topology_weight: float = 0.1,
        volume_weight: float = 1.0,
        symmetry_weight: float = 1.0,
        target_area: Optional[float] = None,
        target_volume: Optional[float] = None,
        epsilon: float = 1e-8
    ):
        """
        Args:
            willmore_weight: Weight for Willmore energy term
            area_weight: Weight for area constraint
            smoothness_weight: Weight for smoothness regularization
            topology_weight: Weight for topology preservation
            volume_weight: Weight for volume constraint (prevents collapse)
            symmetry_weight: Weight for rotational symmetry enforcement
            target_area: Target surface area (None for adaptive)
            target_volume: Target enclosed volume (None for adaptive)
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        self.willmore_weight = willmore_weight
        self.area_weight = area_weight
        self.smoothness_weight = smoothness_weight
        self.topology_weight = topology_weight
        self.volume_weight = volume_weight
        self.symmetry_weight = symmetry_weight
        
        self.willmore_loss = EmbeddingWillmoreLoss(epsilon=epsilon)
        self.area_loss = AreaConstraintLoss(target_area=target_area, epsilon=epsilon)
        self.smoothness_loss = EmbeddingSmoothnessLoss(epsilon=epsilon)
        self.topology_loss = TopologyPreservationLoss(epsilon=epsilon)
        self.volume_loss = VolumeConstraintLoss(target_volume=target_volume, epsilon=epsilon)
        self.symmetry_loss = SymmetryLoss(epsilon=epsilon)
        
        # Store initial weights for reference
        self.initial_willmore_weight = willmore_weight
        self.initial_area_weight = area_weight
        self.initial_smoothness_weight = smoothness_weight
        self.initial_topology_weight = topology_weight
        self.initial_volume_weight = volume_weight
        self.initial_symmetry_weight = symmetry_weight
    
    def update_weights(self, epoch: int, total_epochs: int):
        """
        Progressively adjust loss weights during training.
        Early training: High constraint weights, low Willmore weight
        Late training: Lower constraint weights, moderate Willmore weight
        Smoothness increases throughout to maintain surface quality
        
        Args:
            epoch: Current epoch number (1-indexed)
            total_epochs: Total number of training epochs
        """
        progress = min(1.0, (epoch - 1) / max(1, total_epochs))
        
        # Willmore: gradually increase but keep moderate (0.05 → 0.5, not 1.0)
        # This prevents aggressive optimization that creates roughness
        self.willmore_weight = self.initial_willmore_weight + (0.5 - self.initial_willmore_weight) * progress
        
        # Smoothness: INCREASE over time to counteract roughness (important!)
        self.smoothness_weight = self.initial_smoothness_weight * (1.0 + progress)
        
        # Symmetry: START high, then gradually decrease as shape converges
        # Early training needs strong symmetry enforcement
        self.symmetry_weight = self.initial_symmetry_weight * (1.0 - 0.7 * progress)
        
        # Constraints: gradually decrease (but not too much)
        self.topology_weight = self.initial_topology_weight * (1.0 - 0.5 * progress)
        self.volume_weight = self.initial_volume_weight * (1.0 - 0.5 * progress)
        self.area_weight = self.initial_area_weight * (1.0 - 0.3 * progress)
    
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
        area = self.area_loss(model, uv)
        smoothness = self.smoothness_loss(model, uv)
        topology = self.topology_loss(model, uv)
        volume = self.volume_loss(model, uv)
        symmetry = self.symmetry_loss(model, uv)
        
        # Normalize weights so they sum to 1.0
        weight_sum = (
            self.willmore_weight + 
            self.area_weight + 
            self.smoothness_weight + 
            self.topology_weight + 
            self.volume_weight +
            self.symmetry_weight
        )
        
        # Weighted combination with normalized weights
        total_loss = (
            (self.willmore_weight / weight_sum) * willmore +
            (self.area_weight / weight_sum) * area +
            (self.smoothness_weight / weight_sum) * smoothness +
            (self.topology_weight / weight_sum) * topology +
            (self.volume_weight / weight_sum) * volume +
            (self.symmetry_weight / weight_sum) * symmetry
        )
        
        return {
            'total': total_loss,
            'willmore': willmore.item(),
            'area': area.item(),
            'smoothness': smoothness.item(),
            'topology': topology.item(),
            'volume': volume.item(),
            'symmetry': symmetry.item()
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
    
    return CombinedEmbeddingLoss(
        willmore_weight=loss_config.get("willmore_weight", 1.0),
        area_weight=loss_config.get("area_weight", 0.1),
        smoothness_weight=loss_config.get("smoothness_weight", 0.001),
        topology_weight=loss_config.get("topology_weight", 0.1),
        volume_weight=loss_config.get("volume_weight", 1.0),
        symmetry_weight=loss_config.get("symmetry_weight", 1.0),
        target_area=loss_config.get("target_area", None),
        target_volume=loss_config.get("target_volume", None),
        epsilon=loss_config.get("epsilon", 1e-8)
    )
