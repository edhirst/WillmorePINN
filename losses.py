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
        
        # DEBUG_DELETE: Print intermediate values and check for outliers
        integrand = H * H * area_element
        if torch.rand(1).item() < 0.02:  # Print occasionally
            mean_integrand = torch.mean(integrand).item()
            max_integrand = torch.max(integrand).item()
            min_H = H.min().item()
            max_H = H.max().item()
            print(f"DEBUG_DELETE: mean(H²*dA)={mean_integrand:.2f}, max(H²*dA)={max_integrand:.2f}")
            print(f"DEBUG_DELETE: H range=[{min_H:.2f}, {max_H:.2f}], area_element mean={area_element.mean().item():.2f}")
            if mean_integrand > 10000:
                print(f"DEBUG_DELETE: *** HIGH WILLMORE BATCH DETECTED! ***")
                outliers = integrand > 1000
                print(f"DEBUG_DELETE: {outliers.sum().item()} points with H²*dA > 1000")
        
        willmore_energy = domain_area * torch.mean(integrand)
        
        return willmore_energy


class MetricRegularizationLoss(nn.Module):
    """
    Regularization loss to encourage well-behaved metric properties.
    Penalizes extreme distortion in the first fundamental form.
    """
    
    def __init__(self, target_scale: float = 1.0, epsilon: float = 1e-8):
        """
        Args:
            target_scale: Target scale for metric components
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.target_scale = target_scale
        self.epsilon = epsilon
    
    def forward(self, model: nn.Module, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute metric regularization loss.
        
        Args:
            model: EmbeddingNetwork model
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Regularization loss (scalar)
        """
        uv = uv.requires_grad_(True)
        
        # Compute first fundamental form
        E, F, G, _, _ = model.compute_first_fundamental_form(uv)
        
        # Penalize deviation from target scale
        E_loss = torch.mean((E - self.target_scale) ** 2)
        G_loss = torch.mean((G - self.target_scale) ** 2)
        
        # Penalize large off-diagonal terms (encourage orthogonality)
        F_loss = torch.mean(F ** 2)
        
        # Penalize determinant far from target (area preservation)
        det = E * G - F * F
        det_loss = torch.mean((det - self.target_scale ** 2) ** 2)
        
        return E_loss + G_loss + 0.5 * F_loss + 0.1 * det_loss


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
        
        # Total area (approximate integral over domain)
        total_area = torch.sum(area_element)
        
        if self.target_area is not None:
            # Penalize deviation from target area
            loss = torch.abs(total_area - self.target_area) / self.target_area
        else:
            # For torus with R=2, r=1: area = 4π²Rr ≈ 78.96
            # Penalize extreme values
            expected_area = 80.0  # Approximate
            collapse_penalty = torch.nn.functional.relu(expected_area / 10 - total_area)
            explosion_penalty = torch.nn.functional.relu(total_area - expected_area * 10)
            loss = collapse_penalty + explosion_penalty
        
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
    For a torus, ensures the mapping doesn't create self-intersections.
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
        # det(I) = EG - F² should stay positive (no degeneration)
        det = E * G - F * F
        
        # Penalize negative or near-zero determinant
        degeneracy_loss = torch.mean(torch.nn.functional.relu(self.epsilon - det))
        
        # Also check orientation is preserved (normal should point consistently)
        # Cross product magnitude should stay bounded away from zero
        cross_magnitude = torch.norm(torch.cross(phi_u, phi_v, dim=1), dim=1)
        orientation_loss = torch.mean(torch.nn.functional.relu(0.01 - cross_magnitude))
        
        return degeneracy_loss + orientation_loss


class CombinedEmbeddingLoss(nn.Module):
    """
    Combined loss function for embedding-based Willmore minimization.
    """
    
    def __init__(
        self,
        willmore_weight: float = 1.0,
        regularization_weight: float = 0.01,
        area_weight: float = 0.1,
        smoothness_weight: float = 0.001,
        topology_weight: float = 0.1,
        target_area: Optional[float] = None,
        epsilon: float = 1e-8
    ):
        """
        Args:
            willmore_weight: Weight for Willmore energy term
            regularization_weight: Weight for metric regularization
            area_weight: Weight for area constraint
            smoothness_weight: Weight for smoothness regularization
            topology_weight: Weight for topology preservation
            target_area: Target surface area (None for adaptive)
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        self.willmore_weight = willmore_weight
        self.regularization_weight = regularization_weight
        self.area_weight = area_weight
        self.smoothness_weight = smoothness_weight
        self.topology_weight = topology_weight
        
        self.willmore_loss = EmbeddingWillmoreLoss(epsilon=epsilon)
        self.regularization_loss = MetricRegularizationLoss(epsilon=epsilon)
        self.area_loss = AreaConstraintLoss(target_area=target_area, epsilon=epsilon)
        self.smoothness_loss = EmbeddingSmoothnessLoss(epsilon=epsilon)
        self.topology_loss = TopologyPreservationLoss(epsilon=epsilon)
    
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
        regularization = self.regularization_loss(model, uv)
        area = self.area_loss(model, uv)
        smoothness = self.smoothness_loss(model, uv)
        topology = self.topology_loss(model, uv)
        
        # Weighted combination
        total_loss = (
            self.willmore_weight * willmore +
            self.regularization_weight * regularization +
            self.area_weight * area +
            self.smoothness_weight * smoothness +
            self.topology_weight * topology
        )
        
        # DEBUG_DELETE: Check if willmore is a scalar or needs reduction
        if torch.rand(1).item() < 0.01:
            print(f"DEBUG_DELETE: willmore tensor shape={willmore.shape}, value={willmore.item() if willmore.numel() == 1 else 'NOT_SCALAR'}")
        
        return {
            'total': total_loss,
            'willmore': willmore.item(),
            'regularization': regularization.item(),
            'area': area.item(),
            'smoothness': smoothness.item(),
            'topology': topology.item()
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
        regularization_weight=loss_config.get("regularization_weight", 0.01),
        area_weight=loss_config.get("area_weight", 0.1),
        smoothness_weight=loss_config.get("smoothness_weight", 0.001),
        topology_weight=loss_config.get("topology_weight", 0.1),
        target_area=loss_config.get("target_area", None),
        epsilon=loss_config.get("epsilon", 1e-8)
    )
