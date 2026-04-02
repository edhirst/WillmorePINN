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
import functools
from typing import Optional
print = functools.partial(print, flush=True)


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
        metric_positivity_weight: float = 1.0,
        min_positivity: float = 0.001,
        smoothness_weight: float = 1.0,
        max_metric_value: float = 10.0,
        mean_area_floor: float = 0.0,
        mean_area_floor_weight: float = 0.0,
        log_barrier_weight: float = 0.0
    ):
        """
        Args:
            epsilon: Small constant for numerical stability
            min_area_element: Minimum allowed area element (prevents collapse)
            area_element_weight: Weight for area element loss term
            metric_positivity_weight: Weight for metric positivity term
            min_positivity: Minimum threshold for E = |φ_u|² and G = |φ_v|²; positivity loss fires below this
            smoothness_weight: Weight for smoothness term
            max_metric_value: Upper threshold for E and G; only excess beyond this is penalised
            mean_area_floor: Threshold for mean area element; loss fires when the batch mean
                √(EG−F²) drops below this value, preventing global torus degeneration (e.g.
                sphere-collapse where the hole closes and W→4π per chart)
            mean_area_floor_weight: Weight for the mean area floor loss term
            log_barrier_weight: Weight for log-barrier −log(√(EG−F²)) on the area element.
                Gradient −1/√(EG−F²) is continuous and non-zero everywhere (no ReLU gate),
                diverging as the area element → 0. Motivated as −½ log det(g), the negative
                log-density of the Riemannian volume form.
        """
        super().__init__()
        self.epsilon = epsilon
        self.min_area_element = min_area_element
        self.max_metric_value = max_metric_value
        self.mean_area_floor = mean_area_floor
        self.mean_area_floor_weight = mean_area_floor_weight
        self.log_barrier_weight = log_barrier_weight
        
        # Store initial weights
        self.area_element_weight = area_element_weight
        self.metric_positivity_weight = metric_positivity_weight
        self.min_positivity = min_positivity
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
        
        # Penalize extreme metric distortion (prevents degeneration)
        # E and G should be positive and not too different (isotropy encouragement)
        if self.metric_positivity_weight > 0:
            metric_positivity = torch.mean(torch.nn.functional.relu(self.min_positivity - E) ** 2) + \
                               torch.mean(torch.nn.functional.relu(self.min_positivity - G) ** 2)
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
        
        # Mean area element floor: fires when the batch mean √(EG−F²) drops below
        # mean_area_floor.  For a torus (R, r) the parameter-space average is R·r, so
        # this detects sphere-collapse (R·r → 0) long before the per-point minimum
        # falls below min_area_element.  Provides the critical barrier against the
        # periodic network finding a cheaper sphere-like solution (W→4π < W_genus1_min).
        if self.mean_area_floor_weight > 0 and self.mean_area_floor > 0:
            if self.area_element_weight <= 0:   # compute area_element if not already done
                det_maf = torch.clamp(E * G - F * F, min=self.epsilon)
                area_element = torch.sqrt(det_maf)
            mean_area_floor_loss = torch.nn.functional.relu(
                self.mean_area_floor - area_element.mean()
            ) ** 2
            total_loss += self.mean_area_floor_weight * mean_area_floor_loss
            weight_sum += self.mean_area_floor_weight

        # Log-barrier on area element: −log(√(EG−F²)) = −½ log det(g)
        # Gradient −1/√(EG−F²) is continuous and non-zero for all finite area elements,
        # diverging as area element → 0. Provides gradient signal before collapse occurs,
        # unlike the ReLU-gated terms which are silent within their safe bands.
        if self.log_barrier_weight > 0:
            det_lb = torch.clamp(E * G - F * F, min=self.epsilon)
            ae_lb = torch.sqrt(det_lb)
            # Clamp to ≥ 0: the barrier only fires for area_element < 1 (collapse
            # direction); for area_element ≥ 1 the surface is already safe so the
            # contribution is zero rather than negative (which would reward expansion).
            log_barrier = torch.clamp(
                -torch.log(torch.clamp(ae_lb, min=self.epsilon)), min=0.0
            ).mean()
            total_loss += self.log_barrier_weight * log_barrier
            weight_sum += self.log_barrier_weight

        # Normalize by weight sum if any components are active
        if weight_sum > 0:
            total_loss = total_loss / weight_sum
        
        return total_loss


# ============================================================================
# GENUS 2 MULTI-CHART LOSSES
# ============================================================================


class SeamGluingLoss(nn.Module):
    """
    Gluing loss enforcing C°/C¹/C² smoothness across the seam between the
    two torus charts of the genus-2 parametrisation.

    The gluing map is the radial reflection

        g(r, θ) = (2δ − r, θ)

    which maps the collar {δ(1−w) ≤ r ≤ δ(1+w)} bijectively to itself.
    The Jacobian Dg in Cartesian (u,v) coordinates acts as follows
    for all r in the collar (verified by direct computation):

        Dg: e_r ↦ −e_r  (exact for all r)
        Dg: e_θ ↦ (2δ−r)/r · e_θ  (equals +e_θ only at r = δ)

    Both T₁ and T₂ have their disk centres at the outer equatorial point
    (u=0, v=0), and both reference embeddings have the same tangent-plane
    orientation at that point.

    Sample points:
        p₁(r, θ) = (u₀_T1 + r cos θ,       v₀_T1 + r sin θ)        [mod 2π]
        p₂(r, θ) = (u₀_T2 + (2δ−r) cos θ,  v₀_T2 + (2δ−r) sin θ)  [mod 2π]

    with r ∼ Uniform[δ(1−w), δ(1+w)], θ ∼ Uniform[0, 2π).

    Matching conditions derived from the Dg sign pattern:
        C°:            φ₁(p₁) = φ₂(p₂)
        C¹ radial:     ∂ᵣφ₁ + ∂ᵣφ₂ = 0                    (Dg maps e_r → −e_r)
        C¹ tangential: r·(e_θ·∇φ₁) − (2δ−r)·(e_θ·∇φ₂) = 0  (Dg maps e_θ → (2δ−r)/r · e_θ)
        C² rr:  e_r⊗e_r:∇²φ₁ − e_r⊗e_r:∇²φ₂ = 0
        C² rθ:  r·(e_r⊗e_θ:∇²φ₁) + r₂·(e_r⊗e_θ:∇²φ₂) + (e_θ·∇φ₁ + e_θ·∇φ₂) = 0
        C² θθ: r²·(e_θ⊗e_θ:∇²φ₁) − r₂²·(e_θ⊗e_θ:∇²φ₂) − 2δ·(e_r·∇φ₁) = 0

    The C² rθ/θθ conditions include curvature-of-frame corrections from ∂e_θ/∂θ = −e_r
    and ∂e_r/∂θ = e_θ, derived by differentiating the C⁰ identity twice in (r, θ).
    At r = δ (seam centre) they reduce to the naive Dg⊗Dg forms.

    where e_r = (cos θ, sin θ), e_θ = (−sin θ, cos θ), r₂ = 2δ − r.
    """

    def __init__(
        self,
        num_points: int = 256,
        collar_width: float = 0.5,
        c0_weight: float = 1.0,
        c1_weight: float = 1.0,
        c2_weight: float = 1.0,
    ):
        """
        Args:
            num_points: Number of (r, θ) samples per forward call.
            collar_width: Collar half-width as a fraction of δ; collar is
                r ∈ [δ(1 − collar_width), δ(1 + collar_width)].
            c0_weight: Weight for C° position matching.
            c1_weight: Weight for C¹ matching (radial + tangential combined).
            c2_weight: Weight for C² matching (rr + rθ + θθ combined).
        """
        super().__init__()
        self.num_points = num_points
        self.collar_width = collar_width
        self.c0_weight = c0_weight
        self.c1_weight = c1_weight
        self.c2_weight = c2_weight

    @staticmethod
    def _jacobian_and_hessian(model_fn, uv: torch.Tensor, need_hessian: bool = False):
        """Compute the Jacobian, and optionally the Hessian, of model_fn at uv.

        Args:
            model_fn: maps (N, 2) → (N, 3).
            uv: (N, 2) parameter coordinates (detached; re-attached internally).
            need_hessian: if True, also return the (N, 3, 2, 2) Hessian.

        Returns:
            xyz: (N, 3)
            J:   (N, 3, 2)      J[:, i, j] = ∂φᵢ/∂uⱼ
            H:   (N, 3, 2, 2) or None   H[:, i, j, k] = ∂²φᵢ/(∂uₖ∂uⱼ)
        """
        uv = uv.detach().requires_grad_(True)
        xyz = model_fn(uv)  # (N, 3)
        J_rows = []
        for i in range(3):
            gi = torch.autograd.grad(
                xyz[:, i], uv, torch.ones_like(xyz[:, i]),
                create_graph=True, retain_graph=True,
            )[0]  # (N, 2)
            J_rows.append(gi)
        J = torch.stack(J_rows, dim=1)  # (N, 3, 2)

        if not need_hessian:
            return xyz, J, None

        H_rows = []
        for i in range(3):
            H_cols = []
            for j in range(2):
                Hijk = torch.autograd.grad(
                    J[:, i, j], uv, torch.ones_like(J[:, i, j]),
                    create_graph=True, retain_graph=True,
                )[0]  # (N, 2)
                H_cols.append(Hijk)
            H_rows.append(torch.stack(H_cols, dim=1))  # (N, 2, 2)
        H = torch.stack(H_rows, dim=1)  # (N, 3, 2, 2)
        return xyz, J, H

    def forward(self, model) -> torch.Tensor:
        """Compute seam gluing loss for a Genus2MultiChartNetwork.

        Args:
            model: Genus2MultiChartNetwork instance.

        Returns:
            Scalar seam gluing loss.
        """
        n = self.num_points
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        delta = model.disk_radius
        r_min = delta * (1.0 - self.collar_width)
        r_max = delta * (1.0 + self.collar_width)

        # Sample r uniformly in [δ(1−w), δ(1+w)], θ uniformly in [0, 2π)
        r = torch.rand(n, device=device, dtype=dtype) * (r_max - r_min) + r_min
        theta = torch.rand(n, device=device, dtype=dtype) * 2.0 * np.pi
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        u0_T1, v0_T1 = model.disk_center_T1
        u0_T2, v0_T2 = model.disk_center_T2

        r2 = 2.0 * delta - r  # reflected radius; also lies in [δ(1−w), δ(1+w)]

        # T₁ sample: p₁ = (u₀ + r cos θ, v₀ + r sin θ)  [mod 2π]
        uv_1 = torch.stack([
            (u0_T1 + r * cos_t) % (2.0 * np.pi),
            (v0_T1 + r * sin_t) % (2.0 * np.pi),
        ], dim=1)

        # T₂ sample via g: p₂ = (u₀ + (2δ−r) cos θ, v₀ + (2δ−r) sin θ)  [mod 2π]
        # Same-angle (MAP A) pairing: r-reflection g(r,θ) = (2δ−r, θ).
        uv_2 = torch.stack([
            (u0_T2 + r2 * cos_t) % (2.0 * np.pi),
            (v0_T2 + r2 * sin_t) % (2.0 * np.pi),
        ], dim=1)

        need_hessian = self.c2_weight > 0
        need_jacobian = self.c1_weight > 0 or need_hessian

        if need_jacobian:
            xyz_1, J_1, H_1 = self._jacobian_and_hessian(
                model.forward_torus1, uv_1, need_hessian)
            xyz_2, J_2, H_2 = self._jacobian_and_hessian(
                model.forward_torus2, uv_2, need_hessian)
        else:
            xyz_1 = model.forward_torus1(uv_1.detach())
            xyz_2 = model.forward_torus2(uv_2.detach())

        # C°: position matching
        total = self.c0_weight * ((xyz_1 - xyz_2) ** 2).sum(dim=1).mean()

        if self.c1_weight > 0:
            # Polar frame vectors: e_r = (cos θ, sin θ), e_th = (−sin θ, cos θ)
            e_r = torch.stack([cos_t, sin_t], dim=1)    # (N, 2)
            e_th = torch.stack([-sin_t, cos_t], dim=1)  # (N, 2)

            # Radial directional derivative ∂ᵣφ = J @ e_r  (N, 3)
            J1_r = torch.einsum('nij,nj->ni', J_1, e_r)
            J2_r = torch.einsum('nij,nj->ni', J_2, e_r)
            # Tangential directional derivative e_θ·∇φ = J @ e_th  (N, 3)
            J1_th = torch.einsum('nij,nj->ni', J_1, e_th)
            J2_th = torch.einsum('nij,nj->ni', J_2, e_th)

            # C¹ radial:     ∂ᵣφ₁ + ∂ᵣφ₂ = 0                    (Dg maps e_r → −e_r for all r)
            c1_r = ((J1_r + J2_r) ** 2).sum(dim=1).mean()
            # C¹ tangential: r·(J₁@e_θ) − r₂·(J₂@e_θ) = 0  (Dg maps e_θ → (2δ−r)/r·e_θ)
            c1_th = ((r[:, None] * J1_th - r2[:, None] * J2_th) ** 2).sum(dim=1).mean()
            total = total + self.c1_weight * (c1_r + c1_th)

        if self.c2_weight > 0:
            if self.c1_weight <= 0:
                e_r = torch.stack([cos_t, sin_t], dim=1)
                e_th = torch.stack([-sin_t, cos_t], dim=1)
                # J projections needed for curvature-of-frame corrections
                J1_r = torch.einsum('nij,nj->ni', J_1, e_r)
                J1_th = torch.einsum('nij,nj->ni', J_1, e_th)
                J2_th = torch.einsum('nij,nj->ni', J_2, e_th)

            # Hessian contracted on polar frame vectors:
            # H_ab[n,i] = Σ_{j,k} H[n,i,j,k] · a[n,j] · b[n,k]
            H1_rr = torch.einsum('nijk,nj,nk->ni', H_1, e_r, e_r)
            H2_rr = torch.einsum('nijk,nj,nk->ni', H_2, e_r, e_r)
            H1_rt = torch.einsum('nijk,nj,nk->ni', H_1, e_r, e_th)
            H2_rt = torch.einsum('nijk,nj,nk->ni', H_2, e_r, e_th)
            H1_tt = torch.einsum('nijk,nj,nk->ni', H_1, e_th, e_th)
            H2_tt = torch.einsum('nijk,nj,nk->ni', H_2, e_th, e_th)

            # C² rr:  H₁_rr − H₂_rr = 0
            c2_rr = ((H1_rr - H2_rr) ** 2).sum(dim=1).mean()
            # C² rθ:  r·H₁_rt + r₂·H₂_rt + (J₁_θ + J₂_θ) = 0
            # (curvature-of-frame term from ∂e_r/∂θ = e_θ entering ∂²/∂r∂θ)
            c2_rt = ((r[:, None] * H1_rt + r2[:, None] * H2_rt + (J1_th + J2_th)) ** 2).sum(dim=1).mean()
            # C² θθ: r²·H₁_tt − r₂²·H₂_tt − 2δ·J₁_r = 0
            # (curvature-of-frame term from ∂e_θ/∂θ = −e_r entering ∂²/∂θ²)
            c2_tt = (((r ** 2)[:, None] * H1_tt - (r2 ** 2)[:, None] * H2_tt - (2.0 * delta) * J1_r) ** 2).sum(dim=1).mean()
            total = total + self.c2_weight * (c2_rr + c2_rt + c2_tt)

        return total

class MultiChartCombinedLoss(nn.Module):
    """
    Combined loss for the genus-2 two-chart architecture.

    Aggregates Willmore + regularity over two charts (T₁, T₂) plus a seam
    gluing loss enforcing C⁰/C¹/C² continuity throughout the collar neighbourhood
    of the gluing seam.
    """

    def __init__(
        self,
        willmore_weight: float = 1.0,
        regularity_weight: float = 5.0,
        gluing_weight: float = 10.0,
        epsilon: float = 1e-6,
        h2_clip: Optional[float] = None,
        regularity_area_element_weight: float = 1.0,
        regularity_metric_positivity_weight: float = 0.5,
        regularity_min_positivity: float = 0.001,
        regularity_smoothness_weight: float = 0.5,
        regularity_max_metric_value: float = 10.0,
        regularity_min_area_element: float = 0.01,
        regularity_mean_area_floor: float = 0.0,
        regularity_mean_area_floor_weight: float = 0.0,
        regularity_log_barrier_weight: float = 0.0,
        max_willmore_weight: float = 1.0,
        gluing_num_points: int = 256,
        gluing_collar_width: float = 0.5,
        gluing_c1_weight: float = 0.5,
        gluing_c2_weight: float = 0.3,
        gluing_c1_delay: int = 0,
        gluing_c2_delay: int = 0,
        disk_radius: float = 0.3,
        junction_radius_weight: float = 0.0,
        junction_min_radius: float = 0.1,
        junction_max_radius: Optional[float] = None,
        annular_regularity_weight: float = 0.0,
        annular_radius_factor: float = 2.5,
        annular_num_points: int = 128,
    ):
        super().__init__()
        self.willmore_weight = willmore_weight
        self.initial_willmore_weight = willmore_weight
        self.regularity_weight = regularity_weight
        self.initial_regularity_weight = regularity_weight
        self.gluing_weight = gluing_weight
        self.initial_gluing_weight = gluing_weight
        self.gluing_c1_delay = gluing_c1_delay
        self.initial_gluing_c1_weight = gluing_c1_weight
        self.gluing_c2_delay = gluing_c2_delay
        self.initial_gluing_c2_weight = gluing_c2_weight
        self.max_willmore_weight = max_willmore_weight
        self.initial_weight_sum = willmore_weight + regularity_weight
        self.junction_radius_weight = junction_radius_weight
        self.junction_min_radius = junction_min_radius
        self.junction_max_radius = junction_max_radius
        self.annular_regularity_weight = annular_regularity_weight
        self.annular_radius_factor = annular_radius_factor
        self.annular_num_points = annular_num_points
        self.disk_radius = disk_radius

        # Each torus chart integrates over [0, 2π]² minus the excluded disk D
        # of parameter-space area πδ².
        punctured_torus_area = (2.0 * np.pi) ** 2 - np.pi * disk_radius ** 2
        self.willmore_T = EmbeddingWillmoreLoss(epsilon=epsilon, domain="torus", genus=1,
                                                 h2_clip=h2_clip)
        self.willmore_T.domain_area = punctured_torus_area

        self.regularity_loss = RegularityLoss(
            epsilon=epsilon,
            min_area_element=regularity_min_area_element,
            area_element_weight=regularity_area_element_weight,
            metric_positivity_weight=regularity_metric_positivity_weight,
            min_positivity=regularity_min_positivity,
            smoothness_weight=regularity_smoothness_weight,
            max_metric_value=regularity_max_metric_value,
            mean_area_floor=regularity_mean_area_floor,
            mean_area_floor_weight=regularity_mean_area_floor_weight,
            log_barrier_weight=regularity_log_barrier_weight,
        )
        self.gluing_loss = SeamGluingLoss(
            num_points=gluing_num_points,
            collar_width=gluing_collar_width,
            c0_weight=1.0,
            c1_weight=gluing_c1_weight,
            c2_weight=gluing_c2_weight,
        )
        self.genus = 2

    def update_weights(self, epoch: int, total_epochs: int,
                       regularity_value: Optional[float] = None,
                       adaptive_config: Optional[dict] = None):
        """Update loss weights with Willmore warmup and adaptive safeguards."""
        adaptive_enabled = adaptive_config and adaptive_config.get('enabled', False)

        # Willmore warmup: ramp from willmore_warmup_start to 1.0 over
        # willmore_warmup_epochs so that gluing + regularity stabilise
        # before high-curvature Willmore gradients kick in.
        in_warmup = False
        if adaptive_config:
            warmup_epochs = adaptive_config.get('willmore_warmup_epochs', 0)
            warmup_start = adaptive_config.get('willmore_warmup_start', 1.0)
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                t = (epoch - 1) / warmup_epochs
                self.willmore_weight = self.initial_willmore_weight * (warmup_start + (1.0 - warmup_start) * t)
                in_warmup = True

        if not adaptive_enabled:
            if not in_warmup:
                self.willmore_weight = self.initial_willmore_weight
            self.regularity_weight = self.initial_regularity_weight
            # C¹/C² gluing delays still apply regardless of adaptive mode.
            if self.gluing_c1_delay > 0:
                if epoch <= self.gluing_c1_delay:
                    self.gluing_loss.c1_weight = 0.0
                else:
                    self.gluing_loss.c1_weight = self.initial_gluing_c1_weight
            if self.gluing_c2_delay > 0:
                if epoch <= self.gluing_c2_delay:
                    self.gluing_loss.c2_weight = 0.0
                else:
                    self.gluing_loss.c2_weight = self.initial_gluing_c2_weight
            return

        progress = min(1.0, (epoch - 1) / max(1, total_epochs))
        base_w = self.initial_willmore_weight + (self.max_willmore_weight - self.initial_willmore_weight) * progress
        base_r = self.initial_regularity_weight * (1.0 - 0.5 * progress)
        if not in_warmup:
            self.willmore_weight = base_w
        if adaptive_enabled and regularity_value is not None:
            threshold = adaptive_config.get('regularity_threshold', 0.5)
            boost = adaptive_config.get('regularity_boost_factor', 2.0)
            if regularity_value > threshold:
                base_r *= boost
        self.regularity_weight = base_r

        # C¹ gluing delay: hold c1_weight at 0 until gluing_c1_delay epochs have passed.
        if self.gluing_c1_delay > 0:
            if epoch <= self.gluing_c1_delay:
                self.gluing_loss.c1_weight = 0.0
            else:
                self.gluing_loss.c1_weight = self.initial_gluing_c1_weight
        # C² gluing delay: hold c2_weight at 0 until gluing_c2_delay epochs have passed.
        if self.gluing_c2_delay > 0:
            if epoch <= self.gluing_c2_delay:
                self.gluing_loss.c2_weight = 0.0
            else:
                self.gluing_loss.c2_weight = self.initial_gluing_c2_weight

        # Gluing annealing: after willmore warmup ends, linearly reduce gluing
        # weight to initial_gluing_weight * gluing_annealing.  1.0 = no annealing.
        gluing_annealing = adaptive_config.get('gluing_annealing', 1.0)
        if gluing_annealing < 1.0 and not in_warmup:
            post_warmup_progress = min(1.0, max(0.0,
                (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)))
            self.gluing_weight = self.initial_gluing_weight * (
                1.0 - (1.0 - gluing_annealing) * post_warmup_progress
            )

    def forward(self, model, uv_T1: torch.Tensor, uv_T2: torch.Tensor) -> dict:
        """
        Compute combined two-chart loss.

        Args:
            model: Genus2MultiChartNetwork
            uv_T1: (N₁, 2) samples on T₁ chart (excluding disk D₁)
            uv_T2: (N₂, 2) samples on T₂ chart (excluding disk D₂)

        Returns:
            Dict with 'total', 'willmore', 'regularity', 'gluing' keys.
        """
        total = torch.tensor(0.0, device=uv_T1.device)

        # --- Willmore on each chart ---
        w_T1, w_T1_val = self.willmore_T(model.torus1, uv_T1)
        w_T2, w_T2_val = self.willmore_T(model.torus2, uv_T2)

        willmore_train = w_T1 + w_T2
        willmore_value = w_T1_val + w_T2_val
        total = total + self.willmore_weight * willmore_train

        # --- Regularity on each chart ---
        r_T1 = self.regularity_loss(model.torus1, uv_T1)
        r_T2 = self.regularity_loss(model.torus2, uv_T2)
        regularity = r_T1 + r_T2
        regularity_value = regularity.detach().item()
        total = total + self.regularity_weight * regularity

        # --- Junction circle radius penalty ---
        # Penalise when the gluing circle on either torus collapses in ℝ³.
        # The bridge being fine does not prevent the tori from shrinking their
        # handle to a point; this loss provides the missing topological barrier.
        junction_r1 = junction_r2 = None
        if self.junction_radius_weight > 0:
            n_j = 64
            s_j = torch.linspace(0, 2 * np.pi, n_j + 1,
                                  device=uv_T1.device, dtype=uv_T1.dtype)[:-1]
            uv_j1 = model.disk_boundary_T1(s_j)
            xyz_j1 = model.forward_torus1(uv_j1)
            ctr1 = xyz_j1.mean(0, keepdim=True).detach()  # detach so gradient acts only on radius
            junction_r1 = ((xyz_j1 - ctr1) ** 2).sum(1).sqrt().mean()

            uv_j2 = model.disk_boundary_T2(s_j)
            xyz_j2 = model.forward_torus2(uv_j2)
            ctr2 = xyz_j2.mean(0, keepdim=True).detach()
            junction_r2 = ((xyz_j2 - ctr2) ** 2).sum(1).sqrt().mean()

            junction_penalty = (
                torch.nn.functional.relu(self.junction_min_radius - junction_r1) ** 2
                + torch.nn.functional.relu(self.junction_min_radius - junction_r2) ** 2
            )
            # Optional upper bound: prevent the junction circle from growing so
            # large that each torus chart degenerates to a sphere-like cap.
            if self.junction_max_radius is not None:
                junction_penalty = junction_penalty + (
                    torch.nn.functional.relu(junction_r1 - self.junction_max_radius) ** 2
                    + torch.nn.functional.relu(junction_r2 - self.junction_max_radius) ** 2
                )
            total = total + self.junction_radius_weight * junction_penalty

        # --- Annular near-disk regularity ---
        # The excluded disk is absent from both the Willmore and regularity
        # integrals, so the network can freely form a thin funnel just outside
        # the gluing circle.  Sampling the annular zone δ ≤ r_param ≤ α·δ and
        # applying regularity there prevents area-element collapse in the
        # transition region, enforcing a wide neck rather than a thin tube.
        if self.annular_regularity_weight > 0:
            device = uv_T1.device
            dtype = uv_T1.dtype
            n_ann = self.annular_num_points
            inner = self.disk_radius
            outer = self.disk_radius * self.annular_radius_factor

            def _sample_annulus(u0, v0, n):
                collected = []
                while sum(c.shape[0] for c in collected) < n:
                    cand = torch.rand(n * 8, 2, device=device, dtype=dtype) * (2 * np.pi)
                    du = torch.abs(cand[:, 0] - u0)
                    du = torch.min(du, 2 * np.pi - du)
                    dv = torch.abs(cand[:, 1] - v0)
                    dv = torch.min(dv, 2 * np.pi - dv)
                    r2 = du * du + dv * dv
                    mask = (r2 >= inner * inner) & (r2 <= outer * outer)
                    collected.append(cand[mask])
                return torch.cat(collected, dim=0)[:n]

            u0_T1, v0_T1 = model.disk_center_T1
            u0_T2, v0_T2 = model.disk_center_T2
            uv_ann_T1 = _sample_annulus(u0_T1, v0_T1, n_ann)
            uv_ann_T2 = _sample_annulus(u0_T2, v0_T2, n_ann)
            ann_reg = (self.regularity_loss(model.torus1, uv_ann_T1)
                       + self.regularity_loss(model.torus2, uv_ann_T2))
            total = total + self.annular_regularity_weight * ann_reg

        # --- Gluing loss (C⁰+C¹+C² boundary matching at chart junctions) ---
        gluing_loss_val = self.gluing_loss(model)
        total = total + self.gluing_weight * gluing_loss_val

        # Normalise
        if self.initial_weight_sum > 0:
            total = total / self.initial_weight_sum

        return {
            'total': total,
            'willmore': willmore_value,
            'regularity': regularity_value,
            'gluing': gluing_loss_val.detach().item(),
            'junction_r1': junction_r1.detach().item() if junction_r1 is not None else None,
            'junction_r2': junction_r2.detach().item() if junction_r2 is not None else None,
        }

    def eval_willmore_batched(
        self, model, uv_T1: torch.Tensor, uv_T2: torch.Tensor,
        chunk_size: int = 1000,
    ) -> float:
        """Evaluate total Willmore energy across both charts in batches.

        Computes only Willmore (no regularity or gluing) so that each chart
        can be processed in small chunks, keeping peak autodiff graph memory
        proportional to chunk_size rather than eval_num_points.

        Args:
            model: Genus2MultiChartNetwork
            uv_T1:  (N₁, 2) eval samples on T₁ chart
            uv_T2:  (N₂, 2) eval samples on T₂ chart
            chunk_size: Number of points per autodiff pass.

        Returns:
            Total Willmore energy (float).
        """
        def _torus_chart(model_fn, uv):
            n = len(uv)
            total = 0.0
            for s in range(0, n, chunk_size):
                e = min(s + chunk_size, n)
                _, v = self.willmore_T(model_fn, uv[s:e])
                total += v * (e - s)
            return total / n

        return _torus_chart(model.torus1, uv_T1) + _torus_chart(model.torus2, uv_T2)


class CombinedEmbeddingLoss(nn.Module):
    """
    Combined loss function for embedding-based Willmore minimization.
    
    Used for genus 0 (sphere/ellipsoid) and genus 1 (torus).
    Genus 2 uses MultiChartCombinedLoss instead.
    """
    
    def __init__(
        self,
        willmore_weight: float = 1.0,
        regularity_weight: float = 0.1,
        target_area: Optional[float] = None,
        epsilon: float = 1e-8,
        regularity_area_element_weight: float = 1.0,
        regularity_metric_positivity_weight: float = 1.0,
        regularity_min_positivity: float = 0.001,
        regularity_smoothness_weight: float = 1.0,
        regularity_max_metric_value: float = 10.0,
        regularity_min_area_element: float = 0.01,
        regularity_mean_area_floor: float = 0.0,
        regularity_mean_area_floor_weight: float = 0.0,
        regularity_log_barrier_weight: float = 0.0,
        genus: Optional[int] = None,
        domain: str = "torus",
        max_willmore_weight: float = 0.5,
        h2_clip: Optional[float] = None,
    ):
        """
        Args:
            willmore_weight: Weight for Willmore energy term
            regularity_weight: Weight for metric regularity preservation
            target_area: Target surface area (None for adaptive)
            epsilon: Small constant for numerical stability
            regularity_area_element_weight: Weight for area element term within regularity loss
            regularity_metric_positivity_weight: Weight for metric positivity term within regularity loss
            regularity_min_positivity: Minimum threshold for E and G in the positivity term
            regularity_smoothness_weight: Weight for smoothness term within regularity loss
            regularity_max_metric_value: Upper threshold for E and G in the smoothness term
            regularity_min_area_element: Minimum allowed area element √(EG−F²); collapse below this is penalised
            regularity_mean_area_floor: Batch-mean area element floor; fires when mean √(EG−F²) < this
            regularity_mean_area_floor_weight: Weight for the mean area floor loss
            genus: Surface genus (0 or 1)
            domain: Surface domain type
            max_willmore_weight: Ceiling for willmore_weight annealing schedule
            h2_clip: Per-point H² ceiling passed to EmbeddingWillmoreLoss (None = no clipping)
        """
        super().__init__()
        
        self.willmore_weight = willmore_weight
        self.regularity_weight = regularity_weight
        self.genus = genus
        self.domain = domain
        
        self.willmore_loss = EmbeddingWillmoreLoss(epsilon=epsilon, domain=domain, genus=genus,
                                                   h2_clip=h2_clip)
        self.regularity_loss = RegularityLoss(
            epsilon=epsilon,
            min_area_element=regularity_min_area_element,
            area_element_weight=regularity_area_element_weight,
            metric_positivity_weight=regularity_metric_positivity_weight,
            min_positivity=regularity_min_positivity,
            smoothness_weight=regularity_smoothness_weight,
            max_metric_value=regularity_max_metric_value,
            mean_area_floor=regularity_mean_area_floor,
            mean_area_floor_weight=regularity_mean_area_floor_weight,
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

        # Willmore warmup: ramp willmore_weight from a small fraction to the full configured
        # value over willmore_warmup_epochs epochs, reducing gradient shocks from
        # high-curvature bridge regions immediately after supervised pretraining.
        in_warmup = False
        if adaptive_config:
            warmup_epochs = adaptive_config.get('willmore_warmup_epochs', 0)
            warmup_start = adaptive_config.get('willmore_warmup_start', 1.0)
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                t = (epoch - 1) / warmup_epochs  # 0 at epoch 1, 1 at epoch warmup_epochs
                self.willmore_weight = self.initial_willmore_weight * (warmup_start + (1.0 - warmup_start) * t)
                in_warmup = True

        if not adaptive_enabled:
            if not in_warmup:
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

        if not in_warmup:
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
        Combined loss function (CombinedEmbeddingLoss for genus 0/1,
        MultiChartCombinedLoss for genus 2)
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
    
    # --- Genus 2: multi-chart loss ---
    if genus == 2:
        dt_params = topology_config.get('double_torus', {})
        return MultiChartCombinedLoss(
            willmore_weight=loss_config.get("willmore_weight", 1.0),
            regularity_weight=loss_config.get("regularity_weight", 5.0),
            gluing_weight=loss_config.get("gluing_weight", 50.0),
            epsilon=loss_config.get("epsilon", 1e-6),
            h2_clip=loss_config.get("h2_clip", None),
            regularity_area_element_weight=loss_config.get("regularity_area_element_weight", 1.0),
            regularity_metric_positivity_weight=loss_config.get("regularity_metric_positivity_weight", 0.5),
            regularity_min_positivity=loss_config.get("regularity_min_positivity", 0.001),
            regularity_smoothness_weight=loss_config.get("regularity_smoothness_weight", 0.5),
            regularity_max_metric_value=loss_config.get("regularity_max_metric_value", 10.0),
            regularity_min_area_element=loss_config.get("regularity_min_area_element", 0.01),
            regularity_mean_area_floor=loss_config.get("regularity_mean_area_floor", 0.0),
            regularity_mean_area_floor_weight=loss_config.get("regularity_mean_area_floor_weight", 0.0),
            regularity_log_barrier_weight=loss_config.get("regularity_log_barrier_weight", 0.0),
            max_willmore_weight=loss_config.get("max_willmore_weight", 1.0),
            gluing_num_points=loss_config.get("gluing_num_points", 256),
            gluing_collar_width=loss_config.get("gluing_collar_width", 0.5),
            gluing_c1_weight=loss_config.get("gluing_c1_weight", 0.5),
            gluing_c2_weight=loss_config.get("gluing_c2_weight", 0.3),
            gluing_c1_delay=loss_config.get("gluing_c1_delay", 0),
            gluing_c2_delay=loss_config.get("gluing_c2_delay", 0),
            disk_radius=float(dt_params.get('disk_radius', 0.3)),
            junction_radius_weight=loss_config.get("junction_radius_weight", 0.0),
            junction_min_radius=loss_config.get("junction_min_radius", 0.1),
            junction_max_radius=loss_config.get("junction_max_radius", None),
            annular_regularity_weight=loss_config.get("annular_regularity_weight", 0.0),
            annular_radius_factor=loss_config.get("annular_radius_factor", 2.5),
            annular_num_points=loss_config.get("annular_num_points", 128),
        )
    
    # --- Genus 0 or 1: single-chart loss ---
    from sampling import get_domain_for_genus
    domain = get_domain_for_genus(genus)
    
    return CombinedEmbeddingLoss(
        willmore_weight=loss_config.get("willmore_weight", 1.0),
        regularity_weight=loss_config.get("regularity_weight", 0.1),
        target_area=loss_config.get("target_area", None),
        epsilon=loss_config.get("epsilon", 1e-8),
        regularity_area_element_weight=loss_config.get("regularity_area_element_weight", 1.0),
        regularity_metric_positivity_weight=loss_config.get("regularity_metric_positivity_weight", 1.0),
        regularity_min_positivity=loss_config.get("regularity_min_positivity", 0.001),
        regularity_smoothness_weight=loss_config.get("regularity_smoothness_weight", 1.0),
        regularity_max_metric_value=loss_config.get("regularity_max_metric_value", 10.0),
        regularity_min_area_element=loss_config.get("regularity_min_area_element", 0.01),
        regularity_mean_area_floor=loss_config.get("regularity_mean_area_floor", 0.0),
        regularity_mean_area_floor_weight=loss_config.get("regularity_mean_area_floor_weight", 0.0),
        regularity_log_barrier_weight=loss_config.get("regularity_log_barrier_weight", 0.0),
        genus=genus,
        domain=domain,
        max_willmore_weight=loss_config.get("max_willmore_weight", 0.5),
        h2_clip=loss_config.get("h2_clip", None),
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
