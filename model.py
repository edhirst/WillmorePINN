"""
Neural Network Model for Learning Surface Embeddings

This module defines the neural network architecture that learns an embedding
φ: (u,v) → (x,y,z) from parameter space to R³. The Willmore energy is then
computed from the first and second fundamental forms of this embedding.

Supported topologies:
- Genus 0 (sphere/ellipsoid): Uses polar coordinates [0, π] × [0, 2π]
- Genus 1 (torus): Uses doubly-periodic coordinates [0, 2π] × [0, 2π]
- Genus 2 (double torus): Multi-chart architecture (T₁ + bridge + T₂)
"""

import torch
import torch.nn as nn
import numpy as np
import functools
from typing import List, Optional, Tuple, Dict
print = functools.partial(print, flush=True)


class PeriodicEmbedding(nn.Module):
    """
    Fourier feature embedding for enforcing periodicity.
    
    For torus (genus 1): Maps (u,v) → (sin(2πnu), cos(2πnu), sin(2πmv), cos(2πmv), ...)
                         Both u and v are periodic in [0, 2π]
    
    For ellipsoid (genus 0): Maps (u,v) → (sin(nu), cos(nu), sin(mv), cos(mv), ...)
                             u is periodic [0, 2π], v is not periodic [0, π]
    """
    
    def __init__(self, num_frequencies: int = 4, domain: str = "torus", genus: Optional[int] = None):
        """
        Args:
            num_frequencies: Number of frequency components per dimension
            domain: Surface domain type ('torus', 'ellipsoid')
            genus: If provided, overrides domain (0=ellipsoid, 1=torus)
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        
        # Determine domain from genus if provided
        if genus is not None:
            if genus == 0:
                domain = "ellipsoid"
            elif genus == 1:
                domain = "torus"
        
        self.domain = domain.lower()
        # Output dimension: 2 * num_frequencies * input_dim
        self.output_dim = 2 * num_frequencies * 2  # 2 input dims (u,v)
        # Frequency curriculum weights (coarse-to-fine, NeRF-style).
        # Shape (num_frequencies,) — one scalar per frequency band.
        # All ones by default (all bands active).  Call set_active_frequencies()
        # from the training loop to implement a coarse-to-fine schedule:
        # low frequencies encode global topology; higher ones add fine detail.
        self.register_buffer('freq_alphas', torch.ones(num_frequencies))
    
    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv: Parameters (batch_size, 2) 
                - For torus: [0, 2π] × [0, 2π]
                - For ellipsoid: [0, 2π] × [0, π]
        
        Returns:
            Fourier features (batch_size, 2*num_frequencies*2)
        """
        features = []
        
        for freq in range(1, self.num_frequencies + 1):
            if self.domain in ["torus"]:
                # Both u and v are periodic with period 2π
                features.append(torch.sin(freq * uv[:, 0:1]))
                features.append(torch.cos(freq * uv[:, 0:1]))
                features.append(torch.sin(freq * uv[:, 1:2]))
                features.append(torch.cos(freq * uv[:, 1:2]))
                
            elif self.domain in ["ellipsoid", "sphere"]:
                # u is periodic with period 2π
                # v is a polar angle in [0, π]; sin(v) = 0 at both poles.
                # Multiplying u-dependent features by sin(v) ensures the network
                # output is u-independent at v=0 and v=π, enforcing sphere topology
                # (both poles collapse to single points) without any auxiliary loss.
                sin_v = torch.sin(uv[:, 1:2])  # sin(v), shape (batch, 1)
                features.append(sin_v * torch.sin(freq * uv[:, 0:1]))
                features.append(sin_v * torch.cos(freq * uv[:, 0:1]))
                # v-only basis functions: not multiplied by sin_v so poles can
                # take arbitrary (u-independent) values determined by the network
                v_scaled = uv[:, 1:2]  # v ∈ [0, π]
                features.append(torch.sin(freq * v_scaled))
                features.append(torch.cos(freq * v_scaled))
            
            else:
                # Default: treat as doubly periodic
                features.append(torch.sin(freq * uv[:, 0:1]))
                features.append(torch.cos(freq * uv[:, 0:1]))
                features.append(torch.sin(freq * uv[:, 1:2]))
                features.append(torch.cos(freq * uv[:, 1:2]))
        
        result = torch.cat(features, dim=1)  # (batch, 4 * num_frequencies)
        # Apply frequency curriculum: each alpha controls one frequency band
        # (all 4 sin/cos features for that band).  Ones by default = all active.
        alpha = self.freq_alphas.repeat_interleave(4).unsqueeze(0)  # (1, 4*N)
        return result * alpha

    def set_active_frequencies(self, num_active: int) -> None:
        """Activate the lowest num_active frequency bands; zero out the rest.

        Implements a coarse-to-fine frequency curriculum analogous to NeRF's
        positional encoding schedule.  Low Fourier frequencies encode smooth
        global shape; higher ones capture fine surface detail.  Progressive
        activation stabilises early Willmore training for surfaces with large
        curvature variation (e.g. the genus-2 bridge junction).

        Args:
            num_active: Number of bands to activate, clamped to [1, num_frequencies].
        """
        n = max(1, min(int(num_active), self.num_frequencies))
        alphas = torch.zeros(self.num_frequencies,
                             device=self.freq_alphas.device,
                             dtype=self.freq_alphas.dtype)
        alphas[:n] = 1.0
        self.freq_alphas.copy_(alphas)


class SphericalHarmonicEmbedding(nn.Module):
    """
    Real spherical harmonic embedding for genus 0 (sphere/ellipsoid).

    Maps (u, v) → {Y_l^m(θ, φ) : 0 ≤ l ≤ L, -l ≤ m ≤ l} where φ = u ∈ [0, 2π]
    is the azimuthal angle and θ = v ∈ [0, π] is the polar angle.

    Real spherical harmonics form an orthonormal basis for L²(S²):
      Y_l^0  = c_{l,0} · P_l^0(cos θ)
      Y_l^m  = c_{l,m} · P_l^m(cos θ) · cos(m φ)    (m > 0)
      Y_l^-m = c_{l,m} · P_l^m(cos θ) · sin(m φ)    (m > 0, written as Y_l^{-m})

    where c_{l,m} = sqrt((2l+1)/(4π) · (l-m)!/(l+m)!) · sqrt(2) for m > 0,
                    sqrt((2l+1)/(4π))                              for m = 0.

    Pole regularity is automatic: P_l^m(±1) = 0 for m > 0, so all m≠0 features
    vanish at θ=0 and θ=π, making the output φ-independent at the poles.

    P_l^m is computed via the standard Bonnet-type recurrence (Condon-Shortley
    phase included in the sectoral step):
      P_0^0 = 1
      P_l^l = -(2l-1) · sin θ · P_{l-1}^{l-1}
      P_l^{l-1} = (2l-1) · cos θ · P_{l-1}^{l-1}
      P_l^m = [(2l-1) · cos θ · P_{l-1}^m - (l+m-1) · P_{l-2}^m] / (l-m)
    """

    def __init__(self, max_degree: int = 4):
        """
        Args:
            max_degree: Maximum spherical harmonic degree L; produces (L+1)² features.
        """
        super().__init__()
        self.max_degree = max_degree
        self.output_dim = (max_degree + 1) ** 2

        # Precompute normalization constants c_{l,m}, ordered as
        # (l=0,m=0), (l=1,m=-1), (l=1,m=0), (l=1,m=1), (l=2,m=-2), ...
        norms = []
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                abs_m = abs(m)
                # Factorial ratio (l - |m|)! / (l + |m|)! computed iteratively
                fac = 1.0
                for k in range(l - abs_m + 1, l + abs_m + 1):
                    fac /= k
                c = np.sqrt((2 * l + 1) / (4 * np.pi) * fac)
                if m != 0:
                    c *= np.sqrt(2)
                norms.append(c)
        self.register_buffer('norms', torch.tensor(norms, dtype=torch.float32))

    def _assoc_legendre(self, x: torch.Tensor, s: torch.Tensor) -> dict:
        """
        Compute P_l^m(x) for 0 ≤ m ≤ l ≤ L using the recurrence above.

        Args:
            x: cos θ, shape (batch_size,)
            s: sin θ ≥ 0, shape (batch_size,)

        Returns:
            dict mapping (l, m) → tensor of shape (batch_size,)
        """
        P = {(0, 0): torch.ones_like(x)}
        for l in range(1, self.max_degree + 1):
            # Sectoral recurrence: P_l^l
            P[(l, l)] = -(2 * l - 1) * s * P[(l - 1, l - 1)]
            # One-step recurrence: P_l^{l-1}
            P[(l, l - 1)] = (2 * l - 1) * x * P[(l - 1, l - 1)]
            # General recurrence: P_l^m for m ≤ l-2
            for m in range(l - 2, -1, -1):
                P[(l, m)] = (
                    (2 * l - 1) * x * P[(l - 1, m)] - (l + m - 1) * P[(l - 2, m)]
                ) / (l - m)
        return P

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv: Parameter coordinates (batch_size, 2) with
                uv[:,0] = φ ∈ [0, 2π] (azimuthal / u)
                uv[:,1] = θ ∈ [0, π]  (polar / v)

        Returns:
            Real spherical harmonics (batch_size, (L+1)²), orthonormal on S².
        """
        phi = uv[:, 0]          # azimuthal φ
        theta = uv[:, 1]        # polar θ
        x = torch.cos(theta)    # cos θ
        s = torch.sin(theta)    # sin θ ≥ 0 for θ ∈ [0, π]

        P = self._assoc_legendre(x, s)

        features = []
        norm_idx = 0
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                abs_m = abs(m)
                c = self.norms[norm_idx]
                norm_idx += 1
                p = P[(l, abs_m)]
                if m < 0:
                    f = c * p * torch.sin(abs_m * phi)
                elif m == 0:
                    f = c * p
                else:
                    f = c * p * torch.cos(m * phi)
                features.append(f.unsqueeze(1))
        return torch.cat(features, dim=1)


class EmbeddingNetwork(nn.Module):
    """
    Neural network that learns an embedding from parameter space (u,v) to R³.
    
    Supports genus 0 (sphere/ellipsoid) and genus 1 (torus).
    Genus 2 uses Genus2MultiChartNetwork instead.
    """
    
    def __init__(
        self,
        input_dim: int = 2,  # (u, v) parameters
        output_dim: int = 3,  # (x, y, z) embedding
        hidden_dims: List[int] = [128, 256, 512, 256, 128],
        activation: str = "tanh",
        dropout: float = 0.0,
        use_spectral_features: bool = True,
        num_frequencies: int = 4,
        initialization: str = "xavier_uniform",
        domain: str = "torus",  # Reference domain type
        domain_params: Optional[dict] = None,  # Parameters for reference embedding
        use_residual: bool = True,  # Learn residuals from reference (False = full embedding)
        residual_scale: float = 0.1,  # Scale factor for residuals
        skip_init: bool = False,  # Skip reference initialization (for loading checkpoints)
        supervised_pretraining_config: Optional[dict] = None,  # Config for supervised pretraining
        genus: Optional[int] = None,  # Surface genus (0, 1, or 2)
        topology_params: Optional[Dict] = None  # Topology-specific parameters
    ):
        """
        Initialize the embedding network.
        
        Args:
            input_dim: Dimension of parameter space (2 for u,v)
            output_dim: Dimension of embedding space (3 for x,y,z)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            dropout: Dropout probability
            use_spectral_features: Whether to use a spectral input embedding (spherical harmonics
                for genus 0, Fourier features for genus 1/2)
            num_frequencies: Spectral order for the input embedding. For genus 0, the
                maximum spherical harmonic degree L, giving (L+1)² features. For genus 1/2,
                the number of Fourier frequency components per dimension.
            initialization: Weight initialization method
            domain: Surface domain type ('torus', 'ellipsoid')
            domain_params: Domain-specific parameters (legacy, prefer topology_params)
            use_residual: Whether to learn residuals from reference embedding
            residual_scale: Scale factor for residuals
            skip_init: Skip initialization (for checkpoint loading)
            supervised_pretraining_config: Config for supervised pretraining
            genus: Surface genus (0=ellipsoid, 1=torus). Genus 2 uses Genus2MultiChartNetwork.
            topology_params: Topology-specific parameters from config
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_spectral_features = use_spectral_features
        self.domain_params = domain_params or {}
        self.use_residual = use_residual
        self.residual_scale = residual_scale
        self.supervised_pretraining_config = supervised_pretraining_config or {}
        self.topology_params = topology_params or {}
        
        # Handle genus and domain
        self.genus = genus
        if genus is not None:
            # Derive domain from genus
            from sampling import get_domain_for_genus
            self.domain = get_domain_for_genus(genus)
        else:
            self.domain = domain
        
        # Spectral embedding layer - adapted for the domain type.
        # Genus 0: real spherical harmonics Y_l^m (natural orthonormal basis for S²),
        #          with num_frequencies interpreted as the maximum degree L → (L+1)² features.
        # Genus 1: Fourier features, both u and v periodic in [0, 2π].
        if use_spectral_features:
            if self.domain in ["ellipsoid", "sphere"]:
                self.periodic_layer = SphericalHarmonicEmbedding(max_degree=num_frequencies)
            else:
                self.periodic_layer = PeriodicEmbedding(num_frequencies, domain=self.domain, genus=genus)
            effective_input_dim = self.periodic_layer.output_dim
        else:
            self.periodic_layer = None
            effective_input_dim = input_dim
        
        # Build the network layers
        layers = []
        prev_dim = effective_input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer - no activation to allow full range
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Store layers
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights(initialization)
        
        # For full embedding mode, initialize to approximate reference
        # Skip if loading from checkpoint or if pretraining is disabled
        pretrain_enabled = self.supervised_pretraining_config.get('enabled', True)
        supported_domains = ['torus', 'sphere', 'ellipsoid']
        if not skip_init and not use_residual and self.domain in supported_domains and pretrain_enabled:
            self._init_near_reference()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU()
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation.lower()]
    
    def _initialize_weights(self, method: str):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    raise ValueError(f"Unknown initialization method: {method}")
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_near_reference(self):
        """Initialize network to match reference embedding AND its derivatives."""
        device = next(self.parameters()).device
        
        # Get supervised pretraining config
        pretrain_config = self.supervised_pretraining_config
        num_init_epochs = pretrain_config.get('num_epochs', 30)
        learning_rate = pretrain_config.get('learning_rate', 0.01)
        batch_size = pretrain_config.get('batch_size', 256)
        n_samples_per_epoch = pretrain_config.get('num_points_per_epoch', 2000)
        position_weight = pretrain_config.get('position_weight', 1.0)
        derivative_weight_final = pretrain_config.get('derivative_weight', 0.1)
        
        # Create optimizer for initialization
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Import sampling function
        from sampling import sample_parameters
        
        for epoch in range(num_init_epochs):
            epoch_pos_loss = 0.0
            epoch_deriv_loss = 0.0
            num_batches = n_samples_per_epoch // batch_size
            
            for _ in range(num_batches):
                # Sample random points according to domain type
                uv_batch = sample_parameters(
                    batch_size, 
                    domain=self.domain,
                    device=device,
                    dtype=torch.float32,
                    genus=self.genus
                )
                uv_batch.requires_grad_(True)
                
                # Get reference embedding and derivatives
                xyz_ref = self._get_reference_embedding(uv_batch)
                
                # Compute reference derivatives
                phi_u_ref = []
                phi_v_ref = []
                for i in range(3):
                    grad_outputs = torch.zeros_like(xyz_ref)
                    grad_outputs[:, i] = 1.0
                    grads_ref = torch.autograd.grad(
                        outputs=xyz_ref,
                        inputs=uv_batch,
                        grad_outputs=grad_outputs,
                        create_graph=False,
                        retain_graph=True
                    )[0]
                    phi_u_ref.append(grads_ref[:, 0:1])
                    phi_v_ref.append(grads_ref[:, 1:2])
                phi_u_ref = torch.cat(phi_u_ref, dim=1).detach()
                phi_v_ref = torch.cat(phi_v_ref, dim=1).detach()
                
                # Forward pass for predicted embedding
                uv_batch_pred = uv_batch.detach().clone().requires_grad_(True)
                xyz_pred = self.forward(uv_batch_pred)
                
                # Compute predicted derivatives
                phi_u_pred = []
                phi_v_pred = []
                for i in range(3):
                    grad_outputs = torch.zeros_like(xyz_pred)
                    grad_outputs[:, i] = 1.0
                    grads_pred = torch.autograd.grad(
                        outputs=xyz_pred,
                        inputs=uv_batch_pred,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    phi_u_pred.append(grads_pred[:, 0:1])
                    phi_v_pred.append(grads_pred[:, 1:2])
                phi_u_pred = torch.cat(phi_u_pred, dim=1)
                phi_v_pred = torch.cat(phi_v_pred, dim=1)
                
                # Combined loss: position + derivatives
                pos_loss = torch.mean((xyz_pred - xyz_ref.detach()) ** 2)
                deriv_loss = torch.mean((phi_u_pred - phi_u_ref) ** 2) + torch.mean((phi_v_pred - phi_v_ref) ** 2)
                
                # Gradually increase derivative weight over training
                deriv_weight = derivative_weight_final * min(1.0, epoch / 50.0)
                loss = position_weight * pos_loss + deriv_weight * deriv_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_pos_loss += pos_loss.item()
                epoch_deriv_loss += deriv_loss.item()
            
            avg_pos_loss = epoch_pos_loss / num_batches
            avg_deriv_loss = epoch_deriv_loss / num_batches
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Supervised pretraining epoch [{epoch+1}/{num_init_epochs}]: "
                      f"Position loss = {avg_pos_loss:.6f}, Derivative loss = {avg_deriv_loss:.6f}")
        
        print(f"Supervised pretraining complete after {num_init_epochs} epochs")
        
        # Final validation including Willmore energy
        with torch.no_grad():
            n_val = 500
            uv_val = sample_parameters(
                n_val,
                domain=self.domain,
                device=device,
                dtype=torch.float32,
                genus=self.genus
            )
            xyz_ref_val = self._get_reference_embedding(uv_val)
            xyz_pred_val = self.forward(uv_val)
            val_error = torch.mean((xyz_pred_val - xyz_ref_val) ** 2).item()
            max_error = torch.max(torch.abs(xyz_pred_val - xyz_ref_val)).item()
    
    def _get_reference_embedding(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute reference embedding based on domain type and genus.
        
        Args:
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Reference embedding (batch_size, 3)
        """
        from sampling import get_reference_embedding
        
        if self.domain == "torus":
            # Get tau from topology params or domain params (backward compatibility)
            torus_params = self.topology_params.get('torus', {})
            tau_value = torus_params.get('tau', self.domain_params.get('tau', '1j'))
            # Parse tau if it's a string
            if isinstance(tau_value, str):
                tau = complex(tau_value.replace(' ', ''))
            elif isinstance(tau_value, dict):
                tau = complex(tau_value.get('real', 0), tau_value.get('imag', 1))
            else:
                tau = tau_value
            max_height = torus_params.get('max_height', self.domain_params.get('max_height', None))
            return get_reference_embedding(
                uv, domain="torus", tau=tau, max_height=max_height,
                genus=self.genus, topology_params=self.topology_params
            )
            
        elif self.domain in ["sphere", "ellipsoid"]:
            return get_reference_embedding(
                uv, domain="ellipsoid", genus=self.genus, 
                topology_params=self.topology_params
            )
            
        else:
            # Default: unit sphere
            u, v = uv[:, 0], uv[:, 1]
            x = torch.sin(v) * torch.cos(u)
            y = torch.sin(v) * torch.sin(u)
            z = torch.cos(v)
            return torch.stack([x, y, z], dim=1)
    
    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map from parameter space to embedding space.
        
        Two modes:
        1. Residual mode: Learns corrections to reference (smooth, fast convergence)
        2. Full mode: Learns full embedding with periodic constraints (dramatic evolution)
        
        Args:
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Embedding coordinates (batch_size, 3) representing (x, y, z)
        """
        if self.use_spectral_features:
            features = self.periodic_layer(uv)
            network_output = self.network(features)
        else:
            network_output = self.network(uv)
        
        if self.use_residual:
            # Residual mode: small corrections to reference
            xyz_ref = self._get_reference_embedding(uv)
            correction = self.residual_scale * torch.tanh(network_output)
            return xyz_ref + correction
        else:
            # Full embedding mode: network outputs coordinates directly
            # Periodicity enforced by Fourier features
            return network_output
    
    def compute_first_fundamental_form(
        self, 
        uv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the first fundamental form (metric tensor) from the embedding.
        
        The first fundamental form is:
        I = [E  F]    where E = <φ_u, φ_u>
            [F  G]          F = <φ_u, φ_v>
                            G = <φ_v, φ_v>
        
        Args:
            uv: Parameter coordinates (batch_size, 2) with requires_grad=True
        
        Returns:
            E, F, G: Components of the first fundamental form (batch_size,)
        """
        batch_size = uv.shape[0]
        uv = uv.requires_grad_(True)
        
        # Compute embedding
        xyz = self.forward(uv)  # (batch_size, 3)
        
        # Compute partial derivatives using Jacobian
        # φ_u = ∂φ/∂u for each component (x, y, z)
        phi_u = []
        phi_v = []
        
        for i in range(3):  # For x, y, z components
            # Compute ∂φ_i/∂u and ∂φ_i/∂v
            grad_outputs = torch.zeros_like(xyz)
            grad_outputs[:, i] = 1.0
            
            grads = torch.autograd.grad(
                outputs=xyz,
                inputs=uv,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]  # (batch_size, 2)
            
            phi_u.append(grads[:, 0:1])  # ∂φ_i/∂u
            phi_v.append(grads[:, 1:2])  # ∂φ_i/∂v
        
        phi_u = torch.cat(phi_u, dim=1)  # (batch_size, 3)
        phi_v = torch.cat(phi_v, dim=1)  # (batch_size, 3)
        
        # First fundamental form components
        E = torch.sum(phi_u * phi_u, dim=1)  # <φ_u, φ_u>
        F = torch.sum(phi_u * phi_v, dim=1)  # <φ_u, φ_v>
        G = torch.sum(phi_v * phi_v, dim=1)  # <φ_v, φ_v>
        
        return E, F, G, phi_u, phi_v
    
    def compute_second_fundamental_form(
        self,
        uv: torch.Tensor,
        phi_u: torch.Tensor,
        phi_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the second fundamental form from the embedding.
        
        The second fundamental form is:
        II = [L  M]    where L = <φ_uu, n>
             [M  N]          M = <φ_uv, n>
                             N = <φ_vv, n>
        
        And n is the unit normal: n = (φ_u × φ_v) / |φ_u × φ_v|
        
        Args:
            uv: Parameter coordinates (batch_size, 2)
            phi_u: First partial derivative w.r.t. u (batch_size, 3)
            phi_v: First partial derivative w.r.t. v (batch_size, 3)
        
        Returns:
            L, M, N: Components of second fundamental form (batch_size,)
            normal: Unit normal vector n (batch_size, 3)
        """
        batch_size = uv.shape[0]
        
        # Compute unit normal: N = (φ_u × φ_v) / |φ_u × φ_v|
        normal_unnorm = torch.cross(phi_u, phi_v, dim=1)
        normal_norm = torch.norm(normal_unnorm, dim=1, keepdim=True) + 1e-8
        normal = normal_unnorm / normal_norm
        
        # Compute second derivatives
        # For each component of phi_u, compute derivative w.r.t. u and v
        phi_uu = []
        phi_uv = []
        phi_vv = []
        
        for i in range(3):  # For x, y, z components
            # ∂²φ_i/∂u² and ∂²φ_i/∂u∂v from phi_u
            grad_outputs_u = torch.zeros_like(phi_u)
            grad_outputs_u[:, i] = 1.0
            
            grads_u = torch.autograd.grad(
                outputs=phi_u,
                inputs=uv,
                grad_outputs=grad_outputs_u,
                create_graph=True,
                retain_graph=True
            )[0]  # (batch_size, 2)
            
            phi_uu.append(grads_u[:, 0:1])  # ∂²φ_i/∂u²
            phi_uv.append(grads_u[:, 1:2])  # ∂²φ_i/∂u∂v
            
            # ∂²φ_i/∂v² from phi_v
            grad_outputs_v = torch.zeros_like(phi_v)
            grad_outputs_v[:, i] = 1.0
            
            grads_v = torch.autograd.grad(
                outputs=phi_v,
                inputs=uv,
                grad_outputs=grad_outputs_v,
                create_graph=True,
                retain_graph=True
            )[0]  # (batch_size, 2)
            
            phi_vv.append(grads_v[:, 1:2])  # ∂²φ_i/∂v²
        
        phi_uu = torch.cat(phi_uu, dim=1)  # (batch_size, 3)
        phi_uv = torch.cat(phi_uv, dim=1)  # (batch_size, 3)
        phi_vv = torch.cat(phi_vv, dim=1)  # (batch_size, 3)
        
        # Second fundamental form components
        L = torch.sum(phi_uu * normal, dim=1)
        M = torch.sum(phi_uv * normal, dim=1)
        N = torch.sum(phi_vv * normal, dim=1)
        
        return L, M, N, normal
    
    def compute_mean_curvature(
        self,
        E: torch.Tensor,
        F: torch.Tensor,
        G: torch.Tensor,
        L: torch.Tensor,
        M: torch.Tensor,
        N: torch.Tensor,
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute mean curvature from fundamental forms.
        
        Mean curvature: H = (EN - 2FM + GL) / (2(EG - F²))
        
        Args:
            E, F, G: First fundamental form components
            L, M, N: Second fundamental form components
            epsilon: Small constant for numerical stability
        
        Returns:
            Mean curvature H (batch_size,)
        """
        det = torch.clamp(E * G - F * F, min=epsilon)
        numerator = E * N - 2 * F * M + G * L
        denominator = 2 * det
        H = numerator / denominator
        
        # Don't clamp - let topology and regularity constraints prevent pathological cases
        # Clamping hides the real problems instead of preventing them
        return H
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# GENUS 2 MULTI-CHART ARCHITECTURE
# ============================================================================
#
# A genus-2 surface is the connected sum T₁ # T₂: two tori, each with a small
# disk removed, joined by a cylindrical bridge.
#
# Three charts:
#   Chart T₁:     [0, 2π]² with u, v doubly-periodic (Fourier features)
#                  A disk D₁ of radius δ centred at (u₀, v₀) is excluded from
#                  the Willmore integral.  The network still maps D₁ (Fourier
#                  features are smooth everywhere), but D₁ is "phantom" — it
#                  contributes no gradient.  The bridge replaces D₁ geometrically.
#
#   Chart Bridge:  [0, 2π] × [0, 1] with u periodic, t non-periodic.
#                  A tube connecting ∂D₁ on T₁ to ∂D₂ on T₂.
#
#   Chart T₂:     Same as T₁, with its own disk D₂ excluded.
#
# Euler characteristic:  χ(T₁\D₁) + χ(bridge) + χ(T₂\D₂)
#                      = (-1)      + (0)       + (-1)       = -2  →  genus 2  ✓
#
# Boundary matching loss stitches the charts at ∂D₁ and ∂D₂.
# ============================================================================


def _compute_disk_center_T2(tau2: complex) -> Tuple[float, float]:
    """Compute the T₂ disk centre that maps to the closest point of T₂ toward T₁.

    Returns (u₀, v₀) ∈ [0, 2π]² such that the flat-torus embedding at (u₀, v₀)
    has u_major = π and v_twisted = 0, i.e., the outermost point of the T₂ tube
    on the T₁-facing side.  For purely imaginary τ₂ this reduces to (π, 0).

    Derivation: after transform_square_to_parallelogram and get_flat_torus_embedding,
      u_major = (u₀ + v₀·Re(τ)) mod 2π  and  v_twisted = v₀ + Re(τ)/Im(τ)·u_major.
    Setting u_major = π, v_twisted = 0 gives:
      v₀ = −Re(τ)·π / Im(τ)  (mod 2π)
      u₀ =  π − v₀·Re(τ)     (mod 2π)  ← uses the wrapped v₀

    Args:
        tau2: Complex modulus of T₂ with Im(τ₂) > 0.

    Returns:
        (u₀, v₀): disk centre in the square [0, 2π]² parameter domain.
    """
    a = tau2.real
    b = tau2.imag   # Im(τ) > 0 required
    v0 = (-a * np.pi / b) % (2 * np.pi)
    # u₀ = π − v₀·Re(τ), using the already-wrapped v₀.  The naive derivation
    # (π + Re²·π/Im) is only correct when v₀ needs no wrapping; using v₀
    # directly handles all cases uniformly.
    u0 = (np.pi - v0 * a) % (2 * np.pi)
    return (u0, v0)


class BridgeEmbedding(nn.Module):
    """
    Spectral embedding for the bridge (cylinder) chart.

    u ∈ [0, 2π] periodic: standard Fourier basis sin(ku), cos(ku).
    t ∈ [0, 1] non-periodic: sin(kπt), cos(kπt) for k=1..N, plus raw t.

    At t=0: t-features = [0, 1, 0, 1, ...]  (bridge start — T₁ disk boundary)
    At t=1: t-features = [0, -1, 0, 1, ...]  (bridge end — T₂ disk boundary)
    Distinct → bridge does not self-close.

    Output dim: 4·N + 1  (2N u-features + 2N t-features + 1 raw t).
    """

    def __init__(self, num_frequencies: int = 4):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.output_dim = 4 * num_frequencies + 1

    def forward(self, ut: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ut: (batch, 2) with ut[:,0]=u ∈ [0, 2π], ut[:,1]=t ∈ [0, 1]

        Returns:
            Features (batch, 4·N + 1).
        """
        u = ut[:, 0:1]
        t = ut[:, 1:2]
        features = []
        for k in range(1, self.num_frequencies + 1):
            features.append(torch.sin(k * u))
            features.append(torch.cos(k * u))
            features.append(torch.sin(k * np.pi * t))
            features.append(torch.cos(k * np.pi * t))
        features.append(t)  # raw position along bridge
        return torch.cat(features, dim=1)


class Genus2MultiChartNetwork(nn.Module):
    """
    Multi-chart embedding network for genus-2 surfaces.

    Three sub-networks (one per chart):
      T₁:     EmbeddingNetwork with genus=1 (doubly-periodic Fourier features)
      Bridge:  Small MLP with BridgeEmbedding (u-periodic, t-non-periodic)
      T₂:     EmbeddingNetwork with genus=1 (doubly-periodic Fourier features)

    Each torus has a parametric disk D_i excluded from the Willmore integral.
    The bridge connects ∂D₁ → ∂D₂.  Boundary matching is enforced by a loss,
    not by the features.

    Attributes:
        disk_center_T1: (u₀, v₀) centre of the excluded disk on T₁ (parameter space)
        disk_center_T2: (u₀, v₀) centre of the excluded disk on T₂
        disk_radius:    δ  (parameter-space radius of each excluded disk)
    """

    def __init__(
        self,
        hidden_dims: List[int] = [128, 256, 512, 256, 128],
        activation: str = "tanh",
        dropout: float = 0.0,
        num_frequencies: int = 6,
        initialization: str = "xavier_uniform",
        tau1: complex = 1j,
        tau2: complex = 1j,
        disk_center_T1: Tuple[float, float] = (0.0, 0.0),
        disk_center_T2: Tuple[float, float] = (0.0, 0.0),
        disk_radius: float = 0.3,
        torus2_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        skip_init: bool = False,
        supervised_pretraining_config: Optional[dict] = None,
        topology_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.tau1 = tau1
        self.tau2 = tau2
        self.disk_center_T1 = disk_center_T1
        self.disk_center_T2 = disk_center_T2
        self.disk_radius = disk_radius
        self.torus2_offset = torus2_offset
        self.topology_params = topology_params or {}

        pretrain_cfg = supervised_pretraining_config or {}

        # --- Chart T₁ (genus-1 torus) ---
        torus1_topo = {'torus': {'tau': str(tau1)}}
        self.torus1 = EmbeddingNetwork(
            input_dim=2, output_dim=3,
            hidden_dims=hidden_dims,
            activation=activation, dropout=dropout,
            use_spectral_features=True,
            num_frequencies=num_frequencies,
            initialization=initialization,
            use_residual=False,
            skip_init=skip_init,
            supervised_pretraining_config=pretrain_cfg,
            genus=1,
            topology_params=torus1_topo,
        )

        # --- Chart Bridge (cylinder) ---
        bridge_embed = BridgeEmbedding(num_frequencies)
        bridge_input_dim = bridge_embed.output_dim
        bridge_layers = []
        prev = bridge_input_dim
        for hd in hidden_dims:
            bridge_layers.append(nn.Linear(prev, hd))
            bridge_layers.append(self.torus1._get_activation(activation))
            if dropout > 0:
                bridge_layers.append(nn.Dropout(dropout))
            prev = hd
        bridge_layers.append(nn.Linear(prev, 3))
        self.bridge_embed = bridge_embed
        self.bridge_network = nn.Sequential(*bridge_layers)
        # Initialise bridge weights
        for m in self.bridge_network.modules():
            if isinstance(m, nn.Linear):
                if initialization == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif initialization == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # --- Chart T₂ (genus-1 torus) ---
        torus2_topo = {'torus': {'tau': str(tau2)}}
        self.torus2 = EmbeddingNetwork(
            input_dim=2, output_dim=3,
            hidden_dims=hidden_dims,
            activation=activation, dropout=dropout,
            use_spectral_features=True,
            num_frequencies=num_frequencies,
            initialization=initialization,
            use_residual=False,
            skip_init=skip_init,
            supervised_pretraining_config=pretrain_cfg,
            genus=1,
            topology_params=torus2_topo,
        )

        # Offset T₂ in ℝ³ so the two tori start non-overlapping.
        # Without this, both charts produce the same torus surface and the
        # bridge degenerates.  The offset shifts T₂'s final-layer bias so
        # its output is the standard torus embedding + offset.
        if not skip_init and any(abs(x) > 1e-12 for x in torus2_offset):
            with torch.no_grad():
                # Last module in self.torus2.network is nn.Linear(hidden, 3)
                final_layer = list(self.torus2.network.modules())
                for m in reversed(final_layer):
                    if isinstance(m, nn.Linear) and m.out_features == 3:
                        m.bias.add_(torch.tensor(list(torus2_offset), dtype=m.bias.dtype))
                        break

    # ---- forwarding helpers ------------------------------------------------

    def forward_torus1(self, uv: torch.Tensor) -> torch.Tensor:
        """Evaluate T₁ chart.  uv ∈ [0, 2π]²."""
        return self.torus1(uv)

    def forward_bridge(self, ut: torch.Tensor) -> torch.Tensor:
        """Evaluate bridge chart.  u ∈ [0, 2π], t ∈ [0, 1]."""
        return self.bridge_network(self.bridge_embed(ut))

    def forward_torus2(self, uv: torch.Tensor) -> torch.Tensor:
        """Evaluate T₂ chart.  uv ∈ [0, 2π]²."""
        return self.torus2(uv)

    # ---- disk-boundary parametrisation -------------------------------------

    def disk_boundary_T1(self, s: torch.Tensor) -> torch.Tensor:
        """Map angle s ∈ [0, 2π] to parameter coords on ∂D₁ ⊂ T₁.

        Returns (len(s), 2) tensor with periodic wrapping to [0, 2π]².
        """
        u0, v0 = self.disk_center_T1
        delta = self.disk_radius
        u = (u0 + delta * torch.cos(s)) % (2 * np.pi)
        v = (v0 + delta * torch.sin(s)) % (2 * np.pi)
        return torch.stack([u, v], dim=1)

    def disk_boundary_T2(self, s: torch.Tensor) -> torch.Tensor:
        """Map angle s ∈ [0, 2π] to parameter coords on ∂D₂ ⊂ T₂.

        The u-component uses −cos(s) (reversed relative to ∂D₁) so that the
        corresponding boundary circles on T₁ and T₂ are consistently oriented
        in ℝ³ (both counterclockwise viewed from the T₁ side).  Without this
        sign the bridge cross-section degenerates at t = ½.
        """
        u0, v0 = self.disk_center_T2
        delta = self.disk_radius
        u = (u0 - delta * torch.cos(s)) % (2 * np.pi)
        v = (v0 + delta * torch.sin(s)) % (2 * np.pi)
        return torch.stack([u, v], dim=1)

    # ---- reference embeddings per chart ------------------------------------

    def reference_torus1(self, uv: torch.Tensor) -> torch.Tensor:
        """Reference embedding for T₁ (standard torus at position x₁)."""
        return self.torus1._get_reference_embedding(uv)

    def reference_torus2(self, uv: torch.Tensor) -> torch.Tensor:
        """Reference embedding for T₂ (standard torus + offset)."""
        ref = self.torus2._get_reference_embedding(uv)
        if any(abs(x) > 1e-12 for x in self.torus2_offset):
            offset = torch.tensor(list(self.torus2_offset), device=ref.device, dtype=ref.dtype)
            ref = ref + offset
        return ref

    def reference_bridge(self, ut: torch.Tensor) -> torch.Tensor:
        """Reference embedding for the bridge (catenoid connecting ∂D₁ → ∂D₂).

        Linearly interpolates between the T₁ disk-boundary reference
        and the T₂ disk-boundary reference, with a catenoid-like
        radius modulation.  This is deliberately simple — supervised
        pretraining only needs a plausible initialisation target.
        """
        s = ut[:, 0]
        t = ut[:, 1:2]
        # Evaluate reference tori at disk boundaries
        uv_d1 = self.disk_boundary_T1(s)
        uv_d2 = self.disk_boundary_T2(s)
        xyz1 = self.reference_torus1(uv_d1).detach()
        xyz2 = self.reference_torus2(uv_d2).detach()
        # Lerp with cosh waist modulation
        # At t=0: xyz1,  at t=1: xyz2, with slight inward pinch at t=0.5
        center = (xyz1 + xyz2) / 2.0
        pinch = 0.7 + 0.3 * (2.0 * t - 1.0).pow(2)  # 0.7 at t=0.5, 1.0 at endpoints
        return center + pinch * ((1.0 - t) * (xyz1 - center) + t * (xyz2 - center))

    # ---- supervised pretraining --------------------------------------------

    def pretrain(self, config: dict, device: torch.device):
        """
        Supervised pretraining: fit each chart's network to its reference embedding.

        The torus sub-networks use their built-in _init_near_reference.
        The bridge network is pretrained here against reference_bridge.
        """
        pretrain_cfg = config.get('model', {}).get('supervised_pretraining', {})
        if not pretrain_cfg.get('enabled', True):
            return

        print("\n[Multi-chart] Supervised pretraining: T₁ and T₂ handled internally.")
        print("[Multi-chart] Pretraining bridge network...")

        n_epochs = pretrain_cfg.get('num_epochs', 30)
        lr = pretrain_cfg.get('learning_rate', 0.01)
        batch_size = pretrain_cfg.get('batch_size', 256)
        n_per_epoch = pretrain_cfg.get('num_points_per_epoch', 2000)

        opt = torch.optim.Adam(
            list(self.bridge_embed.parameters()) + list(self.bridge_network.parameters()),
            lr=lr,
        )
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = max(1, n_per_epoch // batch_size)
            for _ in range(n_batches):
                u = torch.rand(batch_size, device=device) * 2 * np.pi
                t = torch.rand(batch_size, device=device)
                ut = torch.stack([u, t], dim=1)
                xyz_pred = self.forward_bridge(ut)
                xyz_ref = self.reference_bridge(ut)
                loss = ((xyz_pred - xyz_ref) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"  Bridge pretrain epoch {epoch+1}/{n_epochs}: MSE = {epoch_loss / n_batches:.6f}")

    # ---- utilities ---------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def domain(self):
        """Compatibility property — the multi-chart network has no single domain."""
        return "genus2_multichart"


def create_embedding_model(config: dict, device: torch.device, skip_init: bool = False) -> nn.Module:
    """
    Factory function to create an embedding model from configuration.
    
    Args:
        config: Configuration dictionary with model parameters
        device: Device to place the model on
        skip_init: If True, skip reference initialization (useful when loading checkpoints)
    
    Returns:
        Initialized embedding model
    """
    model_config = config.get("model", {})
    sampling_config = config.get("sampling", {})
    topology_config = config.get("topology", {})
    supervised_pretraining_config = model_config.get("supervised_pretraining", {})
    
    # Get genus from topology config (defaults to 1 for backward compatibility)
    genus = topology_config.get("genus", 1)
    
    # Validate genus
    if genus < 0:
        raise ValueError(f"Genus must be non-negative, got {genus}")
    if genus > 2:
        raise NotImplementedError(f"Genus {genus} is not supported. Only genus 0, 1, 2 are implemented.")
    
    # Determine domain from genus
    from sampling import get_domain_for_genus
    domain = get_domain_for_genus(genus)
    
    # --- Genus 2: multi-chart architecture ---
    if genus == 2:
        dt_params = topology_config.get('double_torus', {})

        def _parse_tau(raw):
            if isinstance(raw, str):
                return complex(raw.replace('i', 'j').replace(' ', ''))
            elif isinstance(raw, dict):
                return complex(raw.get('real', 0), raw.get('imag', 1))
            return complex(raw)

        tau1 = _parse_tau(dt_params.get('tau1', '1j'))
        tau2 = _parse_tau(dt_params.get('tau2', '1j'))
        disk_radius = float(dt_params.get('disk_radius', 0.3))
        disk_center_T1 = tuple(dt_params.get('disk_center_T1', [0.0, 0.0]))
        # disk_center_T2 is always derived from τ₂ so that the excised disk
        # sits on the part of T₂ closest to T₁ for any complex τ₂.
        disk_center_T2 = _compute_disk_center_T2(tau2)
        torus2_offset = tuple(dt_params.get('torus2_offset', [0.0, 0.0, 0.0]))

        model = Genus2MultiChartNetwork(
            hidden_dims=model_config.get("hidden_dims", [128, 256, 512, 256, 128]),
            activation=model_config.get("activation", "tanh"),
            dropout=model_config.get("dropout", 0.0),
            num_frequencies=model_config.get("num_frequencies", 6),
            initialization=model_config.get("initialization", "xavier_uniform"),
            tau1=tau1, tau2=tau2,
            disk_center_T1=disk_center_T1,
            disk_center_T2=disk_center_T2,
            disk_radius=disk_radius,
            torus2_offset=torus2_offset,
            skip_init=skip_init,
            supervised_pretraining_config=supervised_pretraining_config,
            topology_params=topology_config,
        )
        model = model.to(device)

        # Bridge pretraining (T₁, T₂ pretrain internally via _init_near_reference)
        if not skip_init:
            model.pretrain(config, device)

        print(f"Multi-chart model created for genus 2 (T₁ + bridge + T₂)")
        print(f"  τ₁ = {tau1}, τ₂ = {tau2}")
        print(f"  Disk radius δ = {disk_radius}")
        if any(abs(x) > 1e-12 for x in torus2_offset):
            print(f"  T₂ offset = {torus2_offset}")
        print(f"  Parameters: {model.count_parameters()} trainable")
        return model

    # --- Genus 0 or 1: single-chart ---
    model = EmbeddingNetwork(
        input_dim=model_config.get("input_dim", 2),
        output_dim=model_config.get("output_dim", 3),
        hidden_dims=model_config.get("hidden_dims", [128, 256, 512, 256, 128]),
        activation=model_config.get("activation", "tanh"),
        dropout=model_config.get("dropout", 0.0),
        use_spectral_features=model_config.get("use_spectral_features", True),
        num_frequencies=model_config.get("num_frequencies", 4),
        initialization=model_config.get("initialization", "xavier_uniform"),
        domain=domain,
        domain_params=sampling_config.get("domain_params", {}),
        use_residual=model_config.get("use_residual", False),
        residual_scale=model_config.get("residual_scale", 0.1),
        skip_init=skip_init,
        supervised_pretraining_config=supervised_pretraining_config,
        genus=genus,
        topology_params=topology_config
    )
    
    model = model.to(device)
    
    genus_names = {0: "sphere/ellipsoid", 1: "torus", 2: "double torus"}
    print(f"Embedding model created for genus {genus} ({genus_names.get(genus, 'unknown')})")
    print(f"  Domain: {domain}")
    print(f"  Parameters: {model.count_parameters()} trainable")
    
    return model
