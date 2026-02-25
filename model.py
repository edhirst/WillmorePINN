"""
Neural Network Model for Learning Surface Embeddings

This module defines the neural network architecture that learns an embedding
φ: (u,v) → (x,y,z) from parameter space to R³. The Willmore energy is then
computed from the first and second fundamental forms of this embedding.

Supported topologies:
- Genus 0 (sphere/ellipsoid): Uses polar coordinates [0, π] × [0, 2π]
- Genus 1 (torus): Uses doubly-periodic coordinates [0, 2π] × [0, 2π]
- Genus 2 (double torus): Uses custom coordinates [0, 2π] × [0, 4π]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict


class PeriodicEmbedding(nn.Module):
    """
    Fourier feature embedding for enforcing periodicity.
    
    For torus (genus 1): Maps (u,v) → (sin(2πnu), cos(2πnu), sin(2πmv), cos(2πmv), ...)
                         Both u and v are periodic in [0, 2π]
    
    For ellipsoid (genus 0): Maps (u,v) → (sin(nu), cos(nu), sin(mv), cos(mv), ...)
                             u is periodic [0, 2π], v is not periodic [0, π]
    
    For double torus (genus 2): Both directions periodic but v has period 4π
    """
    
    def __init__(self, num_frequencies: int = 4, domain: str = "torus", genus: Optional[int] = None):
        """
        Args:
            num_frequencies: Number of frequency components per dimension
            domain: Surface domain type ('torus', 'ellipsoid', 'double_torus')
            genus: If provided, overrides domain (0=ellipsoid, 1=torus, 2=double_torus)
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        
        # Determine domain from genus if provided
        if genus is not None:
            if genus == 0:
                domain = "ellipsoid"
            elif genus == 1:
                domain = "torus"
            elif genus == 2:
                domain = "double_torus"
        
        self.domain = domain.lower()
        # Output dimension: 2 * num_frequencies * input_dim
        self.output_dim = 2 * num_frequencies * 2  # 2 input dims (u,v)
    
    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv: Parameters (batch_size, 2) 
                - For torus: [0, 2π] × [0, 2π]
                - For ellipsoid: [0, 2π] × [0, π]
                - For double_torus: [0, 2π] × [0, 4π]
        
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
                
            elif self.domain == "double_torus":
                # u has period 2π, v has period 4π
                features.append(torch.sin(freq * uv[:, 0:1]))
                features.append(torch.cos(freq * uv[:, 0:1]))
                # Scale v so that period 4π maps to 2π for Fourier basis
                v_scaled = uv[:, 1:2] * 0.5  # Now effectively period 2π
                features.append(torch.sin(freq * v_scaled))
                features.append(torch.cos(freq * v_scaled))
            
            else:
                # Default: treat as doubly periodic
                features.append(torch.sin(freq * uv[:, 0:1]))
                features.append(torch.cos(freq * uv[:, 0:1]))
                features.append(torch.sin(freq * uv[:, 1:2]))
                features.append(torch.cos(freq * uv[:, 1:2]))
        
        return torch.cat(features, dim=1)


class EmbeddingNetwork(nn.Module):
    """
    Neural network that learns an embedding from parameter space (u,v) to R³.
    
    Supports different topologies:
    - Genus 0 (sphere/ellipsoid): φ(u,v) maps [0, 2π] × [0, π] to R³
    - Genus 1 (torus): φ(u,v) maps [0, 2π] × [0, 2π] to R³  
    - Genus 2 (double torus): φ(u,v) maps [0, 2π] × [0, 4π] to R³
    
    The network enforces appropriate boundary conditions and learns to minimize Willmore energy.
    """
    
    def __init__(
        self,
        input_dim: int = 2,  # (u, v) parameters
        output_dim: int = 3,  # (x, y, z) embedding
        hidden_dims: List[int] = [128, 256, 512, 256, 128],
        activation: str = "tanh",
        dropout: float = 0.0,
        use_batch_norm: bool = True,
        use_periodic_embedding: bool = True,
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
            use_batch_norm: Whether to use batch normalization
            use_periodic_embedding: Whether to use Fourier features for periodicity
            num_frequencies: Number of frequency components (if using periodic embedding)
            initialization: Weight initialization method
            domain: Surface domain type ('torus', 'ellipsoid', 'double_torus')
            domain_params: Domain-specific parameters (legacy, prefer topology_params)
            use_residual: Whether to learn residuals from reference embedding
            residual_scale: Scale factor for residuals
            skip_init: Skip initialization (for checkpoint loading)
            supervised_pretraining_config: Config for supervised pretraining
            genus: Surface genus (0=ellipsoid, 1=torus, 2=double_torus)
            topology_params: Topology-specific parameters from config
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_periodic_embedding = use_periodic_embedding
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
        
        # Periodic embedding layer - adapted for the domain type
        if use_periodic_embedding:
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
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
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
        supported_domains = ['torus', 'sphere', 'ellipsoid', 'double_torus']
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
            
        elif self.domain == "double_torus":
            return get_reference_embedding(
                uv, domain="double_torus", genus=self.genus,
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
        if self.use_periodic_embedding:
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
        
        # Don't clamp - let topology and volume constraints prevent pathological cases
        # Clamping hides the real problems instead of preventing them
        return H
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
    
    model = EmbeddingNetwork(
        input_dim=model_config.get("input_dim", 2),
        output_dim=model_config.get("output_dim", 3),
        hidden_dims=model_config.get("hidden_dims", [128, 256, 512, 256, 128]),
        activation=model_config.get("activation", "tanh"),
        dropout=model_config.get("dropout", 0.0),
        use_batch_norm=model_config.get("use_batch_norm", True),
        use_periodic_embedding=model_config.get("use_periodic_embedding", True),
        num_frequencies=model_config.get("num_frequencies", 4),
        initialization=model_config.get("initialization", "xavier_uniform"),
        domain=domain,
        domain_params=sampling_config.get("domain_params", {}),
        use_residual=model_config.get("use_residual", False),  # Default to full embedding
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
