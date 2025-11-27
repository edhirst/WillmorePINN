"""
Neural Network Model for Learning Surface Embeddings

This module defines the neural network architecture that learns an embedding
φ: (u,v) → (x,y,z) from parameter space to R³. The Willmore energy is then
computed from the first and second fundamental forms of this embedding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class PeriodicEmbedding(nn.Module):
    """
    Fourier feature embedding for enforcing periodicity.
    Maps (u,v) → (sin(2πnu), cos(2πnu), sin(2πmv), cos(2πmv), ...)
    """
    
    def __init__(self, num_frequencies: int = 4):
        """
        Args:
            num_frequencies: Number of frequency components per dimension
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        # Output dimension: 2 * num_frequencies * input_dim
        self.output_dim = 2 * num_frequencies * 2  # 2 input dims (u,v)
    
    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv: Parameters (batch_size, 2) in [0, 2π] × [0, 2π]
        
        Returns:
            Fourier features (batch_size, 2*num_frequencies*2)
        """
        features = []
        
        for freq in range(1, self.num_frequencies + 1):
            # For u coordinate
            features.append(torch.sin(freq * uv[:, 0:1]))
            features.append(torch.cos(freq * uv[:, 0:1]))
            
            # For v coordinate
            features.append(torch.sin(freq * uv[:, 1:2]))
            features.append(torch.cos(freq * uv[:, 1:2]))
        
        return torch.cat(features, dim=1)


class EmbeddingNetwork(nn.Module):
    """
    Neural network that learns an embedding from parameter space (u,v) to R³.
    
    For a torus: φ(u,v) = (x(u,v), y(u,v), z(u,v))
    The network enforces periodicity and learns to minimize Willmore energy.
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
        residual_scale: float = 0.1  # Scale factor for residuals
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
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_periodic_embedding = use_periodic_embedding
        self.domain = domain
        self.domain_params = domain_params or {}
        self.use_residual = use_residual
        self.residual_scale = residual_scale
        
        # Periodic embedding layer
        if use_periodic_embedding:
            self.periodic_layer = PeriodicEmbedding(num_frequencies)
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
        if not use_residual and domain in ['torus', 'sphere']:
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
        
        print("DEBUG_DELETE: Training network to fit reference geometry + derivatives...")
        
        # Create optimizer for initialization
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        
        # Training loop for initialization
        num_init_epochs = 150  # More epochs since we're matching derivatives too
        batch_size = 256  # Smaller batches for derivative computation
        n_samples_per_epoch = 2000
        
        for epoch in range(num_init_epochs):
            epoch_pos_loss = 0.0
            epoch_deriv_loss = 0.0
            num_batches = n_samples_per_epoch // batch_size
            
            for _ in range(num_batches):
                # Sample random points
                uv_batch = torch.rand(batch_size, 2, device=device) * 2 * np.pi
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
                
                # Weight derivatives less in early epochs, more later
                deriv_weight = min(1.0, epoch / 50.0)
                loss = pos_loss + deriv_weight * deriv_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_pos_loss += pos_loss.item()
                epoch_deriv_loss += deriv_loss.item()
            
            avg_pos_loss = epoch_pos_loss / num_batches
            avg_deriv_loss = epoch_deriv_loss / num_batches
            
            # Print progress
            if (epoch + 1) % 30 == 0:
                print(f"DEBUG_DELETE: Init epoch {epoch+1}/{num_init_epochs}, pos_MSE={avg_pos_loss:.6f}, deriv_MSE={avg_deriv_loss:.6f}")
        
        # Final validation including Willmore energy
        print("DEBUG_DELETE: Computing final initialization quality...")
        with torch.no_grad():
            n_val = 500
            uv_val = torch.rand(n_val, 2, device=device) * 2 * np.pi
            xyz_ref_val = self._get_reference_embedding(uv_val)
            xyz_pred_val = self.forward(uv_val)
            val_error = torch.mean((xyz_pred_val - xyz_ref_val) ** 2).item()
            max_error = torch.max(torch.abs(xyz_pred_val - xyz_ref_val)).item()
            print(f"DEBUG_DELETE: Position - MSE={val_error:.6f}, max_error={max_error:.6f}")
    
    def _get_reference_embedding(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Compute reference embedding based on domain type.
        
        Args:
            uv: Parameter coordinates (batch_size, 2)
        
        Returns:
            Reference embedding (batch_size, 3)
        """
        u, v = uv[:, 0], uv[:, 1]
        
        if self.domain == "torus":
            # Torus: x = (R + r*cos(v))*cos(u), y = (R + r*cos(v))*sin(u), z = r*sin(v)
            R = self.domain_params.get('major_radius', 2.0)
            r = self.domain_params.get('minor_radius', 1.0)
            x = (R + r * torch.cos(v)) * torch.cos(u)
            y = (R + r * torch.cos(v)) * torch.sin(u)
            z = r * torch.sin(v)
            
        elif self.domain == "sphere":
            # Sphere: x = R*sin(v)*cos(u), y = R*sin(v)*sin(u), z = R*cos(v)
            # v ∈ [0, π] for sphere
            R = self.domain_params.get('radius', 1.0)
            x = R * torch.sin(v) * torch.cos(u)
            y = R * torch.sin(v) * torch.sin(u)
            z = R * torch.cos(v)
            
        else:
            # Default: unit sphere
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
        numerator = E * N - 2 * F * M + G * L
        denominator = 2 * (E * G - F * F) + epsilon
        H = numerator / denominator
        
        # DEBUG_DELETE: Check for extreme values before clamping
        if torch.rand(1).item() < 0.02:
            print(f"DEBUG_DELETE: H raw - min={H.min().item():.2f}, max={H.max().item():.2f}, mean={H.mean().item():.2f}")
            print(f"DEBUG_DELETE: denominator - min={denominator.min().item():.6f}, max={denominator.max().item():.6f}")
            n_extreme = (torch.abs(H) > 100).sum().item()
            if n_extreme > 0:
                print(f"DEBUG_DELETE: WARNING! {n_extreme} points with |H| > 100")
        
        # Clamp H to prevent numerical instabilities from dominating
        # For reference: torus R=4, r=0.5 has max|H| ~ 10-20, Clifford has max|H| ~ 5
        H_clamped = torch.clamp(H, min=-100, max=100)
        
        # DEBUG_DELETE: Report if clamping occurred
        if torch.any(H != H_clamped) and torch.rand(1).item() < 0.05:
            n_clamped = (H != H_clamped).sum().item()
            print(f"DEBUG_DELETE: Clamped {n_clamped} H values")
        
        return H_clamped
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_embedding_model(config: dict, device: torch.device) -> nn.Module:
    """
    Factory function to create an embedding model from configuration.
    
    Args:
        config: Configuration dictionary with model parameters
        device: Device to place the model on
    
    Returns:
        Initialized embedding model
    """
    model_config = config.get("model", {})
    sampling_config = config.get("sampling", {})
    
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
        domain=sampling_config.get("domain", "torus"),
        domain_params=sampling_config.get("domain_params", {}),
        use_residual=model_config.get("use_residual", False),  # Default to full embedding
        residual_scale=model_config.get("residual_scale", 0.1)
    )
    
    model = model.to(device)
    
    print(f"Embedding model created with {model.count_parameters()} trainable parameters")
    
    return model
