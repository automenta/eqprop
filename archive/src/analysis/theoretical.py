"""Theoretical guarantee validators for EqProp models."""

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TheoreticalValidation:
    """Results of theoretical guarantee validation."""
    
    # Energy descent (for symmetric models)
    energy_descent_valid: bool
    energy_violations: List[int]    # Indices where E increased
    max_energy_increase: float
    
    # Contraction mapping
    contraction_valid: bool
    lipschitz_constant: float
    contraction_margin: float       # 1 - L (positive = contractive)
    
    # Gradient equivalence
    gradient_equivalence: float     # Cosine similarity
    gradient_equivalence_valid: bool
    
    # Fixed point quality
    fixed_point_error: float
    fixed_point_valid: bool
    
    # Spectral analysis
    spectral_radius: float
    spectral_valid: bool            # ρ(J) < 1
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "Theoretical Validation Summary",
            "=" * 50,
            f"Energy Descent:     {'✓' if self.energy_descent_valid else '✗'} "
            f"(violations: {len(self.energy_violations)})",
            f"Contraction (L<1):  {'✓' if self.contraction_valid else '✗'} "
            f"(L={self.lipschitz_constant:.4f})",
            f"Gradient Equiv:     {'✓' if self.gradient_equivalence_valid else '✗'} "
            f"(cos={self.gradient_equivalence:.4f})",
            f"Fixed Point:        {'✓' if self.fixed_point_valid else '✗'} "
            f"(error={self.fixed_point_error:.6f})",
            f"Spectral (ρ<1):     {'✓' if self.spectral_valid else '✗'} "
            f"(ρ={self.spectral_radius:.4f})",
            "=" * 50
        ]
        return "\n".join(lines)


class TheoreticalValidator:
    """Validates theoretical guarantees for EqProp models."""
    
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        if hasattr(model, 'to'):
            self.model.to(device)
    
    def validate_all(self, x: Tensor, y: Tensor, 
                     trajectory_energies: List[float] = None) -> TheoreticalValidation:
        """Run all theoretical validations.
        
        Args:
            x: Input tensor
            y: Target labels (for gradient equivalence)
            trajectory_energies: Pre-computed energy trajectory
            
        Returns:
            TheoreticalValidation with all results
        """
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Get equilibrium state
        with torch.no_grad():
            output = self.model(x, steps=50)
        h_star = self._get_hidden_state(x)
        
        # 1. Energy descent
        if trajectory_energies:
            energy_valid, violations, max_inc = self._check_energy_descent(trajectory_energies)
        else:
            energy_valid, violations, max_inc = True, [], 0.0
        
        # 2. Contraction mapping
        contraction_valid, lipschitz = self._check_contraction(x)
        
        # 3. Gradient equivalence
        grad_equiv = self._check_gradient_equivalence(x, y)
        
        # 4. Fixed point
        fp_error = self._check_fixed_point(h_star, x)
        
        # 5. Spectral radius
        spectral = self._check_spectral_radius(h_star, x)
        
        return TheoreticalValidation(
            energy_descent_valid=energy_valid,
            energy_violations=violations,
            max_energy_increase=max_inc,
            contraction_valid=contraction_valid,
            lipschitz_constant=lipschitz,
            contraction_margin=1.0 - lipschitz,
            gradient_equivalence=grad_equiv,
            gradient_equivalence_valid=grad_equiv > 0.9,
            fixed_point_error=fp_error,
            fixed_point_valid=fp_error < 1e-3,
            spectral_radius=spectral,
            spectral_valid=spectral < 1.0
        )
    
    def _get_hidden_state(self, x: Tensor, steps: int = 50) -> Tensor:
        """Get equilibrium hidden state."""
        batch_size = x.size(0)
        
        if hasattr(self.model, 'hidden_dim'):
            h = torch.zeros(batch_size, self.model.hidden_dim, device=self.device)
        else:
            h = torch.zeros_like(x)
        
        # Handle ToroidalMLP stacked state
        if hasattr(self.model, 'buffer_size'):
            zeros = torch.zeros(batch_size, self.model.buffer_size, 
                               self.model.hidden_dim, device=self.device)
            h = torch.cat([h.unsqueeze(1), zeros], dim=1)
        
        with torch.no_grad():
            for _ in range(steps):
                h, _ = self.model.forward_step(h, x, None)
        
        if h.dim() == 3:
            return h[:, 0]
        return h
    
    def _check_energy_descent(self, energies: List[float]) -> Tuple[bool, List[int], float]:
        """Check if energy decreases monotonically.
        
        Returns:
            (is_valid, violation_indices, max_increase)
        """
        violations = []
        max_increase = 0.0
        
        for i in range(1, len(energies)):
            delta = energies[i] - energies[i-1]
            if delta > 1e-8:  # Tolerance for numerical error
                violations.append(i)
                max_increase = max(max_increase, delta)
        
        return len(violations) == 0, violations, max_increase
    
    def _check_contraction(self, x: Tensor, n_samples: int = 50) -> Tuple[bool, float]:
        """Check contraction mapping property: ||f(h) - f(h')|| ≤ L||h - h'||.
        
        Returns:
            (is_contractive, lipschitz_constant)
        """
        batch_size = x.size(0)
        hidden_dim = self.model.hidden_dim if hasattr(self.model, 'hidden_dim') else x.size(-1)
        
        max_ratio = 0.0
        
        with torch.no_grad():
            for _ in range(n_samples):
                h1 = torch.randn(batch_size, hidden_dim, device=self.device)
                h2 = h1 + torch.randn_like(h1) * 0.1
                
                # Handle stacked state
                if hasattr(self.model, 'buffer_size'):
                    zeros = torch.zeros(batch_size, self.model.buffer_size, 
                                       hidden_dim, device=self.device)
                    h1 = torch.cat([h1.unsqueeze(1), zeros], dim=1)
                    h2 = torch.cat([h2.unsqueeze(1), zeros], dim=1)
                
                f_h1, _ = self.model.forward_step(h1, x, None)
                f_h2, _ = self.model.forward_step(h2, x, None)
                
                # Extract current state for stacked
                if f_h1.dim() == 3:
                    f_h1, f_h2 = f_h1[:, 0], f_h2[:, 0]
                    h1, h2 = h1[:, 0], h2[:, 0]
                
                dist_h = torch.norm(h2 - h1, dim=-1)
                dist_f = torch.norm(f_h2 - f_h1, dim=-1)
                
                ratio = (dist_f / (dist_h + 1e-8)).max().item()
                max_ratio = max(max_ratio, ratio)
        
        return max_ratio < 1.0, max_ratio
    
    def _check_gradient_equivalence(self, x: Tensor, y: Tensor, beta: float = 0.1) -> float:
        """Check cosine similarity between EqProp and BP gradients.
        
        Returns:
            Cosine similarity (higher = better, target > 0.99)
        """
        from src.training import EquilibriumSolver
        
        # Get BP gradients
        self.model.zero_grad()
        output_bp = self.model(x, steps=50)
        loss_bp = F.cross_entropy(output_bp, y)
        loss_bp.backward()
        
        bp_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                bp_grads[name] = param.grad.clone()
        
        # Get EqProp gradients
        self.model.zero_grad()
        solver = EquilibriumSolver(epsilon=1e-5, max_steps=100)
        
        h_free, _ = solver.solve(self.model, x)
        h_free = h_free.detach()
        
        # Compute nudging gradient
        h_free_var = h_free.clone().requires_grad_(True)
        y_hat = self.model.Head(h_free_var)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        dL_dh = h_free_var.grad.detach()
        
        # Nudged phase
        h_nudged, _ = solver.solve(
            self.model, x, h_init=h_free, nudging=True, 
            target_grads=dL_dh, beta=beta
        )
        h_nudged = h_nudged.detach()
        
        # Compute contrastive gradients
        E_free = self.model.energy(h_free, x)
        E_nudged = self.model.energy(h_nudged, x)
        surrogate_loss = (E_nudged - E_free) / beta
        surrogate_loss.backward()
        
        # Compare gradients
        similarities = []
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in bp_grads:
                bp_g = bp_grads[name].flatten()
                eq_g = param.grad.flatten()
                cos_sim = F.cosine_similarity(bp_g.unsqueeze(0), eq_g.unsqueeze(0)).item()
                similarities.append(cos_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _check_fixed_point(self, h_star: Tensor, x: Tensor) -> float:
        """Check ||h* - f(h*, x)||."""
        with torch.no_grad():
            # Handle stacked state
            if hasattr(self.model, 'buffer_size'):
                zeros = torch.zeros(h_star.size(0), self.model.buffer_size, 
                                   self.model.hidden_dim, device=self.device)
                h_stack = torch.cat([h_star.unsqueeze(1), zeros], dim=1)
                f_h, _ = self.model.forward_step(h_stack, x, None)
                f_h = f_h[:, 0]
            else:
                f_h, _ = self.model.forward_step(h_star, x, None)
            
            error = torch.norm(f_h - h_star, dim=-1).mean().item()
        
        return error
    
    def _check_spectral_radius(self, h_star: Tensor, x: Tensor) -> float:
        """Compute spectral radius of Jacobian at equilibrium."""
        from .metrics import compute_spectral_radius
        
        try:
            if hasattr(self.model, 'buffer_size'):
                zeros = torch.zeros(h_star.size(0), self.model.buffer_size, 
                                   self.model.hidden_dim, device=self.device)
                h_stack = torch.cat([h_star.unsqueeze(1), zeros], dim=1)
                return compute_spectral_radius(self.model, h_stack, x)
            else:
                return compute_spectral_radius(self.model, h_star, x)
        except Exception as e:
            return float('nan')
