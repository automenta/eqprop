"""Per-iteration metrics computation."""

import torch
from torch import Tensor
from typing import List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class IterationMetrics:
    """Metrics computed at a single iteration step."""
    step: int
    h_norm: float                   # ||h_t||
    delta_norm: float               # ||h_t - h_{t-1}||
    energy: float                   # E(h_t)
    energy_delta: float             # E(h_t) - E(h_{t-1})
    jacobian_spectral: Optional[float]  # ρ(J) spectral radius (expensive)
    gradient_cosine: Optional[float]     # Cosine sim with BP gradient


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all iterations."""
    total_steps: int
    converged: bool
    convergence_step: Optional[int]
    
    # Convergence analysis
    convergence_rate: float         # Exponential fit of delta decay
    final_residual: float
    
    # Energy analysis
    initial_energy: float
    final_energy: float
    energy_monotonic: bool          # Does energy decrease monotonically?
    energy_violations: int          # Number of steps where E increased
    
    # Stability
    avg_spectral_radius: Optional[float]
    max_spectral_radius: Optional[float]
    
    # Fixed point quality
    fixed_point_error: float        # ||h* - f(h*, x)||


def compute_iteration_metrics(
    states: List[Tensor],
    energies: List[float],
    model=None,
    x: Tensor = None
) -> List[IterationMetrics]:
    """Compute per-iteration metrics from trajectory.
    
    Args:
        states: List of h_t tensors
        energies: List of E(h_t) values
        model: Optional model for Jacobian computation
        x: Input tensor
        
    Returns:
        List of IterationMetrics per step
    """
    metrics = []
    
    for t, h_t in enumerate(states):
        # Basic norms
        h_norm = torch.norm(h_t).item()
        
        if t > 0:
            delta_norm = torch.norm(h_t - states[t-1]).item()
        else:
            delta_norm = float('inf')
        
        # Energy
        energy = energies[t] if t < len(energies) else float('nan')
        if t > 0 and t < len(energies):
            energy_delta = energies[t] - energies[t-1]
        else:
            energy_delta = 0.0
        
        # Spectral radius (expensive - skip by default)
        jacobian_spectral = None
        
        metrics.append(IterationMetrics(
            step=t,
            h_norm=h_norm,
            delta_norm=delta_norm,
            energy=energy,
            energy_delta=energy_delta,
            jacobian_spectral=jacobian_spectral,
            gradient_cosine=None
        ))
    
    return metrics


def compute_aggregate_metrics(
    iteration_metrics: List[IterationMetrics],
    converged: bool,
    convergence_step: Optional[int]
) -> AggregateMetrics:
    """Compute aggregate metrics from per-iteration data."""
    
    if not iteration_metrics:
        return AggregateMetrics(
            total_steps=0, converged=False, convergence_step=None,
            convergence_rate=0, final_residual=float('inf'),
            initial_energy=0, final_energy=0, 
            energy_monotonic=True, energy_violations=0,
            avg_spectral_radius=None, max_spectral_radius=None,
            fixed_point_error=float('inf')
        )
    
    # Convergence rate: fit exponential to delta decay
    deltas = [m.delta_norm for m in iteration_metrics if m.delta_norm < float('inf')]
    if len(deltas) > 2:
        log_deltas = np.log(np.array(deltas) + 1e-12)
        ts = np.arange(len(log_deltas))
        slope, _ = np.polyfit(ts, log_deltas, 1)
        convergence_rate = -slope  # Positive = converging
    else:
        convergence_rate = 0.0
    
    # Energy analysis
    energies = [m.energy for m in iteration_metrics if not np.isnan(m.energy)]
    energy_deltas = [m.energy_delta for m in iteration_metrics]
    energy_violations = sum(1 for d in energy_deltas if d > 1e-8)
    energy_monotonic = energy_violations == 0
    
    # Spectral analysis
    spectral_radii = [m.jacobian_spectral for m in iteration_metrics 
                      if m.jacobian_spectral is not None]
    
    return AggregateMetrics(
        total_steps=len(iteration_metrics),
        converged=converged,
        convergence_step=convergence_step,
        convergence_rate=convergence_rate,
        final_residual=iteration_metrics[-1].delta_norm,
        initial_energy=energies[0] if energies else float('nan'),
        final_energy=energies[-1] if energies else float('nan'),
        energy_monotonic=energy_monotonic,
        energy_violations=energy_violations,
        avg_spectral_radius=np.mean(spectral_radii) if spectral_radii else None,
        max_spectral_radius=max(spectral_radii) if spectral_radii else None,
        fixed_point_error=iteration_metrics[-1].delta_norm
    )


def compute_spectral_radius(model, h: Tensor, x: Tensor) -> float:
    """Compute spectral radius of Jacobian ∂f/∂h at point h.
    
    Uses power iteration for efficiency.
    """
    h = h.detach().requires_grad_(True)
    
    # Compute f(h)
    f_h, _ = model.forward_step(h, x, None)
    
    # Handle stacked state
    if f_h.dim() == 3:
        f_h = f_h[:, 0]
        h_flat = h[:, 0]
    else:
        h_flat = h
    
    # Power iteration for largest eigenvalue
    v = torch.randn_like(h_flat)
    v = v / torch.norm(v)
    
    for _ in range(20):  # Power iterations
        # Jacobian-vector product: J @ v
        Jv = torch.autograd.grad(f_h, h, grad_outputs=v, 
                                  retain_graph=True, create_graph=False)[0]
        if Jv.dim() == 3:
            Jv = Jv[:, 0]
        
        # Rayleigh quotient estimate
        eigenvalue = torch.dot(v.flatten(), Jv.flatten()) / torch.dot(v.flatten(), v.flatten())
        
        # Update v
        v = Jv / (torch.norm(Jv) + 1e-8)
    
    return abs(eigenvalue.item())
