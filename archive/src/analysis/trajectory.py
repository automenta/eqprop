"""Trajectory recording for per-iteration analysis."""

import torch
from torch import Tensor
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Trajectory:
    """Complete trajectory of an equilibrium solve."""
    states: List[Tensor]           # h_t at each step
    energies: List[float]          # E(h_t) at each step
    deltas: List[float]            # ||h_t - h_{t-1}|| at each step
    converged: bool
    convergence_step: Optional[int]
    final_residual: float
    buffer_states: List[Any] = field(default_factory=list)  # For ToroidalMLP
    
    @property
    def num_steps(self) -> int:
        return len(self.states)
    
    def get_state(self, t: int) -> Tensor:
        """Get state at time t (supports negative indexing)."""
        return self.states[t]
    
    def energy_deltas(self) -> List[float]:
        """Compute E(h_{t+1}) - E(h_t) for energy descent check."""
        if len(self.energies) < 2:
            return []
        return [self.energies[i+1] - self.energies[i] 
                for i in range(len(self.energies) - 1)]


class TrajectoryRecorder:
    """Records complete trajectory during equilibrium solving.
    
    Hooks into a model's forward_step to capture every iteration.
    """
    
    def __init__(self, model, epsilon: float = 1e-4, max_steps: int = 100):
        self.model = model
        self.epsilon = epsilon
        self.max_steps = max_steps
    
    def record(self, x: Tensor) -> Trajectory:
        """Run model to equilibrium, recording all states.
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            Trajectory with complete iteration history
        """
        batch_size = x.size(0)
        device = x.device
        
        states: List[Tensor] = []
        energies: List[float] = []
        deltas: List[float] = []
        buffer_states: List[Any] = []
        
        # Initialize state
        if hasattr(self.model, 'hidden_dim'):
            h = torch.zeros(batch_size, self.model.hidden_dim, device=device)
        else:
            h = torch.zeros_like(x)
        
        # Handle ToroidalMLP stacked state
        if hasattr(self.model, 'buffer_size') and hasattr(self.model, 'forward_step'):
            # Check if forward_step expects stacked input
            zeros = torch.zeros(batch_size, self.model.buffer_size, 
                               self.model.hidden_dim, device=device)
            h = torch.cat([h.unsqueeze(1), zeros], dim=1)
        
        buffer_state = None
        converged = False
        convergence_step = None
        
        with torch.no_grad():
            for t in range(self.max_steps):
                h_prev = h.clone()
                
                # Forward step
                h, buffer_state = self.model.forward_step(h, x, buffer_state)
                
                # Record state (handle stacked vs flat)
                if h.dim() == 3:  # Stacked state
                    states.append(h[:, 0].clone())  # Current state
                    buffer_states.append(h[:, 1:].clone())
                else:
                    states.append(h.clone())
                
                # Compute energy
                if hasattr(self.model, 'energy'):
                    try:
                        e = self.model.energy(h, x, buffer_state)
                        energies.append(e.item() if isinstance(e, Tensor) else e)
                    except:
                        energies.append(float('nan'))
                
                # Compute delta
                if h.dim() == 3:
                    delta = torch.norm(h[:, 0] - h_prev[:, 0], dim=-1).max().item()
                else:
                    delta = torch.norm(h - h_prev, dim=-1).max().item()
                deltas.append(delta)
                
                # Check convergence
                if delta < self.epsilon:
                    converged = True
                    convergence_step = t + 1
                    break
        
        final_residual = deltas[-1] if deltas else float('inf')
        
        return Trajectory(
            states=states,
            energies=energies,
            deltas=deltas,
            converged=converged,
            convergence_step=convergence_step,
            final_residual=final_residual,
            buffer_states=buffer_states
        )
    
    def record_with_gradients(self, x: Tensor, y: Tensor) -> Tuple[Trajectory, Trajectory]:
        """Record both free and nudged phase trajectories.
        
        Args:
            x: Input
            y: Target labels
            
        Returns:
            (free_trajectory, nudged_trajectory)
        """
        import torch.nn.functional as F
        
        # Free phase
        free_traj = self.record(x)
        
        # Get equilibrium state
        h_free = free_traj.states[-1]
        
        # Compute nudging gradient
        h_var = h_free.clone().requires_grad_(True)
        y_hat = self.model.Head(h_var)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        dL_dh = h_var.grad.detach()
        
        # Nudged phase - would need solver with nudging
        # For now, return free trajectory twice as placeholder
        nudged_traj = self.record(x)  # TODO: implement nudged recording
        
        return free_traj, nudged_traj
