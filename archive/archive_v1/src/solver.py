"""Fixed-point equilibrium solver for equilibrium propagation."""

import torch
from torch import Tensor
from typing import Callable, Tuple


class EquilibriumSolver:
    """Fixed-point solver with damped iteration and convergence monitoring.
    
    Solves for h* such that h* = f(h*, x) using damped fixed-point iteration:
        h_{t+1} = (1 - α) h_t + α f(h_t, x)
    
    where α ∈ (0, 1] is the damping factor.
    """

    def __init__(self, max_iters: int = 50, tol: float = 1e-5, damping: float = 0.9):
        """Initialize solver.
        
        Args:
            max_iters: Maximum number of iterations
            tol: Convergence tolerance (L2 norm of residual)
            damping: Damping factor α ∈ (0, 1]. Higher = less damping.
        """
        self.max_iters = int(max_iters)
        self.tol = float(tol)  # Ensure tol is always a float
        self.damping = float(damping)

    def solve(self, f: Callable[[Tensor, Tensor], Tensor], 
              h0: Tensor, x: Tensor) -> Tuple[Tensor, int]:
        """Solve for equilibrium h* = f(h*, x).
        
        Args:
            f: Dynamics function f(h, x) -> h_new
            h0: Initial state
            x: Input (constant during iteration)
            
        Returns:
            Tuple of (equilibrium state, number of iterations)
        """
        h = h0
        
        # Check convergence every few iterations for efficiency
        check_interval = max(1, min(5, self.max_iters // 10))
        
        for t in range(self.max_iters):
            fx = f(h, x)
            
            # Damped update: h = (1-α)*h + α*fx
            # Use out-of-place lerp to preserve gradient graph
            h = torch.lerp(h, fx, self.damping)
            
            # Check convergence periodically (residual check is expensive)
            if (t + 1) % check_interval == 0 or t == self.max_iters - 1:
                # Use max norm (cheaper than L2 norm)
                residual = (fx - h).abs().max().item()
                if residual < self.tol:
                    return h, t + 1
        
        # Did not converge within max_iters
        return h, self.max_iters
