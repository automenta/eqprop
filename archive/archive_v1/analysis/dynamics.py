import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DynamicsProfile:
    """Profile of dynamical system properties."""
    avg_lyapunov: float
    convergence_rate: float
    is_chaotic: bool
    energy_history: List[float]
    trajectory_length: int
    converged: bool
    final_residual: float

class DynamicsAnalyzer:
    """Analyzes the dynamical properties of Equilibrium Propagation.
    
    EqProp is a physical system that settles to equilibrium.
    This analyzer measures:
    1. Lyapunov Exponents: Rate of divergence/convergence.
    2. Phase Transitions: Detecting boundaries between order and chaos.
    3. Convergence Properties: How quickly the system settles.
    
    Usage:
        from analysis.dynamics import DynamicsAnalyzer
        analyzer = DynamicsAnalyzer(model)
        profile = analyzer.analyze_trajectory(x, steps=50)
        print(f"Convergence rate: {profile.convergence_rate}")
    """
    
    def __init__(self, model, device: str = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(model, 'to'):
            self.model.to(self.device)
    
    def analyze_trajectory(self, x: torch.Tensor, steps: int = 50, 
                          tol: float = 1e-5) -> DynamicsProfile:
        """Track the trajectory of the hidden states during inference.
        
        Args:
            x: Input tensor
            steps: Maximum number of equilibrium steps
            tol: Convergence tolerance
            
        Returns:
            DynamicsProfile with dynamical metrics
        """
        x = x.to(self.device)
        
        # Lyapunov estimation via perturbation
        perturbation_scale = 1e-6
        
        # 1. Run clean trajectory
        states_clean, energies_clean = self._collect_trajectory(x, steps, tol)
        
        # 2. Run perturbed trajectory
        x_perturbed = x + torch.randn_like(x) * perturbation_scale
        states_perturbed, _ = self._collect_trajectory(x_perturbed, steps, tol)
        
        # 3. Calculate divergence
        divergences = []
        min_len = min(len(states_clean), len(states_perturbed))
        for i in range(min_len):
            s1 = states_clean[i]
            s2 = states_perturbed[i]
            dist = torch.norm(s1 - s2).item()
            divergences.append(dist)
        
        # 4. Fit exponent: dist(t) ~ dist(0) * e^(lambda * t)
        if len(divergences) > 2:
            log_divs = np.log(np.array(divergences) + 1e-12)
            ts = np.arange(len(log_divs))
            slope, intercept = np.polyfit(ts, log_divs, 1)
        else:
            slope = 0.0
        
        # Check convergence
        if len(states_clean) > 1:
            final_residual = torch.norm(states_clean[-1] - states_clean[-2]).item()
            converged = final_residual < tol
        else:
            final_residual = float('inf')
            converged = False
        
        return DynamicsProfile(
            avg_lyapunov=slope,
            convergence_rate=-slope if slope < 0 else 0,
            is_chaotic=(slope > 0),
            energy_history=energies_clean,
            trajectory_length=len(states_clean),
            converged=converged,
            final_residual=final_residual
        )
    
    def _collect_trajectory(self, x: torch.Tensor, steps: int, 
                           tol: float = 1e-5) -> Tuple[List[torch.Tensor], List[float]]:
        """Collect hidden states over T steps during equilibrium settling.
        
        Attempts to use model-specific methods if available, otherwise
        uses generic forward pass tracking.
        """
        states = []
        energies = []
        
        self.model.eval()
        with torch.no_grad():
            # Method 1: Model has step() method (EqProp models)
            if hasattr(self.model, 'step'):
                state = self._init_state(x)
                for _ in range(steps):
                    state = self.model.step(x, state)
                    states.append(state.clone())
                    
                    if hasattr(self.model, 'energy'):
                        energies.append(self.model.energy(x, state).item())
                    
                    # Check convergence
                    if len(states) > 1:
                        diff = torch.norm(states[-1] - states[-2])
                        if diff < tol:
                            break
            
            # Method 2: Model has forward_with_trajectory()
            elif hasattr(self.model, 'forward_with_trajectory'):
                states, energies = self.model.forward_with_trajectory(x, steps)
            
            # Method 3: Use equilibrium solver directly
            elif hasattr(self.model, 'solver') and hasattr(self.model.solver, 'trajectory'):
                _, trajectory = self.model.solver.solve(self.model.encoder(x), 
                                                        return_trajectory=True)
                states = trajectory
            
            # Method 4: Generic forward pass (captures intermediate activations)
            else:
                # Register hooks to capture activations
                activations = []
                hooks = []
                
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        activations.append(output.detach().clone())
                
                # Register on all transformer blocks
                for name, module in self.model.named_modules():
                    if 'block' in name.lower() or 'layer' in name.lower():
                        hooks.append(module.register_forward_hook(hook_fn))
                
                # Run forward pass
                try:
                    _ = self.model(x)
                    states = activations
                finally:
                    for h in hooks:
                        h.remove()
        
        return states, energies
    
    def _init_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize hidden state for equilibrium models."""
        if hasattr(self.model, 'init_state'):
            return self.model.init_state(x)
        elif hasattr(self.model, 'encoder'):
            return self.model.encoder(x)
        else:
            # Fallback: zeros matching input shape
            return torch.zeros_like(x)
    
    def analyze_beta_sensitivity(self, x: torch.Tensor, 
                                 beta_values: List[float] = None,
                                 steps: int = 50) -> Dict[float, DynamicsProfile]:
        """Analyze dynamics across different β values.
        
        Args:
            x: Input tensor
            beta_values: List of β values to test
            steps: Equilibrium steps per β
            
        Returns:
            Dict mapping β to DynamicsProfile
        """
        if beta_values is None:
            beta_values = [0.05, 0.1, 0.2, 0.22, 0.25, 0.3, 0.5]
        
        results = {}
        original_beta = getattr(self.model, 'beta', None)
        
        for beta in beta_values:
            if hasattr(self.model, 'beta'):
                self.model.beta = beta
            elif hasattr(self.model, 'set_beta'):
                self.model.set_beta(beta)
            
            profile = self.analyze_trajectory(x, steps)
            results[beta] = profile
        
        # Restore original beta
        if original_beta is not None and hasattr(self.model, 'beta'):
            self.model.beta = original_beta
        
        return results
    
    def find_edge_of_chaos(self, x: torch.Tensor,
                           beta_range: Tuple[float, float] = (0.01, 1.0),
                           n_samples: int = 20) -> Optional[float]:
        """Find the β value at the edge of chaos.
        
        The edge of chaos is where Lyapunov exponent ≈ 0, the boundary
        between stable (negative) and chaotic (positive) regimes.
        
        Args:
            x: Input tensor
            beta_range: (min_beta, max_beta) to search
            n_samples: Number of β values to test
            
        Returns:
            Critical β value or None if not found
        """
        betas = np.linspace(beta_range[0], beta_range[1], n_samples)
        profiles = self.analyze_beta_sensitivity(x, list(betas))
        
        lyapunovs = [profiles[b].avg_lyapunov for b in betas]
        
        # Find zero crossing
        for i in range(len(lyapunovs) - 1):
            if lyapunovs[i] * lyapunovs[i + 1] < 0:
                # Linear interpolation to find approximate zero
                alpha = abs(lyapunovs[i]) / (abs(lyapunovs[i]) + abs(lyapunovs[i + 1]))
                critical_beta = betas[i] + alpha * (betas[i + 1] - betas[i])
                return float(critical_beta)
        
        return None
