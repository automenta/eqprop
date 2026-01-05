"""Equilibrium Propagation trainer with configurable update strategies."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Callable
from .solver import EquilibriumSolver
from .updates import UpdateStrategy, MSEProxyUpdate, VectorFieldUpdate, LocalHebbianUpdate


class EqPropTrainer:
    """Equilibrium Propagation training loop.
    
    Supports configurable update mechanisms via the strategy pattern:
    - 'mse_proxy': MSE loss between model output and nudged equilibrium
    - 'vector_field': Vector field backprop through equilibrium
    
    References:
        Scellier & Bengio (2017), "Equilibrium Propagation: Bridging the
        Gap between Energy-Based Models and Backpropagation"
    """

    def __init__(self, model: nn.Module, solver: EquilibriumSolver, 
                 output_head: nn.Module, beta: float = 0.1, lr: float = 1e-3, 
                 update_mode: str = 'mse_proxy', beta_schedule=None):
        """Initialize EqProp trainer.
        
        Args:
            model: Equilibrium model (e.g., LoopedTransformerBlock)
            solver: Fixed-point solver for finding equilibria
            output_head: Classification/output head
            beta: Nudge strength parameter
            lr: Learning rate
            update_mode: Update strategy ('mse_proxy' or 'vector_field')
            beta_schedule: Optional callable (epoch -> beta) for β annealing
            
        Raises:
            ValueError: If update_mode is not recognized
        """
        self.model = model
        self.solver = solver
        self.output_head = output_head
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.current_epoch = 0
        
        # Initialize update strategy
        self.update_strategy = self._create_update_strategy(update_mode, beta)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(output_head.parameters()),
            lr=lr
        )
    
    @staticmethod
    def _create_update_strategy(update_mode: str, beta: float) -> UpdateStrategy:
        """Factory method for creating update strategy."""
        if update_mode == 'mse_proxy':
            return MSEProxyUpdate(beta)
        elif update_mode == 'vector_field':
            return VectorFieldUpdate(beta)
        elif update_mode == 'local_hebbian':
            return LocalHebbianUpdate(beta)
        else:
            raise ValueError(
                f"Unknown update_mode '{update_mode}'. "
                f"Must be 'mse_proxy', 'vector_field', or 'local_hebbian'"
            )
    
    def update_beta(self, epoch: int = None):
        """Update β value based on schedule or epoch.
        
        Args:
            epoch: Current epoch number. If None, uses self.current_epoch
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.beta_schedule is not None:
            new_beta = self.beta_schedule(self.current_epoch)
            self.beta = new_beta
            self.update_strategy.beta = new_beta
    
    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Single training step using Equilibrium Propagation.
        
        Nudge Sign Convention:
            - Compute grads = ∇L (gradient of loss)
            - Apply nudge as: h_new - β * grads
            - This moves h in direction that decreases loss
            - Equivalent to: h_new + β * ∇(-L)
        
        Args:
            x: Input [seq, batch, d_model]
            y: Target labels [batch]
            
        Returns:
            Dictionary with loss, accuracy, and iteration counts
        """
        # Register hooks for LocalHebbianUpdate if needed
        if isinstance(self.update_strategy, LocalHebbianUpdate):
            self.update_strategy.register_hooks(self.model)
        
        # Phase 1: Find free equilibrium (without nudging)
        if isinstance(self.update_strategy, LocalHebbianUpdate):
            self.update_strategy.phase = 'free'
        h_free, iters_free = self._solve_free_equilibrium(x)
        
        # Phase 2: Find nudged equilibrium (with loss-based nudging)
        if isinstance(self.update_strategy, LocalHebbianUpdate):
            self.update_strategy.phase = 'nudged'
        h_nudged, iters_nudged = self._solve_nudged_equilibrium(h_free, x, y)
        
        # Phase 3: Update parameters using configured strategy
        self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        total_loss = self._compute_and_apply_updates(h_free, h_nudged, x, y)
        self.optimizer.step()
        
        # Compute metrics
        metrics = self._compute_metrics(h_free, y, iters_free, iters_nudged, total_loss)
        return metrics
    
    def _solve_free_equilibrium(self, x: Tensor) -> tuple[Tensor, int]:
        """Find equilibrium without nudging."""
        h0 = torch.zeros_like(x)
        with torch.no_grad():
            h_free, iters = self.solver.solve(self.model, h0, x)
        return h_free, iters
    
    def _solve_nudged_equilibrium(self, h_free: Tensor, x: Tensor, 
                                   y: Tensor) -> tuple[Tensor, int]:
        """Find equilibrium with loss-based nudging."""
        def nudged_dynamics(h: Tensor, x: Tensor) -> Tensor:
            h = h.detach().requires_grad_(True)
            h_new = self.model(h, x)
            y_pred = self.output_head(h_new.mean(dim=0))
            loss = F.cross_entropy(y_pred, y)
            
            # Nudge in direction that decreases loss: h - β * ∇L
            grads = torch.autograd.grad(loss, h_new, create_graph=True, 
                                       retain_graph=True)[0]
            return h_new - self.beta * grads
        
        h_nudged, iters = self.solver.solve(nudged_dynamics, h_free.detach(), x)
        return h_nudged, iters
    
    def _compute_and_apply_updates(self, h_free: Tensor, h_nudged: Tensor,
                                    x: Tensor, y: Tensor) -> Tensor:
        """Compute gradients using update strategy and return total loss."""
        # Model parameter update
        model_loss = self.update_strategy.compute_model_update(
            self.model, h_free, h_nudged, x
        )
        
        # Output head update
        head_loss = self.update_strategy.compute_head_update(
            self.output_head, h_free, y
        )
        
        # Special handling for LocalHebbianUpdate (O(1) memory mode)
        if isinstance(self.update_strategy, LocalHebbianUpdate) and model_loss is None:
            # Pure Hebbian: apply weight updates directly without autodiff
            # This is the TRUE O(1) memory implementation
            head_loss.backward()  # Only backprop for output head
            
            # Apply Hebbian updates directly to model parameters
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear) and name in self.update_strategy.weight_updates:
                        # Get learning rate from optimizer
                        lr = self.optimizer.param_groups[0]['lr']
                        # Apply update: W = W + lr * ΔW
                        module.weight.data += lr * self.update_strategy.weight_updates[name]
            
            return head_loss
        
        # Standard backprop path (MSE proxy or vector field)
        if model_loss is not None:
            # MSE proxy: single backward pass
            total_loss = model_loss + head_loss
            total_loss.backward()
            return total_loss
        else:
            # Vector field: gradients already accumulated for model
            head_loss.backward()
            return head_loss
    
    def _compute_metrics(self, h_free: Tensor, y: Tensor, iters_free: int,
                        iters_nudged: int, total_loss: Tensor) -> Dict[str, float]:
        """Compute training metrics."""
        with torch.no_grad():
            y_pred = self.output_head(h_free.mean(dim=0))
            acc = (y_pred.argmax(-1) == y).float().mean()
        
        return {
            "loss": total_loss.item(),
            "accuracy": acc.item(),
            "iters_free": iters_free,
            "iters_nudged": iters_nudged
        }
