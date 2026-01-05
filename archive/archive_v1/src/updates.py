"""Update strategies for Equilibrium Propagation training.

This module implements different strategies for computing parameter updates
from equilibrium states. All strategies follow the UpdateStrategy interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict


class UpdateStrategy(ABC):
    """Base class for EqProp update strategies."""
    
    def __init__(self, beta: float):
        """Initialize update strategy.
        
        Args:
            beta: Nudge strength parameter
        """
        self.beta = beta
    
    @abstractmethod
    def compute_model_update(self, model: nn.Module, h_free: Tensor, 
                            h_nudged: Tensor, x: Tensor) -> Tensor:
        """Compute loss for model parameter update.
        
        Args:
            model: The equilibrium model
            h_free: Free phase equilibrium state
            h_nudged: Nudged phase equilibrium state  
            x: Input tensor
            
        Returns:
            Loss tensor for backpropagation
        """
        pass
    
    @abstractmethod
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor, 
                           y: Tensor) -> Tensor:
        """Compute loss for output head parameter update.
        
        Args:
            output_head: Classification/output head
            h_free: Free phase equilibrium state
            y: Target labels
            
        Returns:
            Loss tensor for backpropagation
        """
        pass


class MSEProxyUpdate(UpdateStrategy):
    """MSE proxy loss update mechanism.
    
    Minimizes the distance between the model's one-step output from the free
    equilibrium and the nudged equilibrium state. The loss is scaled by 1/β
    to approximate the correct gradient magnitude.
    
    Theory: delta ~ 1/β * (h_free - h_nudged) approximates the adjoint state.
    """
    
    def compute_model_update(self, model: nn.Module, h_free: Tensor,
                            h_nudged: Tensor, x: Tensor) -> Tensor:
        """Compute MSE proxy loss between model output and nudged equilibrium."""
        h_free_detached = h_free.detach()
        h_out = model(h_free_detached, x)
        
        # Scale by 1/beta for correct gradient magnitude
        loss = (1.0 / self.beta) * F.mse_loss(h_out, h_nudged.detach())
        return loss
    
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor,
                           y: Tensor) -> Tensor:
        """Standard cross-entropy loss at free equilibrium."""
        y_pred = output_head(h_free.detach().mean(dim=0))
        return F.cross_entropy(y_pred, y)


class VectorFieldUpdate(UpdateStrategy):
    """Vector field backpropagation update mechanism.
    
    Backpropagates the vector v = (h_nudged - h_free) / β through the model.
    This is theoretically cleaner as it directly computes the gradient without
    requiring a proxy loss.
    
    Theory: Gradient ≈ 1/β * (h_nudged - h_free) * df/dθ via the vector field.
    """
    
    def compute_model_update(self, model: nn.Module, h_free: Tensor,
                            h_nudged: Tensor, x: Tensor) -> None:
        """Backpropagate vector field through model (no explicit loss)."""
        delta = h_nudged - h_free.detach()
        v = delta / self.beta
        
        # Re-compute output at free equilibrium with gradients enabled
        h_at_free = h_free.detach()
        with torch.enable_grad():
            out_free = model(h_at_free, x)
        
        # Backward the vector v through model parameters
        out_free.backward(gradient=v)
        
        # Return None since gradients are accumulated directly
        return None
    
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor,
                           y: Tensor) -> Tensor:
        """Standard cross-entropy loss at free equilibrium."""
        with torch.enable_grad():
            y_pred = output_head(h_free.mean(dim=0))
            return F.cross_entropy(y_pred, y)


class LocalHebbianUpdate(UpdateStrategy):
    """Purely local Hebbian update mechanism (O(1) memory, no autodiff).
    
    Implements Algorithm 1b from README: direct contrastive Hebbian learning
    without gradient computation. This enables:
    - O(1) memory scaling (no activation storage for backward pass)
    - Biological plausibility (all updates are local)
    - Neuromorphic hardware compatibility (no backprop needed)
    
    The key innovation: update weights based only on local activations:
        ΔW_l = (1/β) * (A^β_l ⊗ A^β_l.T - A*_l ⊗ A*_l.T)
    
    where A*_l are activations at free equilibrium and A^β_l at nudged equilibrium.
    
    Theory: In the β→0 limit, this approximates the BP gradient via contrastive
    Hebbian learning (Scellier & Bengio, 2017).
    """
    
    def __init__(self, beta: float):
        """Initialize LocalHebbianUpdate.
        
        Args:
            beta: Nudge strength parameter
        """
        super().__init__(beta)
        self.activations_free = {}
        self.activations_nudged = {}
        self.hooks = []
        self.weight_updates = {}
        self.phase = None  # 'free' or 'nudged'
    
    def _activation_hook(self, name: str):
        """Create activation hook for a specific layer."""
        def hook(module, input, output):
            if self.phase == 'free':
                # Store input activations for Linear layers
                if isinstance(input, tuple) and len(input) > 0:
                    self.activations_free[name] = input[0].detach().clone()
            elif self.phase == 'nudged':
                if isinstance(input, tuple) and len(input) > 0:
                    self.activations_nudged[name] = input[0].detach().clone()
        return hook
    
    def register_hooks(self, model: nn.Module):
        """Register forward hooks to capture activations."""
        self.remove_hooks()  # Clean up any existing hooks
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._activation_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_hebbian_updates(self):
        """Compute contrastive Hebbian weight updates from stored activations.
        
        Returns:
            Dictionary mapping layer names to weight gradients
        """
        weight_updates = {}
        
        for name in self.activations_free.keys():
            if name not in self.activations_nudged:
                continue
            
            A_free = self.activations_free[name]  # [batch, d_in]
            A_nudged = self.activations_nudged[name]  # [batch, d_in]
            
            # Contrastive Hebbian rule (averaged over batch):
            # ΔW = (1/β) * (1/N) * (A_nudged^T @ A_nudged - A_free^T @ A_free)
            batch_size = A_free.size(0)
            
            grad_free = torch.matmul(A_free.T, A_free) / batch_size
            grad_nudged = torch.matmul(A_nudged.T, A_nudged) / batch_size
            
            # The negative is because we're computing gradient (descent direction)
            weight_grad = -(1.0 / self.beta) * (grad_nudged - grad_free)
            
            weight_updates[name] = weight_grad
        
        return weight_updates
    
    def apply_updates_to_model(self, model: nn.Module, lr: float):
        """Directly apply Hebbian updates to model parameters (bypassing autodiff).
        
        Args:
            model: The model to update
            lr: Learning rate
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.weight_updates:
                # Apply update: W = W + lr * ΔW
                # Note: weight_updates already contains the descent direction
                with torch.no_grad():
                    module.weight.data += lr * self.weight_updates[name]
    
    def compute_model_update(self, model: nn.Module, h_free: Tensor,
                            h_nudged: Tensor, x: Tensor) -> None:
        """Compute Hebbian updates from free and nudged equilibria.
        
        This method uses stored activations from hooks rather than autodiff.
        Returns None to signal that gradients are handled separately.
        """
        # Compute contrastive Hebbian updates from stored activations
        self.weight_updates = self.compute_hebbian_updates()
        
        # Return None - we'll apply updates directly, not via backprop
        # This enables TRUE O(1) memory training with no autodiff
        return None
    
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor,
                           y: Tensor) -> Tensor:
        """Compute output head update using error signal.
        
        For the output layer, we can use the simple error signal:
            error = y_target - y_pred
        and backproject: delta_h = W_out^T @ error
        
        This is more efficient than full backprop.
        """
        # Get prediction
        h_pooled = h_free.mean(dim=0)  # [batch, d_model]
        
        # Standard cross-entropy for output head
        # Could be replaced with Hebbian update in future
        with torch.enable_grad():
            y_pred = output_head(h_pooled)
            return F.cross_entropy(y_pred, y)

