"""Update strategies for Equilibrium Propagation training.

This module implements different strategies for computing parameter updates
from equilibrium states. All strategies follow the UpdateStrategy interface.

Strategies:
- MSEProxyUpdate: MSE loss between model output and nudged equilibrium
- VectorFieldUpdate: Direct vector field backpropagation
- LocalHebbianUpdate: O(1) memory contrastive Hebbian learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, Optional


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
                            h_nudged: Tensor, x: Tensor) -> Optional[Tensor]:
        """Compute loss for model parameter update.
        
        Args:
            model: The equilibrium model
            h_free: Free phase equilibrium state
            h_nudged: Nudged phase equilibrium state  
            x: Input tensor
            
        Returns:
            Loss tensor for backpropagation, or None if gradients handled separately
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
        # Call forward_step for our model interface
        h_out, _ = model.forward_step(h_free_detached, x, None)
        
        # Scale by 1/beta for correct gradient magnitude
        loss = (1.0 / self.beta) * F.mse_loss(h_out, h_nudged.detach())
        return loss
    
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor,
                           y: Tensor) -> Tensor:
        """Standard cross-entropy loss at free equilibrium."""
        y_pred = output_head(h_free.detach())
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
            out_free, _ = model.forward_step(h_at_free, x, None)
        
        # Backward the vector v through model parameters
        out_free.backward(gradient=v)
        
        # Return None since gradients are accumulated directly
        return None
    
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor,
                           y: Tensor) -> Tensor:
        """Standard cross-entropy loss at free equilibrium."""
        with torch.enable_grad():
            y_pred = output_head(h_free)
            return F.cross_entropy(y_pred, y)


class LocalHebbianUpdate(UpdateStrategy):
    """Purely local Hebbian update mechanism (O(1) memory, no autodiff).
    
    Implements contrastive Hebbian learning without gradient computation:
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
        super().__init__(beta)
        self.activations_free: Dict[str, Tensor] = {}
        self.activations_nudged: Dict[str, Tensor] = {}
        self.hooks = []
        self.weight_updates: Dict[str, Tensor] = {}
        self.phase: Optional[str] = None  # 'free' or 'nudged'
    
    def _activation_hook(self, name: str):
        """Create activation hook for a specific layer."""
        def hook(module, input, output):
            # Store both input and output for proper Hebbian learning
            if self.phase == 'free':
                if isinstance(input, tuple) and len(input) > 0:
                    # Store input and output activations
                    self.activations_free[f"{name}_in"] = input[0].detach().clone()
                    self.activations_free[f"{name}_out"] = output.detach().clone()
            elif self.phase == 'nudged':
                if isinstance(input, tuple) and len(input) > 0:
                    self.activations_nudged[f"{name}_in"] = input[0].detach().clone()
                    self.activations_nudged[f"{name}_out"] = output.detach().clone()
        return hook
    
    def register_hooks(self, model: nn.Module):
        """Register forward hooks to capture activations."""
        self.remove_hooks()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._activation_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_hebbian_updates(self) -> Dict[str, Tensor]:
        """Compute contrastive Hebbian weight updates from stored activations."""
        weight_updates = {}
        
        # Find all layer names (without _in/_out suffix)
        layer_names = set()
        for key in self.activations_free.keys():
            if key.endswith('_in'):
                layer_names.add(key[:-3])
        
        for name in layer_names:
            in_key = f"{name}_in"
            out_key = f"{name}_out"
            
            if (in_key not in self.activations_free or out_key not in self.activations_free or
                in_key not in self.activations_nudged or out_key not in self.activations_nudged):
                continue
            
            # Get input and output activations for both phases
            in_free = self.activations_free[in_key]
            out_free = self.activations_free[out_key]
            in_nudged = self.activations_nudged[in_key]
            out_nudged = self.activations_nudged[out_key]
            
            batch_size = in_free.size(0)
            
            # Contrastive Hebbian rule for weight matrix W (output_dim x input_dim):
            # ΔW = (1/β) * (1/N) * (out_nudged^T @ in_nudged - out_free^T @ in_free)
            grad_free = torch.matmul(out_free.T, in_free) / batch_size
            grad_nudged = torch.matmul(out_nudged.T, in_nudged) / batch_size
            
            # Gradient for descent direction
            # NOTE: Archive v1 used negative sign.
            # Descent: W -= lr * grad
            # effective grad ~ (nudged - free).
            # So W -= lr * (nudged - free).
            # Here we accumulate into weight_updates which is added: W += lr * update.
            # So update should be -(nudged - free).
            weight_grad = -(1.0 / self.beta) * (grad_nudged - grad_free)
            weight_updates[name] = weight_grad
        
        return weight_updates
    
    # Removed apply_updates_to_model as we now inject into .grad
    
    def compute_update(self, model: nn.Module, h_free: Tensor, h_nudged: Tensor,
                      x: Tensor, y: Tensor, optimizer):
        """Unified interface for trainer: compute and apply Hebbian updates.
        
        This method is called by EqPropTrainer when update_strategy is set.
        """
        # Register hooks if not already done
        if not self.hooks:
            self.register_hooks(model)
        
        # Record activations during free phase
        self.phase = 'free'
        # Crucial fix: Use the ACTUAL equilibrium state, not a fresh 1-step run
        # This triggers the hooks with the converged stable state
        with torch.no_grad():
             model.forward_step(h_free, x, None)
        
        # Record activations during nudged phase  
        self.phase = 'nudged'
        with torch.no_grad():
             model.forward_step(h_nudged, x, None)
        
        # Compute Hebbian updates
        self.weight_updates = self.compute_hebbian_updates()
        
        # Inject Hebbian updates into gradients (for Body)
        # This allows using the optimizer (Adam, etc) instead of manual updates
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.weight_updates:
                if name == 'Head': continue # Skip Head, train via loss
                
                # Handle Spectral Normalization (PyTorch Parametrizations)
                # module.weight is a computed non-leaf tensor.
                # The actual parameter is usually in module.parametrizations.weight.original
                param = None
                
                if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
                     # Modern Pytorch SN
                     param = module.parametrizations.weight.original
                elif hasattr(module, 'weight_orig'):
                     # Legacy Pytorch SN
                     param = module.weight_orig
                else:
                     # Standard Linear
                     param = module.weight
                
                # Check if param is a leaf (crucial for optimizer)
                if not param.is_leaf:
                    # Fallback: try to find the leaf parameter by shape
                    for p in module.parameters():
                        if p.shape == module.weight.shape and p.is_leaf:
                            param = p
                            break
                
                # Gradient = weight_grad (which is free - nudged)
                if param is not None and param.requires_grad:
                    if param.grad is None:
                        param.grad = self.weight_updates[name]
                    else:
                        param.grad += self.weight_updates[name]
        
        # Train Output Head via standard Backprop (Supervised)
        # Nudged phase energy includes Cost, but explicit head training is more robust
        head_loss = self.compute_head_update(model.Head, h_free, y)
        head_loss.backward() # Adds to Head.weight.grad
        
        # Step optimizer
        optimizer.step()
        
        # Clear stored activations
        self.activations_free.clear()
        self.activations_nudged.clear()
        self.phase = None
    
    def compute_model_update(self, model: nn.Module, h_free: Tensor,
                            h_nudged: Tensor, x: Tensor) -> None:
        """Compute Hebbian updates from free and nudged equilibria.
        
        Returns None to signal that gradients are handled via direct weight updates.
        """
        self.weight_updates = self.compute_hebbian_updates()
        return None
    
    def compute_head_update(self, output_head: nn.Module, h_free: Tensor,
                           y: Tensor) -> Tensor:
        """Compute output head update using standard cross-entropy."""
        with torch.enable_grad():
            y_pred = output_head(h_free)
            return F.cross_entropy(y_pred, y)
