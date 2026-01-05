"""Synthetic tasks for probing specific model behaviors."""

import torch
from torch import Tensor
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class TaskResult:
    """Result of running a model on a synthetic task."""
    accuracy: float
    success: bool
    details: dict


class SyntheticTask(ABC):
    """Base class for synthetic probing tasks."""
    
    name: str
    description: str
    input_dim: int
    output_dim: int
    
    @abstractmethod
    def generate_data(self, n_samples: int, device: str = "cpu") -> Tuple[Tensor, Tensor]:
        """Generate (X, Y) pairs for this task."""
        pass
    
    @abstractmethod
    def success_criterion(self, model_output: Tensor, expected: Tensor) -> bool:
        """Check if model output meets success criterion."""
        pass
    
    @abstractmethod
    def theoretical_expectation(self) -> str:
        """Describe expected behavior for EqProp models."""
        pass


class IdentityTask(SyntheticTask):
    """f(x) = x. Tests information preservation through equilibrium.
    
    A well-functioning equilibrium model should be able to preserve
    input information at its fixed point.
    """
    
    name = "Identity"
    description = "Output should equal input (information preservation)"
    
    def __init__(self, dim: int = 10):
        self.input_dim = dim
        self.output_dim = dim
    
    def generate_data(self, n_samples: int, device: str = "cpu") -> Tuple[Tensor, Tensor]:
        X = torch.randn(n_samples, self.input_dim, device=device)
        Y = X.clone()  # Identity mapping
        return X, Y
    
    def success_criterion(self, model_output: Tensor, expected: Tensor, threshold: float = 0.1) -> bool:
        mse = torch.mean((model_output - expected) ** 2).item()
        return mse < threshold
    
    def theoretical_expectation(self) -> str:
        return """
        Identity Task Expectations:
        - Fixed point should preserve input information
        - Energy should decrease as h → x
        - Symmetric models: Guaranteed convergence
        - Buffer models: Should stabilize faster
        """


class LinearTask(SyntheticTask):
    """f(x) = Ax. Tests ability to learn linear transformations.
    
    This probes gradient computation accuracy since linear tasks
    have closed-form solutions.
    """
    
    name = "Linear"
    description = "Learn a random linear transformation"
    
    def __init__(self, input_dim: int = 10, output_dim: int = 5, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        torch.manual_seed(seed)
        self.A = torch.randn(output_dim, input_dim) * 0.5
    
    def generate_data(self, n_samples: int, device: str = "cpu") -> Tuple[Tensor, Tensor]:
        X = torch.randn(n_samples, self.input_dim, device=device)
        Y = X @ self.A.T.to(device)
        return X, Y
    
    def success_criterion(self, model_output: Tensor, expected: Tensor, threshold: float = 0.1) -> bool:
        mse = torch.mean((model_output - expected) ** 2).item()
        return mse < threshold
    
    def theoretical_expectation(self) -> str:
        return """
        Linear Task Expectations:
        - EqProp gradients should exactly match BP gradients
        - Spectral radius of learned Wh should be < 1
        - Convergence rate proportional to 1 - ρ(A)
        """


class XORTask(SyntheticTask):
    """XOR/Parity task. Tests non-linear separability.
    
    XOR is not linearly separable, requiring the model to learn
    non-linear representations in its hidden state.
    """
    
    name = "XOR"
    description = "Learn XOR function (non-linear separation)"
    
    def __init__(self, n_bits: int = 2):
        self.n_bits = n_bits
        self.input_dim = n_bits
        self.output_dim = 2  # Binary classification
    
    def generate_data(self, n_samples: int, device: str = "cpu") -> Tuple[Tensor, Tensor]:
        # Generate all 2^n_bits patterns, then sample
        all_patterns = []
        for i in range(2 ** self.n_bits):
            pattern = [(i >> j) & 1 for j in range(self.n_bits)]
            all_patterns.append(pattern)
        
        patterns = torch.tensor(all_patterns, dtype=torch.float32, device=device)
        parity = patterns.sum(dim=1) % 2  # XOR = parity
        
        # Sample with replacement
        indices = torch.randint(0, len(patterns), (n_samples,))
        X = patterns[indices]
        Y = parity[indices].long()
        
        return X, Y
    
    def success_criterion(self, model_output: Tensor, expected: Tensor, threshold: float = 0.9) -> bool:
        preds = model_output.argmax(dim=-1)
        accuracy = (preds == expected).float().mean().item()
        return accuracy > threshold
    
    def theoretical_expectation(self) -> str:
        return """
        XOR Task Expectations:
        - Requires hidden capacity >= 2 units for 2-bit XOR
        - Energy landscape should have 2 distinct basins
        - Symmetric weights may limit expressivity
        - Gated models may show faster separation
        """


class MemorizationTask(SyntheticTask):
    """Memorize fixed input-output pairs. Tests associative memory.
    
    This probes the model's ability to form stable attractors
    in its energy landscape, similar to Hopfield networks.
    """
    
    name = "Memorization"
    description = "Memorize N fixed input-output associations"
    
    def __init__(self, n_patterns: int = 10, input_dim: int = 20, output_dim: int = 5, seed: int = 42):
        self.n_patterns = n_patterns
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        torch.manual_seed(seed)
        self.patterns_x = torch.randn(n_patterns, input_dim)
        self.patterns_y = torch.randint(0, output_dim, (n_patterns,))
    
    def generate_data(self, n_samples: int, device: str = "cpu") -> Tuple[Tensor, Tensor]:
        indices = torch.randint(0, self.n_patterns, (n_samples,))
        X = self.patterns_x[indices].to(device)
        Y = self.patterns_y[indices].to(device)
        return X, Y
    
    def success_criterion(self, model_output: Tensor, expected: Tensor, threshold: float = 1.0) -> bool:
        preds = model_output.argmax(dim=-1)
        accuracy = (preds == expected).float().mean().item()
        return accuracy >= threshold  # Must memorize perfectly
    
    def theoretical_expectation(self) -> str:
        return f"""
        Memorization Task Expectations:
        - Model should form {self.n_patterns} stable attractors
        - Energy minima should correspond to stored patterns
        - Capacity scales with hidden_dim (roughly 0.14 * hidden_dim for Hopfield)
        - ToroidalMLP buffer may help with pattern interference
        """


class AttractorTask(SyntheticTask):
    """Multiple inputs map to same output. Tests basin structure.
    
    Probes the energy landscape to verify multiple inputs
    converge to shared attractors.
    """
    
    name = "Attractor"
    description = "Multiple inputs should converge to same attractor"
    
    def __init__(self, n_attractors: int = 3, inputs_per_attractor: int = 5, 
                 input_dim: int = 20, seed: int = 42):
        self.n_attractors = n_attractors
        self.inputs_per_attractor = inputs_per_attractor
        self.input_dim = input_dim
        self.output_dim = n_attractors
        
        torch.manual_seed(seed)
        # Generate cluster centers
        self.centers = torch.randn(n_attractors, input_dim) * 2
        
        # Generate inputs around each center
        all_x = []
        all_y = []
        for i in range(n_attractors):
            noise = torch.randn(inputs_per_attractor, input_dim) * 0.3
            x = self.centers[i] + noise
            y = torch.full((inputs_per_attractor,), i, dtype=torch.long)
            all_x.append(x)
            all_y.append(y)
        
        self.patterns_x = torch.cat(all_x)
        self.patterns_y = torch.cat(all_y)
    
    def generate_data(self, n_samples: int, device: str = "cpu") -> Tuple[Tensor, Tensor]:
        indices = torch.randint(0, len(self.patterns_x), (n_samples,))
        X = self.patterns_x[indices].to(device)
        Y = self.patterns_y[indices].to(device)
        return X, Y
    
    def success_criterion(self, model_output: Tensor, expected: Tensor, threshold: float = 0.9) -> bool:
        preds = model_output.argmax(dim=-1)
        accuracy = (preds == expected).float().mean().item()
        return accuracy > threshold
    
    def theoretical_expectation(self) -> str:
        return f"""
        Attractor Task Expectations:
        - Model should form {self.n_attractors} basins of attraction
        - Inputs within same cluster should converge to similar h*
        - Basin boundaries visible in energy landscape
        - Convergence path should be smooth (no oscillation)
        """


def get_all_tasks() -> dict:
    """Return dictionary of all synthetic tasks."""
    return {
        "identity": IdentityTask(),
        "linear": LinearTask(),
        "xor": XORTask(),
        "memorization": MemorizationTask(),
        "attractor": AttractorTask()
    }
