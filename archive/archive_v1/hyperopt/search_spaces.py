from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import random

@dataclass
class SearchSpace(ABC):
    """Base class for hyperparameter search spaces."""
    
    @abstractmethod
    def sample(self, rng: random.Random = None) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        pass
    
    @abstractmethod
    def grid(self) -> List[Dict[str, Any]]:
        """Return all configurations in the grid."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Return the size of the search space."""
        pass


@dataclass
class EqPropSearchSpace(SearchSpace):
    """Search space for EqProp-specific hyperparameters.
    
    Covers all tunable aspects of Equilibrium Propagation:
    - Nudge strength (β)
    - Equilibrium solver parameters (damping, iterations, tolerance)
    - Architecture choices (attention type, symmetric mode)
    - Update mechanisms
    """
    
    # Nudge strength: CRITICAL - varies by task! 
    # β=0.22 previously assumed optimal, but experiments show higher β often works better
    beta: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    
    # Damping: controls convergence speed vs stability tradeoff
    damping: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 0.95])
    
    # Max iterations: compute budget for equilibrium finding
    max_iters: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    
    # Convergence tolerance
    tol: List[float] = field(default_factory=lambda: [1e-4, 1e-5, 1e-6])
    
    # Attention type: linear required for symmetric mode
    attention_type: List[str] = field(default_factory=lambda: ["linear"])
    
    # Symmetric mode: theoretical guarantees vs practical performance
    symmetric: List[bool] = field(default_factory=lambda: [False, True])
    
    # Update mechanism: how gradients are approximated
    update_mode: List[str] = field(default_factory=lambda: ["mse_proxy", "vector_field"])
    
    # Model size: now includes tiny sizes for micro tasks
    d_model: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    
    # Learning rate
    lr: List[float] = field(default_factory=lambda: [5e-4, 1e-3, 2e-3])
    
    def sample(self, rng: random.Random = None) -> Dict[str, Any]:
        """Sample a random EqProp configuration."""
        if rng is None:
            rng = random.Random()
        
        config = {
            "algorithm": "eqprop",
            "beta": rng.choice(self.beta),
            "damping": rng.choice(self.damping),
            "max_iters": rng.choice(self.max_iters),
            "tol": rng.choice(self.tol),
            "attention_type": rng.choice(self.attention_type),
            "symmetric": rng.choice(self.symmetric),
            "update_mode": rng.choice(self.update_mode),
            "d_model": rng.choice(self.d_model),
            "lr": rng.choice(self.lr),
        }
        
        # Symmetric mode requires linear attention
        if config["symmetric"] and config["attention_type"] != "linear":
            config["attention_type"] = "linear"
        
        return config
    
    def grid(self) -> List[Dict[str, Any]]:
        """Generate full grid of EqProp configurations."""
        import itertools
        
        configs = []
        for beta, damping, max_iters, tol, attn, sym, mode, d_model, lr in itertools.product(
            self.beta, self.damping, self.max_iters, self.tol,
            self.attention_type, self.symmetric, self.update_mode,
            self.d_model, self.lr
        ):
            # Skip invalid: symmetric requires linear attention
            if sym and attn != "linear":
                continue
            
            configs.append({
                "algorithm": "eqprop",
                "beta": beta,
                "damping": damping,
                "max_iters": max_iters,
                "tol": tol,
                "attention_type": attn,
                "symmetric": sym,
                "update_mode": mode,
                "d_model": d_model,
                "lr": lr,
            })
        
        return configs
    
    def size(self) -> int:
        """Approximate size of search space."""
        # Account for symmetric requiring linear attention
        valid_sym_combos = len(self.attention_type)  # symmetric=True only with linear
        valid_nonsym_combos = len(self.attention_type)  # symmetric=False with any
        
        base = (len(self.beta) * len(self.damping) * len(self.max_iters) * 
                len(self.tol) * len(self.update_mode) * len(self.d_model) * len(self.lr))
        
        return base * (valid_sym_combos + valid_nonsym_combos)


@dataclass
class BaselineSearchSpace(SearchSpace):
    """Search space for baseline (BP) hyperparameters.
    
    Covers standard backpropagation training parameters to ensure
    fair comparison with optimized EqProp.
    """
    
    # Learning rate
    lr: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    
    # Optimizer choice
    optimizer: List[str] = field(default_factory=lambda: ["adam", "adamw"])
    
    # Model size (match EqProp options - includes tiny sizes)
    d_model: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    
    # Weight decay for AdamW
    weight_decay: List[float] = field(default_factory=lambda: [0, 1e-4, 1e-3])
    
    # Scheduler
    scheduler: List[str] = field(default_factory=lambda: ["none", "cosine"])
    
    def sample(self, rng: random.Random = None) -> Dict[str, Any]:
        """Sample a random baseline configuration."""
        if rng is None:
            rng = random.Random()
        
        config = {
            "algorithm": "bp",
            "lr": rng.choice(self.lr),
            "optimizer": rng.choice(self.optimizer),
            "d_model": rng.choice(self.d_model),
            "weight_decay": rng.choice(self.weight_decay),
            "scheduler": rng.choice(self.scheduler),
        }
        
        return config
    
    def grid(self) -> List[Dict[str, Any]]:
        """Generate full grid of baseline configurations."""
        import itertools
        
        configs = []
        for lr, opt, d_model, wd, sched in itertools.product(
            self.lr, self.optimizer, self.d_model, self.weight_decay, self.scheduler
        ):
            configs.append({
                "algorithm": "bp",
                "lr": lr,
                "optimizer": opt,
                "d_model": d_model,
                "weight_decay": wd,
                "scheduler": sched,
            })
        
        return configs
    
    def size(self) -> int:
        """Size of baseline search space."""
        return (len(self.lr) * len(self.optimizer) * len(self.d_model) * 
                len(self.weight_decay) * len(self.scheduler))
