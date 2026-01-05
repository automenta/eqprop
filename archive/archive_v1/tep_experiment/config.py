"""
Configuration dataclasses for TEP experiments.

Defines search spaces, phase configurations, and trial results per the
experiment specification.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Callable
from enum import Enum
from pathlib import Path


class Algorithm(Enum):
    """Supported algorithms for comparison."""
    TEP = "tep"
    BP = "bp"


class TaskDifficulty(Enum):
    """Task difficulty tiers."""
    PHASE1 = "phase1"  # XOR, 8x8 digits
    PHASE2 = "phase2"  # MNIST 28x28, CartPole
    PHASE3 = "phase3"  # CIFAR-10, sequences


# =============================================================================
# SEARCH SPACES
# =============================================================================

@dataclass
class SharedSearchSpace:
    """Hyperparameters shared between TEP and BP.
    
    Both algorithms receive identical treatment for these parameters:
    - Same search ranges
    - Same sampling distributions
    - Same optimization budget
    """
    # Architecture
    n_hidden_layers: Tuple[int, int] = (1, 4)
    hidden_units_range: Tuple[int, int] = (4, 512)  # log-uniform sampling
    activation_choices: Tuple[str, ...] = ("tanh", "relu")
    
    # Training
    learning_rate_range: Tuple[float, float] = (1e-4, 1e-1)  # log sampling
    batch_size_choices: Tuple[int, ...] = (32, 64, 128, 256)
    
    def get_optuna_suggestions(self, trial) -> Dict[str, Any]:
        """Sample parameters using Optuna trial."""
        return {
            "n_hidden_layers": trial.suggest_int(
                "n_hidden_layers", 
                self.n_hidden_layers[0], 
                self.n_hidden_layers[1]
            ),
            "hidden_units": trial.suggest_int(
                "hidden_units",
                self.hidden_units_range[0],
                self.hidden_units_range[1],
                log=True
            ),
            "activation": trial.suggest_categorical(
                "activation",
                list(self.activation_choices)
            ),
            "lr": trial.suggest_float(
                "lr",
                self.learning_rate_range[0],
                self.learning_rate_range[1],
                log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                list(self.batch_size_choices)
            ),
        }


@dataclass
class TEPSearchSpace:
    """TEP-specific hyperparameters.
    
    These are added to the shared space for TEP trials only.
    """
    # Nudging strength β: controls gradient-based nudge magnitude
    nudging_beta_range: Tuple[float, float] = (0.01, 0.5)  # log-uniform
    
    # Dampening factor γ: stability control for equilibrium iterations
    dampening_gamma_range: Tuple[float, float] = (0.5, 0.99)
    
    # Equilibrium iterations per step
    equilibrium_iters_range: Tuple[int, int] = (5, 50)
    
    # Convergence tolerance for equilibrium solver
    tolerance_range: Tuple[float, float] = (1e-5, 1e-3)  # log-uniform
    
    # Attention mechanism type
    attention_type_choices: Tuple[str, ...] = ("linear", "softmax")
    
    # Symmetric mode (fixed-point properties)
    symmetric_choices: Tuple[bool, ...] = (False, True)
    
    def get_optuna_suggestions(self, trial) -> Dict[str, Any]:
        """Sample TEP-specific parameters."""
        return {
            "beta": trial.suggest_float(
                "beta",
                self.nudging_beta_range[0],
                self.nudging_beta_range[1],
                log=True
            ),
            "gamma": trial.suggest_float(
                "gamma",
                self.dampening_gamma_range[0],
                self.dampening_gamma_range[1]
            ),
            "eq_iters": trial.suggest_int(
                "eq_iters",
                self.equilibrium_iters_range[0],
                self.equilibrium_iters_range[1]
            ),
            "tolerance": trial.suggest_float(
                "tolerance",
                self.tolerance_range[0],
                self.tolerance_range[1],
                log=True
            ),
            "attention_type": trial.suggest_categorical(
                "attention_type",
                list(self.attention_type_choices)
            ),
            "symmetric": trial.suggest_categorical(
                "symmetric",
                list(self.symmetric_choices)
            ),
        }


@dataclass
class BPSearchSpace:
    """BP-specific hyperparameters.
    
    Standard backpropagation training parameters.
    Note: d_model comes from shared space as hidden_units.
    """
    # Optimizer choice
    optimizer_choices: Tuple[str, ...] = ("adam", "adamw")
    
    # Weight decay (for adamw)
    weight_decay_choices: Tuple[float, ...] = (0.0, 1e-4, 1e-3)
    
    def get_optuna_suggestions(self, trial) -> Dict[str, Any]:
        """Sample BP-specific parameters."""
        return {
            "optimizer": trial.suggest_categorical(
                "optimizer",
                list(self.optimizer_choices)
            ),
            "weight_decay": trial.suggest_categorical(
                "weight_decay", 
                list(self.weight_decay_choices)
            ),
        }


# =============================================================================
# TRIAL RESULTS
# =============================================================================

@dataclass
class TrialResult:
    """Result of a single trial with all 4 objectives.
    
    Multi-objective optimization targets:
    1. accuracy: Maximize final validation accuracy
    2. wall_time: Minimize total training wall-time
    3. param_count: Minimize number of trainable parameters
    4. convergence_steps: Minimize steps to 90% of best accuracy
    """
    # Trial identification
    trial_id: str
    algorithm: str
    task: str
    seed: int
    config: Dict[str, Any]
    
    # 4 Objectives
    accuracy: float = 0.0
    wall_time_seconds: float = 0.0
    param_count: int = 0
    convergence_steps: int = 0  # Steps to reach 90% of best accuracy
    
    # Additional metrics
    convergence_curve: List[float] = field(default_factory=list)
    final_loss: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Status
    status: str = "pending"  # pending, running, complete, failed, timeout, pruned
    error: str = ""
    log_path: str = ""
    
    @property
    def objectives(self) -> Tuple[float, float, float, float]:
        """Return objectives as tuple for Optuna.
        
        Returns: (accuracy, wall_time, param_count, convergence_steps)
        Note: Optuna handles maximization/minimization via directions.
        """
        return (
            self.accuracy,
            self.wall_time_seconds,
            float(self.param_count),
            float(self.convergence_steps)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trial_id": self.trial_id,
            "algorithm": self.algorithm,
            "task": self.task,
            "seed": self.seed,
            "config": self.config,
            "accuracy": self.accuracy,
            "wall_time_seconds": self.wall_time_seconds,
            "param_count": self.param_count,
            "convergence_steps": self.convergence_steps,
            "convergence_curve": self.convergence_curve,
            "final_loss": self.final_loss,
            "peak_memory_mb": self.peak_memory_mb,
            "status": self.status,
            "error": self.error,
        }


# =============================================================================
# PHASE CONFIGURATIONS
# =============================================================================

@dataclass
class PhaseConfig:
    """Configuration for an experiment phase."""
    name: str
    phase_number: int
    
    # Tasks for this phase
    tasks: List[str]
    
    # Trial budget
    n_trials_per_algorithm: int  # 300-600 for Phase 1
    
    # Timing constraints
    target_trial_duration: Tuple[float, float]  # (min, max) seconds
    trial_timeout_seconds: float  # Hard timeout
    total_budget_hours: float
    
    # Pruning configuration
    pruner_min_resource: int = 1  # epochs/steps
    pruner_reduction_factor: int = 3
    
    # Success criteria threshold
    # Phase 1: TEP Pareto front dominates BP on ≥1 task across ≥3 seeds
    min_seeds_for_success: int = 3
    
    def __str__(self) -> str:
        return f"Phase {self.phase_number}: {self.name}"


# Phase configurations per specification
PHASE_CONFIGS: Dict[int, PhaseConfig] = {
    1: PhaseConfig(
        name="Rapid Signal Detection - Single Layer",
        phase_number=1,
        tasks=["digits_8x8"],  # Only 8x8 digits - XOR too simple
        n_trials_per_algorithm=300,  # 300-600 range, start lower
        target_trial_duration=(15.0, 45.0),
        trial_timeout_seconds=90.0,
        total_budget_hours=8.0,
        pruner_min_resource=1,
        pruner_reduction_factor=3,
        min_seeds_for_success=3,
    ),
    2: PhaseConfig(
        name="Validation on Medium Tasks",
        phase_number=2,
        tasks=["mnist_28x28", "cartpole_v1"],
        n_trials_per_algorithm=150,
        target_trial_duration=(60.0, 300.0),
        trial_timeout_seconds=600.0,
        total_budget_hours=18.0,
        pruner_min_resource=1,
        pruner_reduction_factor=3,
        min_seeds_for_success=5,
    ),
    3: PhaseConfig(
        name="Comprehensive Benchmarking",
        phase_number=3,
        tasks=["cifar10", "sequence_copy", "acrobot_v1"],
        n_trials_per_algorithm=100,
        target_trial_duration=(120.0, 600.0),
        trial_timeout_seconds=1800.0,
        total_budget_hours=48.0,
        pruner_min_resource=2,
        pruner_reduction_factor=3,
        min_seeds_for_success=5,
    ),
}

# Smoke test configuration (for quick validation)
SMOKE_TEST_CONFIG = PhaseConfig(
    name="Smoke Test",
    phase_number=0,
    tasks=["xor"],
    n_trials_per_algorithm=5,
    target_trial_duration=(5.0, 30.0),
    trial_timeout_seconds=60.0,
    total_budget_hours=0.1,  # 6 minutes
    pruner_min_resource=1,
    pruner_reduction_factor=2,
    min_seeds_for_success=2,
)


# =============================================================================
# VARIATION SPACE (CONFIGURABLE KNOBS)
# =============================================================================

@dataclass
class VariationKnobs:
    """All configurable knobs per specification.
    
    Start experiments with advanced features disabled; enable one at a time.
    """
    # Algorithm selection
    algorithm: Algorithm = Algorithm.TEP
    
    # Symmetry: Non-symmetric fixed based on prior evidence
    symmetric: bool = False
    
    # Core TEP parameters (sampled from search space)
    loop_radius: int = 4
    nudging_beta: float = 0.1
    dampening_gamma: float = 0.9
    equilibrium_iters: int = 20
    
    # Advanced features (disabled by default)
    memory_augmentation: str = "none"  # none, simple_ca, neural_ca, tiny_ntm
    time_dynamics: str = "discrete"    # discrete, continuous_ode
    contrastive_nudge: bool = False
    loop_topology: str = "toroidal"    # toroidal, radial_spokes


# =============================================================================
# REPORTING CONFIGURATION
# =============================================================================

@dataclass
class ReportingConfig:
    """Configuration for result reporting."""
    # Results to include
    top_n_configs: int = 5
    n_seeds_for_final: int = 5
    
    # Statistical tests
    significance_alpha: float = 0.05
    use_wilcoxon: bool = True
    use_bootstrap: bool = True
    n_bootstrap_samples: int = 10000
    
    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("tep_results"))
    
    # Visualization
    generate_pareto_plots: bool = True
    generate_training_curves: bool = True
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
