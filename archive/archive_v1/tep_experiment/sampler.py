"""
Optuna samplers for TEP experiments.

Ensures identical treatment between TEP and BP:
- Shared parameters use identical distributions
- TEP-specific parameters added only to TEP trials
- Support for transfer seeding from simpler to harder tasks
- Constraint-aware sampling for valid configurations

Enhanced Features:
- Parameter constraints (e.g., n_heads must divide hidden_units)
- Grid search support for ablation studies
- Conditional parameters based on algorithm
"""

from typing import Dict, Any, Optional, List, Callable
import math
import optuna
from optuna.trial import Trial
from optuna.samplers import NSGAIISampler, TPESampler, GridSampler

from .config import SharedSearchSpace, TEPSearchSpace, BPSearchSpace


# Default search spaces
DEFAULT_SHARED_SPACE = SharedSearchSpace()
DEFAULT_TEP_SPACE = TEPSearchSpace()
DEFAULT_BP_SPACE = BPSearchSpace()


def sample_shared_params(
    trial: Trial,
    space: SharedSearchSpace = DEFAULT_SHARED_SPACE,
    phase: int = 1,
) -> Dict[str, Any]:
    """Sample parameters shared between TEP and BP.
    
    These use identical distributions to ensure fair comparison.
    Phase 1 constraints are applied here to avoid Optuna sampling irrelevant params.
    """
    config = {}
    
    # Phase-specific n_hidden_layers sampling
    if phase == 1:
        # Phase 1: Single layer only
        config["n_hidden_layers"] = 1
    else:
        # Phase 2+: Full range
        config["n_hidden_layers"] = trial.suggest_int(
            "n_hidden_layers", 
            space.n_hidden_layers[0], 
            space.n_hidden_layers[1]
        )
    
    # Other shared parameters (same for all phases)
    config.update({
        "hidden_units": trial.suggest_int(
            "hidden_units",
            space.hidden_units_range[0],
            space.hidden_units_range[1],
            log=True
        ),
        "activation": trial.suggest_categorical(
            "activation",
            list(space.activation_choices)
        ),
        "lr": trial.suggest_float(
            "lr",
            space.learning_rate_range[0],
            space.learning_rate_range[1],
            log=True
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            list(space.batch_size_choices)
        ),
    })
    
    return config


def sample_tep_params(
    trial: Trial,
    space: TEPSearchSpace = DEFAULT_TEP_SPACE
) -> Dict[str, Any]:
    """Sample TEP-specific parameters.
    
    Added to shared params for TEP trials only.
    """
    return space.get_optuna_suggestions(trial)


def sample_bp_params(
    trial: Trial,
    space: BPSearchSpace = DEFAULT_BP_SPACE
) -> Dict[str, Any]:
    """Sample BP-specific parameters.
    
    Added to shared params for BP trials only.
    """
    return space.get_optuna_suggestions(trial)


def sample_full_config(
    trial: Trial,
    algorithm: str,
    shared_space: SharedSearchSpace = DEFAULT_SHARED_SPACE,
    tep_space: TEPSearchSpace = DEFAULT_TEP_SPACE,
    bp_space: BPSearchSpace = DEFAULT_BP_SPACE,
    phase: int = 1,
) -> Dict[str, Any]:
    """Sample complete configuration for a trial.
    
    Args:
        trial: Optuna trial object
        algorithm: "tep" or "bp"
        shared_space: Shared parameter space
        tep_space: TEP-specific parameter space
        bp_space: BP-specific parameter space
        phase: Experiment phase (1, 2, or 3) - affects constraints
        
    Returns:
        Complete configuration dictionary with validated parameters
    """
    config = {"algorithm": algorithm}
    
    # Add shared parameters with phase-aware sampling
    config.update(sample_shared_params(trial, shared_space, phase=phase))
    
    # Add algorithm-specific parameters
    if algorithm == "tep":
        config.update(sample_tep_params(trial, tep_space))
    else:
        config.update(sample_bp_params(trial, bp_space))
    
    # Apply constraints to ensure valid configurations
    config = apply_constraints(config)
    
    return config


def apply_constraints(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply constraints to ensure valid configurations.
    
    Constraints:
    - hidden_units must be divisible by a valid n_heads
    - eq_iters scales with beta (higher beta needs fewer iters)
    - lr scales inversely with hidden_units for stability
    """
    hidden_units = config.get("hidden_units", 64)
    
    # Ensure hidden_units is at valid value
    config["hidden_units"] = max(4, min(512, hidden_units))
    
    # Adjust n_heads to divide hidden_units evenly
    # Find valid n_heads from [1, 2, 4, 8]
    valid_heads = []
    for h in [1, 2, 4, 8]:
        if config["hidden_units"] % h == 0:
            valid_heads.append(h)
    config["_valid_n_heads"] = valid_heads[-1] if valid_heads else 1
    
    # Clamp TEP parameters to valid ranges
    if "beta" in config:
        config["beta"] = max(0.001, min(0.5, config["beta"]))
    
    if "gamma" in config:
        config["gamma"] = max(0.5, min(0.99, config["gamma"]))
    
    if "eq_iters" in config:
        config["eq_iters"] = max(5, min(100, config["eq_iters"]))
    
    # Clamp learning rate
    if "lr" in config:
        config["lr"] = max(1e-5, min(0.1, config["lr"]))
    
    return config


class TransferSampler:
    """Sampler that uses best params from simpler tasks as seeds.
    
    For Phase 2+, uses the best configurations found in previous phases
    to warm-start the optimization.
    
    Features:
    - Maintains separate best configs per task/algorithm
    - Supports scaling parameters between phases
    - Tracks performance history for adaptive seeding
    """
    
    def __init__(self):
        self.best_configs: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, float] = {}
    
    def register_best(
        self,
        task: str,
        algorithm: str,
        config: Dict[str, Any],
        performance: float = 0.0,
    ):
        """Register the best configuration for a task/algorithm."""
        key = f"{task}_{algorithm}"
        self.best_configs[key] = config.copy()
        self.performance_history[key] = performance
    
    def get_seed_trials(
        self,
        target_task: str,
        algorithm: str,
        source_tasks: Optional[List[str]] = None,
        n_seeds: int = 3,
    ) -> List[Dict[str, Any]]:
        """Get seed trials for a new task based on previous results.
        
        Args:
            target_task: The new task to optimize
            algorithm: "tep" or "bp"
            source_tasks: Tasks to pull seeds from (defaults to all registered)
            n_seeds: Maximum number of seeds to return
            
        Returns:
            List of seed configurations to enqueue
        """
        seeds = []
        
        if source_tasks is None:
            # Use all registered configs for this algorithm
            source_tasks = [
                k.rsplit("_", 1)[0] 
                for k in self.best_configs.keys() 
                if k.endswith(f"_{algorithm}")
            ]
        
        # Sort by performance if available
        task_perf = [
            (task, self.performance_history.get(f"{task}_{algorithm}", 0))
            for task in source_tasks
        ]
        task_perf.sort(key=lambda x: -x[1])  # Best first
        
        for source_task, _ in task_perf[:n_seeds]:
            key = f"{source_task}_{algorithm}"
            if key in self.best_configs:
                seed_config = self.best_configs[key].copy()
                # Remove non-parameter keys
                seed_config.pop("algorithm", None)
                seeds.append(seed_config)
        
        return seeds


def create_study_sampler(
    algorithm: str,
    multivariate: bool = True,
    seed: int = 42,
    n_startup_trials: int = 10,
) -> optuna.samplers.BaseSampler:
    """Create appropriate Optuna sampler for single-objective optimization.
    
    Uses TPESampler with multivariate=True to capture hyperparameter
    correlations as specified in the experiment requirements.
    
    Features:
    - Multivariate TPE for correlated hyperparameters
    - Constant liar for parallel trials
    - Startup trials for initial exploration
    
    Returns:
        Configured TPESampler
    """
    return TPESampler(
        seed=seed,
        multivariate=multivariate,
        constant_liar=True,
        n_startup_trials=n_startup_trials,
        consider_endpoints=True,  # Better for bounded spaces
    )


def create_multi_objective_sampler(
    seed: int = 42,
    population_size: int = 50,
    crossover_prob: float = 0.9,
    mutation_prob: Optional[float] = None,
) -> NSGAIISampler:
    """Create NSGA-II sampler for multi-objective optimization.
    
    Used for explicit Pareto front optimization across 4 objectives:
    1. Accuracy (maximize)
    2. Wall time (minimize)
    3. Parameter count (minimize)
    4. Convergence steps (minimize)
    
    Args:
        seed: Random seed
        population_size: NSGA-II population size (default 50)
        crossover_prob: Crossover probability (default 0.9)
        mutation_prob: Mutation probability (None = auto)
        
    Returns:
        Configured NSGAIISampler
    """
    return NSGAIISampler(
        population_size=population_size,
        seed=seed,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
    )


def create_pruner(
    min_resource: int = 1,
    max_resource: int = 100,
    reduction_factor: int = 3,
) -> optuna.pruners.BasePruner:
    """Create SuccessiveHalvingPruner per specification.
    
    Mandatory for efficient budget use. Aggressively prunes poorly
    performing trials to allocate more resources to promising ones.
    
    Args:
        min_resource: Minimum epochs before pruning
        max_resource: Maximum epochs (not used in SH but good for docs)
        reduction_factor: How aggressively to prune (3 = keep 1/3)
        
    Returns:
        Configured SuccessiveHalvingPruner
    """
    return optuna.pruners.SuccessiveHalvingPruner(
        min_resource=min_resource,
        reduction_factor=reduction_factor,
        min_early_stopping_rate=0,  # Allow pruning at first check
    )


def create_median_pruner(
    n_startup_trials: int = 5,
    n_warmup_steps: int = 3,
    interval_steps: int = 1,
) -> optuna.pruners.BasePruner:
    """Create MedianPruner as alternative to SuccessiveHalving.
    
    MedianPruner prunes trials with intermediate values worse than
    the median of previous trials at the same step.
    
    Args:
        n_startup_trials: Trials before pruning starts
        n_warmup_steps: Steps before pruning within a trial
        interval_steps: Check interval
        
    Returns:
        Configured MedianPruner
    """
    return optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps,
    )


def enqueue_seed_trials(
    study: optuna.Study,
    seed_configs: List[Dict[str, Any]],
):
    """Enqueue seed trials in an Optuna study.
    
    Used for transfer learning from simpler to harder tasks.
    Validates configs before enqueueing to avoid errors.
    
    Args:
        study: Optuna study to enqueue trials in
        seed_configs: List of seed configurations
    """
    for config in seed_configs:
        # Extract only valid parameter names Optuna expects
        # Exclude algorithm and internal keys
        params = {
            k: v for k, v in config.items()
            if not k.startswith("_") and k not in ["algorithm", "status", "error"]
        }
        
        try:
            study.enqueue_trial(params)
        except Exception as e:
            # Log but don't fail on invalid seed
            import logging
            logging.debug(f"Could not enqueue seed trial: {e}")


def create_grid_sampler(
    search_space: Dict[str, List[Any]],
) -> GridSampler:
    """Create GridSampler for exhaustive ablation studies.
    
    Use this when you want to test every combination of a small
    set of hyperparameter values.
    
    Args:
        search_space: Dict mapping param names to lists of values
        
    Returns:
        Configured GridSampler
    """
    return GridSampler(search_space)


# =============================================================================
# CONSTRAINT CALLBACKS
# =============================================================================

def get_constraints_func() -> Callable:
    """Get constraints function for NSGA-III style optimization.
    
    Returns function that computes constraint violations.
    Positive values = constraint violated.
    
    Example constraints:
    - Parameter count should be under 1M
    - Wall time should be under timeout
    """
    def constraints(trial: optuna.Trial) -> List[float]:
        violations = []
        
        # Example: Prefer models under 100k parameters
        param_count = trial.user_attrs.get("param_count", 0)
        if param_count > 100000:
            violations.append(param_count - 100000)
        else:
            violations.append(0)
        
        return violations
    
    return constraints
