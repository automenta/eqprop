from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

@dataclass
class CostMetrics:
    """Cost metrics for a single trial."""
    wall_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    total_iterations: int = 0  # For EqProp: equilibration iterations
    param_count: int = 0
    flops_estimate: float = 0.0
    
    # Convergence tracking
    convergence_curve: List[float] = field(default_factory=list)  # Per-epoch performance
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        # Convert lists to JSON-serializable format
        return d


@dataclass
class HyperOptTrial:
    """A single hyperparameter optimization trial."""
    trial_id: str
    algorithm: str  # "eqprop" or "bp"
    config: Dict[str, Any]
    task: str  # e.g., "mnist", "cartpole"
    seed: int
    
    # Results
    performance: float = 0.0
    performance_metric: str = "accuracy"  # or "reward"
    cost: CostMetrics = field(default_factory=CostMetrics)
    
    # Meta
    status: str = "pending"  # pending, running, complete, failed
    timestamp: str = ""
    log_path: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "trial_id": self.trial_id,
            "algorithm": self.algorithm,
            "config": self.config,
            "task": self.task,
            "seed": self.seed,
            "performance": self.performance,
            "performance_metric": self.performance_metric,
            "cost": self.cost.to_dict(),
            "status": self.status,
            "timestamp": self.timestamp,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "HyperOptTrial":
        cost = CostMetrics(**d.get("cost", {}))
        return cls(
            trial_id=d["trial_id"],
            algorithm=d["algorithm"],
            config=d.get("config", {}),
            task=d.get("task", "unknown"),
            seed=d.get("seed", 0),
            performance=d.get("performance", 0.0),
            performance_metric=d.get("performance_metric", "accuracy"),
            cost=cost,
            status=d.get("status", "pending"),
            timestamp=d.get("timestamp", ""),
            log_path=d.get("log_path", ""),
            error=d.get("error", ""),
        )
