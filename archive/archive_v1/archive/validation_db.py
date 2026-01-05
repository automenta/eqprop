#!/usr/bin/env python3
"""
Validation Database - Stores and queries experiment results.

Provides:
- ExperimentRun dataclass for structured results
- ValidationDB for persistence and querying
- Gap detection for scheduling
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import hashlib


@dataclass
class ExperimentRun:
    """A single experiment run with all metadata."""
    
    # Identity
    experiment_id: str          # e.g., "rl_cartpole_eqprop_seed0"
    algorithm: str              # "eqprop" | "bp"
    environment: str            # "CartPole-v1"
    seed: int
    timestamp: str              # ISO format
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    model_params: int = 0       # Total parameter count
    
    # Results
    primary_metric: float = 0.0          # e.g., avg_reward
    secondary_metrics: Dict[str, float] = field(default_factory=dict)
    solved: bool = False
    
    # Fairness tracking
    walltime_seconds: float = 0.0
    gpu_memory_mb: float = 0.0
    iterations_total: int = 0
    
    # Status
    status: str = "pending"     # "pending" | "running" | "complete" | "failed"
    log_path: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRun":
        return cls(**d)
    
    @property
    def key(self) -> Tuple[str, str, int]:
        """Unique key: (environment, algorithm, seed)."""
        return (self.environment, self.algorithm, self.seed)


@dataclass
class ValidationMatrix:
    """Tracks validation progress across dimensions."""
    
    environments: List[str]
    algorithms: List[str]
    seeds: List[int]
    
    completed: Set[Tuple[str, str, int]] = field(default_factory=set)
    
    @property
    def required(self) -> Set[Tuple[str, str, int]]:
        """All required (env, algo, seed) combinations."""
        result = set()
        for env in self.environments:
            for algo in self.algorithms:
                for seed in self.seeds:
                    result.add((env, algo, seed))
        return result
    
    @property
    def gaps(self) -> Set[Tuple[str, str, int]]:
        """Missing experiments."""
        return self.required - self.completed
    
    @property
    def progress(self) -> float:
        """Completion percentage."""
        total = len(self.required)
        if total == 0:
            return 1.0
        return len(self.completed) / total
    
    def is_environment_complete(self, env: str) -> bool:
        """Check if all seeds for both algorithms are done for this env."""
        for algo in self.algorithms:
            for seed in self.seeds:
                if (env, algo, seed) not in self.completed:
                    return False
        return True
    
    def get_env_progress(self, env: str) -> Tuple[int, int]:
        """Get (completed, total) for an environment."""
        total = len(self.algorithms) * len(self.seeds)
        completed = sum(1 for algo in self.algorithms for seed in self.seeds 
                       if (env, algo, seed) in self.completed)
        return completed, total


class ValidationDB:
    """Persistent database for validation experiments."""
    
    def __init__(self, db_path: str = "data/validation_results.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs: Dict[str, ExperimentRun] = {}
        self._load()
    
    def _load(self):
        """Load database from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    data = json.load(f)
                for run_data in data.get("runs", []):
                    run = ExperimentRun.from_dict(run_data)
                    self.runs[run.experiment_id] = run
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load database: {e}")
                self.runs = {}
    
    def _save(self):
        """Save database to disk."""
        data = {
            "last_updated": datetime.now().isoformat(),
            "runs": [run.to_dict() for run in self.runs.values()]
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_run(self, run: ExperimentRun):
        """Add or update an experiment run."""
        self.runs[run.experiment_id] = run
        self._save()
    
    def get_run(self, experiment_id: str) -> Optional[ExperimentRun]:
        """Get a specific run by ID."""
        return self.runs.get(experiment_id)
    
    def get_runs(self, 
                 environment: Optional[str] = None,
                 algorithm: Optional[str] = None,
                 status: Optional[str] = None) -> List[ExperimentRun]:
        """Query runs with filters."""
        results = []
        for run in self.runs.values():
            if environment and run.environment != environment:
                continue
            if algorithm and run.algorithm != algorithm:
                continue
            if status and run.status != status:
                continue
            results.append(run)
        return results
    
    def get_completed_runs(self, environment: str, algorithm: str) -> List[ExperimentRun]:
        """Get all completed runs for an (env, algo) pair."""
        return [r for r in self.runs.values() 
                if r.environment == environment 
                and r.algorithm == algorithm 
                and r.status == "complete"]
    
    def get_validation_matrix(self, 
                              environments: List[str],
                              algorithms: List[str],
                              seeds: List[int]) -> ValidationMatrix:
        """Get validation matrix with completion status."""
        matrix = ValidationMatrix(
            environments=environments,
            algorithms=algorithms,
            seeds=seeds
        )
        
        for run in self.runs.values():
            if run.status == "complete":
                matrix.completed.add(run.key)
        
        return matrix
    
    def get_metrics_for_comparison(self, 
                                    environment: str,
                                    metric: str = "primary_metric") -> Dict[str, List[float]]:
        """Get metrics for each algorithm for statistical comparison."""
        result = {"eqprop": [], "bp": []}
        
        for run in self.runs.values():
            if run.environment != environment or run.status != "complete":
                continue
            
            if metric == "primary_metric":
                value = run.primary_metric
            else:
                value = run.secondary_metrics.get(metric, 0)
            
            if run.algorithm in result:
                result[run.algorithm].append(value)
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get database summary statistics."""
        total = len(self.runs)
        by_status = {}
        by_env = {}
        by_algo = {}
        
        for run in self.runs.values():
            by_status[run.status] = by_status.get(run.status, 0) + 1
            by_env[run.environment] = by_env.get(run.environment, 0) + 1
            by_algo[run.algorithm] = by_algo.get(run.algorithm, 0) + 1
        
        return {
            "total_runs": total,
            "by_status": by_status,
            "by_environment": by_env,
            "by_algorithm": by_algo
        }
    
    def clear(self):
        """Clear all runs (for testing)."""
        self.runs = {}
        self._save()


def generate_experiment_id(environment: str, algorithm: str, seed: int) -> str:
    """Generate unique experiment ID."""
    return f"{environment.lower().replace('-', '_')}_{algorithm}_seed{seed}"


# Self-test
if __name__ == "__main__":
    print("Testing ValidationDB...")
    
    db = ValidationDB("data/test_validation.json")
    db.clear()
    
    # Add some test runs
    for seed in range(3):
        for algo in ["eqprop", "bp"]:
            run = ExperimentRun(
                experiment_id=generate_experiment_id("CartPole-v1", algo, seed),
                algorithm=algo,
                environment="CartPole-v1",
                seed=seed,
                timestamp=datetime.now().isoformat(),
                primary_metric=300.0 + seed * 10 if algo == "eqprop" else 180.0 + seed * 5,
                status="complete"
            )
            db.add_run(run)
    
    # Test queries
    matrix = db.get_validation_matrix(
        environments=["CartPole-v1", "Acrobot-v1"],
        algorithms=["eqprop", "bp"],
        seeds=list(range(10))
    )
    
    print(f"Progress: {matrix.progress:.1%}")
    print(f"Gaps: {len(matrix.gaps)} experiments remaining")
    print(f"CartPole complete: {matrix.is_environment_complete('CartPole-v1')}")
    
    metrics = db.get_metrics_for_comparison("CartPole-v1")
    print(f"EqProp metrics: {metrics['eqprop']}")
    print(f"BP metrics: {metrics['bp']}")
    
    print("\n✅ ValidationDB tests passed!")
