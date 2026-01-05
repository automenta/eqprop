"""Core per-iteration analyzer and report generation."""

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import json
import numpy as np

from .trajectory import TrajectoryRecorder, Trajectory
from .metrics import (
    IterationMetrics, AggregateMetrics,
    compute_iteration_metrics, compute_aggregate_metrics
)
from .theoretical import TheoreticalValidator, TheoreticalValidation
from .synthetic_tasks import SyntheticTask, get_all_tasks


@dataclass
class AnalysisReport:
    """Complete analysis report for a model on a task."""
    
    model_name: str
    task_name: str
    
    # Per-iteration data
    trajectory: Trajectory
    iteration_metrics: List[IterationMetrics]
    aggregate_metrics: AggregateMetrics
    
    # Theoretical validation
    theoretical: Optional[TheoreticalValidation] = None
    
    # Task performance
    task_accuracy: Optional[float] = None
    task_success: bool = False
    
    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "convergence": {
                "converged": self.trajectory.converged,
                "steps": self.trajectory.convergence_step,
                "final_residual": self.trajectory.final_residual
            },
            "aggregate": {
                "total_steps": self.aggregate_metrics.total_steps,
                "convergence_rate": self.aggregate_metrics.convergence_rate,
                "energy_monotonic": self.aggregate_metrics.energy_monotonic,
                "energy_violations": self.aggregate_metrics.energy_violations
            },
            "theoretical": {
                "energy_descent": self.theoretical.energy_descent_valid if self.theoretical else None,
                "contraction": self.theoretical.contraction_valid if self.theoretical else None,
                "lipschitz": self.theoretical.lipschitz_constant if self.theoretical else None,
                "gradient_equiv": self.theoretical.gradient_equivalence if self.theoretical else None,
                "spectral_radius": self.theoretical.spectral_radius if self.theoretical else None
            },
            "task": {
                "accuracy": self.task_accuracy,
                "success": self.task_success
            },
            "config": self.config
        }
    
    def to_json(self, path: str = None) -> str:
        """Export to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str
    
    def to_markdown(self) -> str:
        """Generate markdown summary."""
        lines = [
            f"# Analysis Report: {self.model_name} on {self.task_name}",
            "",
            "## Convergence",
            f"- **Converged**: {'✓' if self.trajectory.converged else '✗'}",
            f"- **Steps**: {self.trajectory.convergence_step or self.trajectory.num_steps}",
            f"- **Final Residual**: {self.trajectory.final_residual:.6f}",
            f"- **Convergence Rate**: {self.aggregate_metrics.convergence_rate:.4f}",
            "",
            "## Energy Analysis",
            f"- **Initial Energy**: {self.aggregate_metrics.initial_energy:.4f}",
            f"- **Final Energy**: {self.aggregate_metrics.final_energy:.4f}",
            f"- **Monotonic Descent**: {'✓' if self.aggregate_metrics.energy_monotonic else '✗'}",
            f"- **Violations**: {self.aggregate_metrics.energy_violations}",
            ""
        ]
        
        if self.theoretical:
            lines.extend([
                "## Theoretical Guarantees",
                f"- **Energy Descent**: {'✓' if self.theoretical.energy_descent_valid else '✗'}",
                f"- **Contraction (L<1)**: {'✓' if self.theoretical.contraction_valid else '✗'} "
                f"(L={self.theoretical.lipschitz_constant:.4f})",
                f"- **Gradient Equivalence**: {self.theoretical.gradient_equivalence:.4f}",
                f"- **Spectral Radius**: {self.theoretical.spectral_radius:.4f}",
                ""
            ])
        
        if self.task_accuracy is not None:
            lines.extend([
                "## Task Performance",
                f"- **Accuracy**: {self.task_accuracy:.2%}",
                f"- **Success**: {'✓' if self.task_success else '✗'}",
                ""
            ])
        
        return "\n".join(lines)
    
    def summary(self) -> str:
        """One-line summary."""
        status = "✓" if self.task_success and (not self.theoretical or 
                 (self.theoretical.contraction_valid and self.theoretical.energy_descent_valid)) else "✗"
        return (f"{status} {self.model_name} on {self.task_name}: "
                f"conv={self.trajectory.converged}, "
                f"steps={self.trajectory.convergence_step or 'N/A'}, "
                f"acc={self.task_accuracy:.2%}" if self.task_accuracy else "N/A")


class IterationAnalyzer:
    """Main analyzer for per-iteration model validation."""
    
    def __init__(self, model, device: str = "cpu", epsilon: float = 1e-4, max_steps: int = 100):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.max_steps = max_steps
        
        if hasattr(model, 'to'):
            model.to(device)
        
        self.recorder = TrajectoryRecorder(model, epsilon, max_steps)
        self.validator = TheoreticalValidator(model, device)
    
    def analyze(self, x: Tensor, y: Tensor = None, 
                validate_theory: bool = True) -> AnalysisReport:
        """Run complete analysis on a single batch.
        
        Args:
            x: Input tensor
            y: Optional target labels
            validate_theory: Whether to run theoretical validation
            
        Returns:
            AnalysisReport with all metrics
        """
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
        
        # 1. Record trajectory
        trajectory = self.recorder.record(x)
        
        # 2. Compute per-iteration metrics
        iter_metrics = compute_iteration_metrics(
            trajectory.states, trajectory.energies, self.model, x
        )
        
        # 3. Compute aggregate metrics
        agg_metrics = compute_aggregate_metrics(
            iter_metrics, trajectory.converged, trajectory.convergence_step
        )
        
        # 4. Theoretical validation
        theoretical = None
        if validate_theory and y is not None:
            theoretical = self.validator.validate_all(x, y, trajectory.energies)
        
        # 5. Task accuracy
        task_acc = None
        if y is not None:
            with torch.no_grad():
                output = self.model(x, steps=self.max_steps)
                preds = output.argmax(dim=-1)
                task_acc = (preds == y).float().mean().item()
        
        model_name = self.model.__class__.__name__
        
        return AnalysisReport(
            model_name=model_name,
            task_name="custom",
            trajectory=trajectory,
            iteration_metrics=iter_metrics,
            aggregate_metrics=agg_metrics,
            theoretical=theoretical,
            task_accuracy=task_acc,
            task_success=task_acc > 0.9 if task_acc else False
        )
    
    def analyze_task(self, task: SyntheticTask, n_samples: int = 100,
                     validate_theory: bool = True) -> AnalysisReport:
        """Run analysis on a synthetic task.
        
        Args:
            task: SyntheticTask instance
            n_samples: Number of samples to use
            validate_theory: Whether to run theoretical validation
            
        Returns:
            AnalysisReport
        """
        X, Y = task.generate_data(n_samples, self.device)
        
        report = self.analyze(X, Y, validate_theory)
        report.task_name = task.name
        report.task_success = task.success_criterion(
            self.model(X, steps=self.max_steps), Y
        )
        
        return report
    
    def analyze_all_tasks(self, n_samples: int = 100) -> Dict[str, AnalysisReport]:
        """Run analysis on all synthetic tasks."""
        tasks = get_all_tasks()
        reports = {}
        
        for name, task in tasks.items():
            # Verify dimensions match
            if hasattr(self.model, 'input_dim') and self.model.input_dim != task.input_dim:
                print(f"  Skipping {name}: input_dim mismatch "
                      f"({self.model.input_dim} vs {task.input_dim})")
                continue
            
            print(f"  Analyzing {name}...")
            try:
                report = self.analyze_task(task, n_samples)
                reports[name] = report
                print(f"    {report.summary()}")
            except Exception as e:
                print(f"    Error: {e}")
        
        return reports


def run_model_comparison(models: Dict[str, Any], task: SyntheticTask,
                         n_samples: int = 100, device: str = "cpu") -> Dict[str, AnalysisReport]:
    """Compare multiple models on the same task.
    
    Args:
        models: Dict of model_name -> model instance
        task: SyntheticTask to evaluate on
        n_samples: Number of samples
        device: Device to use
        
    Returns:
        Dict of model_name -> AnalysisReport
    """
    reports = {}
    
    for name, model in models.items():
        print(f"Analyzing {name}...")
        analyzer = IterationAnalyzer(model, device)
        report = analyzer.analyze_task(task, n_samples)
        report.model_name = name
        reports[name] = report
        print(f"  {report.summary()}")
    
    return reports
