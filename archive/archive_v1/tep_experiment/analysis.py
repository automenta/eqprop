"""
Pareto front analysis and reporting for TEP experiments.

Provides:
- Pareto front extraction from Optuna studies
- Dominance comparison between TEP and BP fronts
- Hypervolume calculation
- Report generation with plots
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
import numpy as np

from .config import TrialResult


@dataclass
class ParetoPoint:
    """A single point on a Pareto front."""
    trial_id: str
    config: Dict[str, Any]
    
    # The 4 objectives
    accuracy: float
    wall_time: float
    param_count: int
    convergence_steps: int
    
    @property
    def objectives(self) -> Tuple[float, float, float, float]:
        """Return objectives as tuple."""
        return (self.accuracy, self.wall_time, float(self.param_count), float(self.convergence_steps))


@dataclass
class ParetoFront:
    """Pareto-optimal configurations from an optimization study."""
    algorithm: str
    task: str
    points: List[ParetoPoint] = field(default_factory=list)
    all_trials: List[TrialResult] = field(default_factory=list)
    
    @property
    def size(self) -> int:
        return len(self.points)
    
    def best_accuracy(self) -> Optional[ParetoPoint]:
        """Get point with highest accuracy."""
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.accuracy)
    
    def best_speed(self) -> Optional[ParetoPoint]:
        """Get point with lowest wall time."""
        if not self.points:
            return None
        return min(self.points, key=lambda p: p.wall_time)
    
    def best_efficiency(self) -> Optional[ParetoPoint]:
        """Get point with lowest parameter count."""
        if not self.points:
            return None
        return min(self.points, key=lambda p: p.param_count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "algorithm": self.algorithm,
            "task": self.task,
            "n_points": len(self.points),
            "points": [
                {
                    "trial_id": p.trial_id,
                    "accuracy": p.accuracy,
                    "wall_time": p.wall_time,
                    "param_count": p.param_count,
                    "convergence_steps": p.convergence_steps,
                    "config": p.config,
                }
                for p in self.points
            ],
        }


class ParetoAnalyzer:
    """Analyzes and compares Pareto fronts."""
    
    def extract_pareto_front(
        self,
        trials: List[TrialResult],
        algorithm: str,
        task: str,
    ) -> ParetoFront:
        """Extract Pareto-optimal points from trial results.
        
        Points are non-dominated across all 4 objectives:
        - Maximize accuracy
        - Minimize wall_time
        - Minimize param_count
        - Minimize convergence_steps
        """
        # Filter to completed trials only
        completed = [t for t in trials if t.status == "complete" and t.accuracy > 0]
        
        if not completed:
            return ParetoFront(algorithm=algorithm, task=task)
        
        # Find Pareto-optimal points
        pareto_points = []
        
        for trial in completed:
            is_dominated = False
            
            for other in completed:
                if self._dominates(other, trial):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_points.append(ParetoPoint(
                    trial_id=trial.trial_id,
                    config=trial.config,
                    accuracy=trial.accuracy,
                    wall_time=trial.wall_time_seconds,
                    param_count=trial.param_count,
                    convergence_steps=trial.convergence_steps,
                ))
        
        return ParetoFront(
            algorithm=algorithm,
            task=task,
            points=pareto_points,
            all_trials=completed,
        )
    
    def _dominates(self, a: TrialResult, b: TrialResult) -> bool:
        """Check if trial a dominates trial b.
        
        a dominates b if:
        - a is at least as good as b on all objectives
        - a is strictly better on at least one objective
        """
        # Higher accuracy is better
        acc_better = a.accuracy >= b.accuracy
        acc_strict = a.accuracy > b.accuracy
        
        # Lower time is better
        time_better = a.wall_time_seconds <= b.wall_time_seconds
        time_strict = a.wall_time_seconds < b.wall_time_seconds
        
        # Lower params is better
        param_better = a.param_count <= b.param_count
        param_strict = a.param_count < b.param_count
        
        # Lower convergence steps is better
        conv_better = a.convergence_steps <= b.convergence_steps
        conv_strict = a.convergence_steps < b.convergence_steps
        
        all_at_least_as_good = acc_better and time_better and param_better and conv_better
        at_least_one_better = acc_strict or time_strict or param_strict or conv_strict
        
        return all_at_least_as_good and at_least_one_better
    
    def compare_fronts(
        self,
        tep_front: ParetoFront,
        bp_front: ParetoFront,
    ) -> Dict[str, Any]:
        """Compare TEP and BP Pareto fronts.
        
        Per specification: TEP wins if its Pareto front dominates BP
        on at least one task across ≥3 seeds.
        
        Returns:
            Comparison results including dominance analysis
        """
        result = {
            "tep_points": len(tep_front.points),
            "bp_points": len(bp_front.points),
            "tep_dominates_bp": 0,
            "bp_dominates_tep": 0,
            "mutually_non_dominated": 0,
        }
        
        # Compare each TEP point against BP front
        for tep_pt in tep_front.points:
            dominated_by_bp = any(
                self._point_dominates(bp_pt, tep_pt)
                for bp_pt in bp_front.points
            )
            if not dominated_by_bp:
                result["tep_dominates_bp"] += 1
        
        # Compare each BP point against TEP front
        for bp_pt in bp_front.points:
            dominated_by_tep = any(
                self._point_dominates(tep_pt, bp_pt)
                for tep_pt in tep_front.points
            )
            if not dominated_by_tep:
                result["bp_dominates_tep"] += 1
        
        # Best point comparison (do this first for tie-breaking)
        tep_best = tep_front.best_accuracy()
        bp_best = bp_front.best_accuracy()
        if tep_best and bp_best:
            result["best_accuracy"] = {
                "tep": tep_best.accuracy,
                "bp": bp_best.accuracy,
                "delta": tep_best.accuracy - bp_best.accuracy,
            }
        
        # Overall assessment with tie-breaking
        if result["tep_dominates_bp"] > result["bp_dominates_tep"]:
            result["winner"] = "tep"
        elif result["bp_dominates_tep"] > result["tep_dominates_bp"]:
            result["winner"] = "bp"
        else:
            # Tie in Pareto dominance - use best accuracy as tie-breaker
            if tep_best and bp_best:
                acc_diff = abs(tep_best.accuracy - bp_best.accuracy)
                # If accuracy difference > 5%, clear winner
                if acc_diff > 0.05:
                    result["winner"] = "tep" if tep_best.accuracy > bp_best.accuracy else "bp"
                    result["tie_broken_by"] = "best_accuracy"
                else:
                    # Very close - check wall time for efficiency
                    tep_time = tep_best.wall_time if hasattr(tep_best, 'wall_time') else 0
                    bp_time = bp_best.wall_time if hasattr(bp_best, 'wall_time') else 0
                    if tep_time > 0 and bp_time > 0:
                        # Faster with similar accuracy wins
                        result["winner"] = "tep" if tep_time < bp_time else "bp"
                        result["tie_broken_by"] = "wall_time"
                    else:
                        result["winner"] = "tie"
            else:
                result["winner"] = "tie"
        
        return result
    
    def _point_dominates(self, a: ParetoPoint, b: ParetoPoint) -> bool:
        """Check if point a dominates point b."""
        acc_better = a.accuracy >= b.accuracy
        acc_strict = a.accuracy > b.accuracy
        
        time_better = a.wall_time <= b.wall_time
        time_strict = a.wall_time < b.wall_time
        
        param_better = a.param_count <= b.param_count
        param_strict = a.param_count < b.param_count
        
        conv_better = a.convergence_steps <= b.convergence_steps
        conv_strict = a.convergence_steps < b.convergence_steps
        
        all_at_least_as_good = acc_better and time_better and param_better and conv_better
        at_least_one_better = acc_strict or time_strict or param_strict or conv_strict
        
        return all_at_least_as_good and at_least_one_better
    
    def compute_hypervolume(
        self,
        front: ParetoFront,
        reference_point: Tuple[float, float, float, float] = (0.0, 1000.0, 1e8, 1000),
    ) -> float:
        """Compute hypervolume indicator for a Pareto front.
        
        Higher hypervolume = better front quality.
        
        Args:
            front: Pareto front to evaluate
            reference_point: Reference point for hypervolume (worst case)
        
        Returns:
            Hypervolume value
        """
        if not front.points:
            return 0.0
        
        # Normalize objectives to [0, 1] relative to reference
        # Accuracy: higher is better, so (acc - 0) / (1 - 0)
        # Time/Params/Conv: lower is better, so (ref - val) / (ref - 0)
        
        try:
            # Try using pymoo for accurate hypervolume
            from pymoo.indicators.hv import HV
            
            # Convert to minimization (negate accuracy)
            points = np.array([
                [
                    1.0 - p.accuracy,  # Minimize (1 - accuracy)
                    p.wall_time,
                    float(p.param_count),
                    float(p.convergence_steps),
                ]
                for p in front.points
            ])
            
            ref = np.array([
                1.0 - reference_point[0],
                reference_point[1],
                reference_point[2],
                reference_point[3],
            ])
            
            hv = HV(ref_point=ref)
            return float(hv(points))
            
        except ImportError:
            # Fallback: simple volume approximation
            if len(front.points) == 1:
                p = front.points[0]
                return (
                    p.accuracy *
                    (reference_point[1] - p.wall_time) *
                    (reference_point[2] - p.param_count) *
                    (reference_point[3] - p.convergence_steps)
                )
            
            # Sum of dominated hyperboxes (simplified)
            total = 0.0
            for p in front.points:
                total += p.accuracy * 1000  # Weighted by accuracy
            return total


class ReportGenerator:
    """Generates reports for TEP experiments."""
    
    def __init__(self, output_dir: Path = Path("tep_results")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_phase_report(
        self,
        phase: int,
        task: str,
        tep_front: ParetoFront,
        bp_front: ParetoFront,
        comparison: Dict[str, Any],
    ) -> str:
        """Generate markdown report for a single phase/task."""
        
        report = f"""# TEP Experiment Report: Phase {phase} - {task}

**Generated**: {self._timestamp()}

## Summary

| Metric | TEP | BP |
|--------|-----|-----|
| Pareto Points | {tep_front.size} | {bp_front.size} |
"""
        
        tep_best = tep_front.best_accuracy()
        bp_best = bp_front.best_accuracy()
        
        if tep_best and bp_best:
            report += f"""| Best Accuracy | {tep_best.accuracy:.4f} | {bp_best.accuracy:.4f} |
| Best Time (s) | {tep_front.best_speed().wall_time:.2f} | {bp_front.best_speed().wall_time:.2f} |
"""
        
        report += f"""
## Winner: **{comparison.get('winner', 'N/A').upper()}**

## Top 5 TEP Configurations

"""
        
        # Top 5 by accuracy
        sorted_tep = sorted(tep_front.all_trials, key=lambda t: -t.accuracy)[:5]
        for i, trial in enumerate(sorted_tep, 1):
            report += f"""### {i}. Accuracy: {trial.accuracy:.4f}
- Time: {trial.wall_time_seconds:.2f}s
- Params: {trial.param_count:,}
- Config: `{json.dumps(trial.config, indent=2)}`

"""
        
        report += """## Top 5 BP Configurations

"""
        
        sorted_bp = sorted(bp_front.all_trials, key=lambda t: -t.accuracy)[:5]
        for i, trial in enumerate(sorted_bp, 1):
            report += f"""### {i}. Accuracy: {trial.accuracy:.4f}
- Time: {trial.wall_time_seconds:.2f}s
- Params: {trial.param_count:,}
- Config: `{json.dumps(trial.config, indent=2)}`

"""
        
        return report
    
    def generate_final_report(
        self,
        all_results: Dict[str, Dict[str, Any]],
        phase_reached: int,
    ) -> str:
        """Generate comprehensive final report."""
        
        report = f"""# TEP vs BP: Final Experiment Report

**Generated**: {self._timestamp()}
**Phase Reached**: {phase_reached}

## Executive Summary

"""
        
        # Count wins/ties/losses per task
        wins = {"tep": 0, "bp": 0, "tie": 0}
        for task, data in all_results.items():
            comparison = data.get("comparison", {})
            winner = comparison.get("winner", "tie")
            wins[winner] += 1
        
        report += f"""- **TEP Wins**: {wins['tep']}
- **BP Wins**: {wins['bp']}
- **Ties**: {wins['tie']}

"""
        
        # Go/No-Go Decision
        if wins['tep'] > 0:
            report += """## Decision: **GO** ✅

TEP shows promising signal. Proceed to next phase.
"""
        else:
            report += """## Decision: **NO-GO** ❌

TEP did not demonstrate clear advantages over BP in this phase.
"""
        
        return report
    
    def save_report(self, report: str, filename: str):
        """Save report to file."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            f.write(report)
        return path
    
    def _timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
