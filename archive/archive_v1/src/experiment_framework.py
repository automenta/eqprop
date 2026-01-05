"""
Experiment Framework for TorEqProp

Modular, extensible infrastructure for running experiments with:
- Abstract base classes for experiments and metrics
- Plugin architecture for new experiment types
- Configuration-driven experiment definitions
- Unified logging and result tracking
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import json
import time
import subprocess
import sys
import os
import re


# ============================================================================
# Core Abstractions
# ============================================================================

class ExperimentStatus(Enum):
    """Status of an experiment run."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    name: str
    status: ExperimentStatus
    metrics: Dict[str, float]
    duration_sec: float
    timestamp: str
    log_path: Optional[str] = None
    error: Optional[str] = None
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def passed(self, threshold_metric: str, threshold_value: float) -> bool:
        """Check if experiment passed based on metric threshold."""
        if self.status != ExperimentStatus.SUCCESS:
            return False
        return self.metrics.get(threshold_metric, 0) >= threshold_value
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


class MetricExtractor(ABC):
    """Abstract base for extracting metrics from experiment output."""
    
    @abstractmethod
    def extract(self, output: str) -> Tuple[Dict[str, float], List[str]]:
        """Extract metrics and insights from output text.
        
        Returns:
            (metrics_dict, insights_list)
        """
        pass


class Experiment(ABC):
    """Abstract base class for experiments."""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._start_time: Optional[float] = None
        
    @property
    @abstractmethod
    def category(self) -> str:
        """Experiment category (e.g., 'classification', 'algorithmic', 'rl')."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> str:
        """Priority level: HIGH, MEDIUM, LOW."""
        pass
    
    @property
    @abstractmethod
    def expected_duration_min(self) -> float:
        """Expected duration in minutes."""
        pass
    
    @abstractmethod
    def build_command(self) -> str:
        """Build the command to run this experiment."""
        pass
    
    @abstractmethod
    def get_metric_extractor(self) -> MetricExtractor:
        """Get the metric extractor for this experiment type."""
        pass
    
    @abstractmethod
    def get_success_criteria(self) -> Tuple[str, float]:
        """Get (metric_name, threshold) for success determination."""
        pass
    
    def get_hypothesis(self) -> str:
        """Get the hypothesis being tested."""
        return self.config.get("hypothesis", "No hypothesis specified")
    
    def pre_run_hook(self) -> None:
        """Hook called before experiment runs. Override for setup."""
        pass
    
    def post_run_hook(self, result: ExperimentResult) -> None:
        """Hook called after experiment runs. Override for cleanup/analysis."""
        pass
    
    def run(self, output_dir: Path, dry_run: bool = False) -> ExperimentResult:
        """Execute the experiment and return results."""
        self._start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        if dry_run:
            return ExperimentResult(
                name=self.name,
                status=ExperimentStatus.SKIPPED,
                metrics={},
                duration_sec=0,
                timestamp=timestamp,
                insights=["Dry run - not executed"]
            )
        
        self.pre_run_hook()
        
        command = self.build_command()
        log_path = output_dir / f"{self.name.replace(' ', '_').lower()}.log"
        
        try:
            # Run with timeout (3x expected duration)
            timeout = self.expected_duration_min * 60 * 3
            
            # Use Popen for unbuffered, real-time output
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # Force Python unbuffered mode
            
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                env=env
            )
            
            # Read output line by line with real-time display
            output_lines = []
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)  # Real-time output
                    output_lines.append(line)
                    sys.stdout.flush()
            except Exception:
                pass
            
            # Wait for completion
            try:
                if process.poll() is None:
                    process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                raise
            
            duration = time.time() - self._start_time
            output = ''.join(output_lines)
            
            # Save log
            with open(log_path, "w") as f:
                f.write(f"Command: {command}\n")
                f.write(f"Duration: {duration:.1f}s\n")
                f.write(f"Exit code: {process.returncode}\n")
                f.write("=" * 70 + "\n")
                f.write(output)
            
            # Extract metrics
            extractor = self.get_metric_extractor()
            metrics, insights = extractor.extract(output)
            
            # Determine status
            if process.returncode != 0:
                status = ExperimentStatus.ERROR
                insights.append(f"Exit code: {process.returncode}")
            elif not metrics:
                status = ExperimentStatus.ERROR
                insights.append("No metrics extracted from output")
            else:
                metric_name, threshold = self.get_success_criteria()
                if metrics.get(metric_name, 0) >= threshold:
                    status = ExperimentStatus.SUCCESS
                else:
                    status = ExperimentStatus.FAILURE
            
            exp_result = ExperimentResult(
                name=self.name,
                status=status,
                metrics=metrics,
                duration_sec=duration,
                timestamp=timestamp,
                log_path=str(log_path),
                insights=insights,
                metadata={"command": command, "config": self.config}
            )
            
        except subprocess.TimeoutExpired:
            exp_result = ExperimentResult(
                name=self.name,
                status=ExperimentStatus.TIMEOUT,
                metrics={},
                duration_sec=time.time() - self._start_time,
                timestamp=timestamp,
                error="Timeout",
                insights=["Experiment timed out"]
            )
            
        except Exception as e:
            exp_result = ExperimentResult(
                name=self.name,
                status=ExperimentStatus.ERROR,
                metrics={},
                duration_sec=time.time() - self._start_time,
                timestamp=timestamp,
                error=str(e),
                insights=[f"Exception: {type(e).__name__}"]
            )
        
        self.post_run_hook(exp_result)
        return exp_result


# ============================================================================
# Metric Extractors
# ============================================================================

class AccuracyExtractor(MetricExtractor):
    """Extract accuracy metrics from training output."""
    
    def extract(self, output: str) -> Tuple[Dict[str, float], List[str]]:
        metrics = {}
        insights = []
        
        lines = output.strip().split("\n")
        
        # Find test accuracy
        for line in reversed(lines):
            if "Test Acc:" in line or "test/accuracy" in line or "test_accuracy" in line:
                try:
                    parts = line.split("Acc:")[-1].split("accuracy:")[-1]
                    value = float(parts.strip().split()[0].strip(","))
                    if value > 1.0:
                        value = value / 100
                    metrics["test_accuracy"] = value
                    break
                except (ValueError, IndexError):
                    continue
        
        # Find train accuracy
        for line in reversed(lines):
            if "Train Acc:" in line or "train/accuracy" in line:
                try:
                    parts = line.split("Acc:")[-1].split("accuracy:")[-1]
                    value = float(parts.strip().split()[0].strip(","))
                    if value > 1.0:
                        value = value / 100
                    metrics["train_accuracy"] = value
                    break
                except (ValueError, IndexError):
                    continue
        
        # Generate insights
        if "test_accuracy" in metrics:
            acc = metrics["test_accuracy"]
            if acc >= 0.90:
                insights.append("Excellent performance")
            elif acc >= 0.75:
                insights.append("Good performance - consider scaling")
            elif acc >= 0.50:
                insights.append("Learning signal present - needs tuning")
            else:
                insights.append("Weak signal - architecture changes may help")
        
        return metrics, insights


class RLRewardExtractor(MetricExtractor):
    """Extract reward metrics from RL output."""
    
    def extract(self, output: str) -> Tuple[Dict[str, float], List[str]]:
        metrics = {}
        insights = []
        
        lines = output.strip().split("\n")
        
        for line in reversed(lines):
            if "Average Reward:" in line or "avg_reward" in line or "Best Average" in line:
                try:
                    parts = line.split(":")[-1]
                    value = float(parts.strip().split()[0])
                    metrics["avg_reward"] = value
                    break
                except (ValueError, IndexError):
                    continue
        
        # Check for solved
        for line in lines:
            if "SOLVED" in line.upper():
                metrics["solved"] = 1.0
                insights.append("Environment solved!")
                break
        else:
            metrics["solved"] = 0.0
        
        return metrics, insights


class MemoryExtractor(MetricExtractor):
    """Extract memory profiling metrics."""
    
    def extract(self, output: str) -> Tuple[Dict[str, float], List[str]]:
        metrics = {}
        insights = []
        
        # Look for table row: d_model batch eqprop bp ratio
        # Example:       32     32         18.6         18.1     1.03x
        dataset_regex = re.compile(r'\s*\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x')
        
        lines = output.strip().split("\n")
        
        for line in lines:
            match = dataset_regex.search(line)
            if match:
                try:
                    eqprop_mem = float(match.group(1))
                    bp_mem = float(match.group(2))
                    ratio = float(match.group(3))
                    
                    metrics["memory_eqprop_mb"] = eqprop_mem
                    metrics["memory_bp_mb"] = bp_mem
                    metrics["memory_ratio"] = ratio
                    
                    if ratio < 0.5:
                        insights.append("Clear O(1) advantage!")
                    elif ratio < 1.0:
                        insights.append("Memory advantage present")
                    else:
                        insights.append("No memory advantage at this scale")
                    break 
                except (ValueError, IndexError):
                    continue
        
        return metrics, insights


class IterationExtractor(MetricExtractor):
    """Extract iteration count metrics for adaptive compute analysis."""
    
    def extract(self, output: str) -> Tuple[Dict[str, float], List[str]]:
        metrics = {}
        insights = []
        
        lines = output.strip().split("\n")
        
        # Look for iteration statistics
        for line in lines:
            if "Iterations:" in line and "±" in line:
                try:
                    parts = line.split("Iterations:")[-1].split("±")
                    mean = float(parts[0].strip())
                    std = float(parts[1].strip().split()[0])
                    metrics["mean_iterations"] = mean
                    metrics["std_iterations"] = std
                    
                    if std > 1.0:
                        insights.append("High iteration variance - adaptive compute?")
                    else:
                        insights.append("Uniform convergence")
                    break
                except (ValueError, IndexError):
                    continue
        
        # Also get accuracy
        acc_extractor = AccuracyExtractor()
        acc_metrics, acc_insights = acc_extractor.extract(output)
        metrics.update(acc_metrics)
        insights.extend(acc_insights)
        
        return metrics, insights


# ============================================================================
# Experiment Registry
# ============================================================================

class ExperimentRegistry:
    """Registry for experiment types and factories."""
    
    _experiment_types: Dict[str, Type[Experiment]] = {}
    _metric_extractors: Dict[str, Type[MetricExtractor]] = {}
    
    @classmethod
    def register_experiment(cls, name: str, experiment_class: Type[Experiment]):
        """Register an experiment type."""
        cls._experiment_types[name] = experiment_class
    
    @classmethod
    def register_extractor(cls, name: str, extractor_class: Type[MetricExtractor]):
        """Register a metric extractor."""
        cls._metric_extractors[name] = extractor_class
    
    @classmethod
    def get_experiment_class(cls, name: str) -> Type[Experiment]:
        """Get experiment class by name."""
        if name not in cls._experiment_types:
            raise ValueError(f"Unknown experiment type: {name}")
        return cls._experiment_types[name]
    
    @classmethod
    def get_extractor_class(cls, name: str) -> Type[MetricExtractor]:
        """Get extractor class by name."""
        if name not in cls._metric_extractors:
            raise ValueError(f"Unknown extractor: {name}")
        return cls._metric_extractors[name]
    
    @classmethod
    def list_experiment_types(cls) -> List[str]:
        """List registered experiment types."""
        return list(cls._experiment_types.keys())


# Register default extractors
ExperimentRegistry.register_extractor("accuracy", AccuracyExtractor)
ExperimentRegistry.register_extractor("rl_reward", RLRewardExtractor)
ExperimentRegistry.register_extractor("memory", MemoryExtractor)
ExperimentRegistry.register_extractor("iteration", IterationExtractor)


# ============================================================================
# Concrete Experiment Implementations
# ============================================================================

class ClassificationExperiment(Experiment):
    """Classification task experiment."""
    
    @property
    def category(self) -> str:
        return "classification"
    
    @property
    def priority(self) -> str:
        return self.config.get("priority", "MEDIUM")
    
    @property
    def expected_duration_min(self) -> float:
        return self.config.get("expected_time_min", 10)
    
        return cmd
    
    def build_command(self) -> str:
        dataset = self.config.get("dataset", "mnist")
        epochs = self.config.get("epochs", 3)
        rapid = self.config.get("rapid", True)
        d_model = self.config.get("d_model", 64)
        
        cmd = f"python train.py --dataset {dataset} --epochs {epochs} --d-model {d_model}"
        if rapid:
            cmd += " --rapid"
            
        # Pass additional config parameters if present
        for param in ["max_iters", "n_heads", "d_ff", "batch_size"]:
            if param in self.config:
                cmd += f" --{param.replace('_', '-')} {self.config[param]}"
        
        # Add any extra args (handle boolean flags correctly)
        for key, value in self.config.get("extra_args", {}).items():
            flag_name = key.replace('_', '-')
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    cmd += f" --{flag_name}"
                # Skip if False
            else:
                cmd += f" --{flag_name} {value}"
        
        return cmd
    
    def get_metric_extractor(self) -> MetricExtractor:
        return AccuracyExtractor()
    
    def get_success_criteria(self) -> Tuple[str, float]:
        threshold = self.config.get("success_threshold", 0.70)
        return ("test_accuracy", threshold)


class AlgorithmicExperiment(Experiment):
    """Algorithmic reasoning task experiment."""
    
    @property
    def category(self) -> str:
        return "algorithmic"
    
    @property
    def priority(self) -> str:
        return self.config.get("priority", "HIGH")
    
    @property
    def expected_duration_min(self) -> float:
        return self.config.get("expected_time_min", 5)
    
    def build_command(self) -> str:
        task = self.config.get("task", "parity")
        seq_len = self.config.get("seq_len", 8)
        epochs = self.config.get("epochs", 10)
        
        cmd = f"python train_algorithmic.py --task {task} --seq-len {seq_len} --epochs {epochs}"
        
        if self.config.get("analyze_difficulty", False):
            cmd += " --analyze-difficulty"
            
        # Pass additional config parameters if present
        for param in ["max_iters", "n_heads", "d_ff", "d_model", "batch_size", "lr"]:
            if param in self.config:
                cmd += f" --{param.replace('_', '-')} {self.config[param]}"
        
        for key, value in self.config.get("extra_args", {}).items():
            cmd += f" --{key.replace('_', '-')} {value}"
        
        return cmd
    
    def get_metric_extractor(self) -> MetricExtractor:
        if self.config.get("analyze_difficulty", False):
            return IterationExtractor()
        return AccuracyExtractor()
    
    def get_success_criteria(self) -> Tuple[str, float]:
        threshold = self.config.get("success_threshold", 0.85)
        return ("test_accuracy", threshold)


class RLExperiment(Experiment):
    """Reinforcement learning experiment."""
    
    @property
    def category(self) -> str:
        return "rl"
    
    @property
    def priority(self) -> str:
        return self.config.get("priority", "HIGH")
    
    @property
    def expected_duration_min(self) -> float:
        return self.config.get("expected_time_min", 15)
    
    def build_command(self) -> str:
        env = self.config.get("env", "CartPole-v1")
        episodes = self.config.get("episodes", 500)
        use_bp = self.config.get("use_bp", False)
        
        cmd = f"python train_rl.py --env {env} --episodes {episodes}"
        if use_bp:
            cmd += " --use-bp"
            
        # Pass additional config parameters if present
        for param in ["max_iters", "damping"]:
            if param in self.config:
                cmd += f" --{param.replace('_', '-')} {self.config[param]}"
        
        return cmd
    
    def get_metric_extractor(self) -> MetricExtractor:
        return RLRewardExtractor()
    
    def get_success_criteria(self) -> Tuple[str, float]:
        threshold = self.config.get("success_threshold", 195.0)
        return ("avg_reward", threshold)


class MemoryProfilingExperiment(Experiment):
    """Memory profiling experiment."""
    
    @property
    def category(self) -> str:
        return "memory"
    
    @property
    def priority(self) -> str:
        return self.config.get("priority", "MEDIUM")
    
    @property
    def expected_duration_min(self) -> float:
        return self.config.get("expected_time_min", 10)
    
    def build_command(self) -> str:
        # For memory profiling, we want to test the ACTUAL d_model specified, not smoke test override
        # So we use config's d_model but it should be the original value, not smoke test's 32
        # The smoke test filter should not override d_model for memory experiments
        d_model = self.config.get("d_model", 256)
        max_iters = self.config.get("max_iters", 100)
        
        cmd = f"python profile_memory.py --d-model {d_model} --max-iters {max_iters}"
        
        if "batch_size" in self.config:
            cmd += f" --batch-size {self.config['batch_size']}"
            
        return cmd
    
    def get_metric_extractor(self) -> MetricExtractor:
        return MemoryExtractor()
    
    def get_success_criteria(self) -> Tuple[str, float]:
        threshold = self.config.get("success_threshold", 1.0)
        return ("memory_ratio", threshold)


# Register experiment types
ExperimentRegistry.register_experiment("classification", ClassificationExperiment)
ExperimentRegistry.register_experiment("algorithmic", AlgorithmicExperiment)
ExperimentRegistry.register_experiment("rl", RLExperiment)
ExperimentRegistry.register_experiment("memory", MemoryProfilingExperiment)


# ============================================================================
# Experiment Builder (Factory)
# ============================================================================

class ExperimentBuilder:
    """Build experiments from configuration."""
    
    @staticmethod
    def from_dict(config: dict) -> Experiment:
        """Build experiment from dictionary config."""
        name = config.get("name", "Unnamed")
        exp_type = config.get("type", "classification")
        
        exp_class = ExperimentRegistry.get_experiment_class(exp_type)
        return exp_class(name, config)
    
    @staticmethod
    def from_yaml(path: Path) -> List[Experiment]:
        """Load experiments from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        
        experiments = []
        for exp_config in data.get("experiments", []):
            experiments.append(ExperimentBuilder.from_dict(exp_config))
        
        return experiments
    
    @staticmethod
    def from_json(path: Path) -> List[Experiment]:
        """Load experiments from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        experiments = []
        for exp_config in data.get("experiments", []):
            experiments.append(ExperimentBuilder.from_dict(exp_config))
        
        return experiments


# ============================================================================
# Results Aggregation
# ============================================================================

@dataclass
class CampaignSummary:
    """Summary of a full experiment campaign."""
    total_experiments: int
    passed: int
    failed: int
    errors: int
    skipped: int
    total_duration_min: float
    results_by_category: Dict[str, Dict[str, int]]
    best_results: Dict[str, ExperimentResult]
    insights: List[str]
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["best_results"] = {k: v.to_dict() for k, v in self.best_results.items()}
        return d


class ResultsAggregator:
    """Aggregate and analyze experiment results."""
    
    def __init__(self, results: List[ExperimentResult]):
        self.results = results
    
    def summarize(self) -> CampaignSummary:
        """Generate campaign summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == ExperimentStatus.SUCCESS)
        failed = sum(1 for r in self.results if r.status == ExperimentStatus.FAILURE)
        errors = sum(1 for r in self.results if r.status == ExperimentStatus.ERROR)
        skipped = sum(1 for r in self.results if r.status == ExperimentStatus.SKIPPED)
        
        total_duration = sum(r.duration_sec for r in self.results) / 60
        
        # Categorize results
        by_category: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            category = r.metadata.get("config", {}).get("type", "unknown")
            if category not in by_category:
                by_category[category] = {"passed": 0, "failed": 0, "errors": 0}
            
            if r.status == ExperimentStatus.SUCCESS:
                by_category[category]["passed"] += 1
            elif r.status == ExperimentStatus.FAILURE:
                by_category[category]["failed"] += 1
            else:
                by_category[category]["errors"] += 1
        
        # Find best results by category
        best_results: Dict[str, ExperimentResult] = {}
        for r in self.results:
            if r.status != ExperimentStatus.SUCCESS:
                continue
            category = r.metadata.get("config", {}).get("type", "unknown")
            
            # Use test_accuracy or avg_reward as comparison metric
            metric_val = r.metrics.get("test_accuracy", r.metrics.get("avg_reward", 0))
            
            if category not in best_results:
                best_results[category] = r
            else:
                existing_val = best_results[category].metrics.get(
                    "test_accuracy", 
                    best_results[category].metrics.get("avg_reward", 0)
                )
                if metric_val > existing_val:
                    best_results[category] = r
        
        # Generate insights
        insights = []
        for r in self.results:
            if r.insights:
                insights.extend([f"{r.name}: {i}" for i in r.insights])
        
        return CampaignSummary(
            total_experiments=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total_duration_min=total_duration,
            results_by_category=by_category,
            best_results=best_results,
            insights=insights[:20]  # Top 20 insights
        )
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        summary = self.summarize()
        
        lines = ["# Experiment Campaign Results\n"]
        lines.append(f"**Total Duration**: {summary.total_duration_min:.1f} minutes\n")
        lines.append(f"## Summary\n")
        lines.append(f"- Total: {summary.total_experiments}")
        lines.append(f"- ✅ Passed: {summary.passed}")
        lines.append(f"- ❌ Failed: {summary.failed}")
        lines.append(f"- ⚠️ Errors: {summary.errors}\n")
        
        lines.append("## Results by Category\n")
        for cat, stats in summary.results_by_category.items():
            lines.append(f"### {cat.capitalize()}")
            lines.append(f"- Passed: {stats['passed']}, Failed: {stats['failed']}, Errors: {stats['errors']}\n")
        
        lines.append("## Best Results\n")
        for cat, result in summary.best_results.items():
            metric = result.metrics.get("test_accuracy", result.metrics.get("avg_reward", 0))
            lines.append(f"- **{cat}**: {result.name} ({metric:.4f})")
        
        lines.append("\n## Key Insights\n")
        for insight in summary.insights[:10]:
            lines.append(f"- {insight}")
        
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_default_campaign() -> List[Experiment]:
    """Create the default experiment campaign."""
    configs = [
        # Phase 1: Classification
        {"name": "MNIST Rapid", "type": "classification", "dataset": "mnist", "epochs": 3, "rapid": True, "success_threshold": 0.80, "priority": "HIGH", "expected_time_min": 5, "hypothesis": "Baseline validation"},
        {"name": "Fashion Rapid", "type": "classification", "dataset": "fashion", "epochs": 3, "rapid": True, "success_threshold": 0.70, "priority": "HIGH", "expected_time_min": 5, "hypothesis": "Harder than MNIST"},
        {"name": "CIFAR-10 Rapid", "type": "classification", "dataset": "cifar10", "epochs": 3, "rapid": True, "success_threshold": 0.35, "priority": "HIGH", "expected_time_min": 8, "hypothesis": "Complexity jump"},
        {"name": "SVHN Rapid", "type": "classification", "dataset": "svhn", "epochs": 3, "rapid": True, "success_threshold": 0.40, "priority": "MEDIUM", "expected_time_min": 8, "hypothesis": "Real-world digits"},
        
        # Phase 2: Algorithmic
        {"name": "Parity N=8", "type": "algorithmic", "task": "parity", "seq_len": 8, "epochs": 10, "success_threshold": 0.90, "priority": "HIGH", "expected_time_min": 5, "hypothesis": "Adaptive compute test"},
        {"name": "Parity N=12", "type": "algorithmic", "task": "parity", "seq_len": 12, "epochs": 15, "success_threshold": 0.85, "priority": "MEDIUM", "expected_time_min": 8, "hypothesis": "Longer sequences"},
        {"name": "Copy Task", "type": "algorithmic", "task": "copy", "seq_len": 8, "epochs": 5, "success_threshold": 0.95, "priority": "MEDIUM", "expected_time_min": 3, "hypothesis": "Easy baseline"},
        {"name": "Addition 4-digit", "type": "algorithmic", "task": "addition", "seq_len": 8, "epochs": 20, "n_digits": 4, "success_threshold": 0.50, "priority": "HIGH", "expected_time_min": 10, "hypothesis": "Sequential carry propagation"},
        
        # Phase 3: RL
        {"name": "CartPole EqProp", "type": "rl", "env": "CartPole-v1", "episodes": 500, "use_bp": False, "success_threshold": 195.0, "priority": "HIGH", "expected_time_min": 15, "hypothesis": "Can EqProp solve control?"},
        {"name": "CartPole BP", "type": "rl", "env": "CartPole-v1", "episodes": 500, "use_bp": True, "success_threshold": 195.0, "priority": "HIGH", "expected_time_min": 10, "hypothesis": "BP baseline"},
        
        # Phase 4: Accuracy Push (longer runs)
        {"name": "MNIST Extended", "type": "classification", "dataset": "mnist", "epochs": 100, "rapid": False, "d_model": 256, "success_threshold": 0.945, "priority": "HIGH", "expected_time_min": 180, "hypothesis": "Extended training", "extra_args": {"beta": 0.22, "dropout": 0.1, "compile": True}},
        
        # Phase 5: Memory
        {"name": "Memory d=256", "type": "memory", "d_model": 256, "max_iters": 100, "success_threshold": 1.5, "priority": "MEDIUM", "expected_time_min": 5, "hypothesis": "Baseline memory"},
        {"name": "Memory d=1024", "type": "memory", "d_model": 1024, "max_iters": 100, "success_threshold": 0.8, "priority": "HIGH", "expected_time_min": 10, "hypothesis": "O(1) emergence"},
        {"name": "Memory d=2048", "type": "memory", "d_model": 2048, "max_iters": 100, "success_threshold": 0.5, "priority": "HIGH", "expected_time_min": 20, "hypothesis": "Clear O(1) advantage"},
    ]
    
    return [ExperimentBuilder.from_dict(c) for c in configs]
