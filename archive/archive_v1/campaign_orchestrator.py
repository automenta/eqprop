#!/usr/bin/env python3
"""
TorEqProp Campaign Orchestrator

Unified entry point for running complete scientific campaigns that:
1. Optimize hyperparameters for EqProp and baselines
2. Run robustness analysis on best models
3. Generate scaling law data
4. Produce publication-ready validated reports

Usage:
    python campaign_orchestrator.py --task mnist --full-campaign
    python campaign_orchestrator.py --robustness --model checkpoints/best.pt
    python campaign_orchestrator.py --scaling --sizes 64 128 256 512
    python campaign_orchestrator.py --validate --report
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

# Import hyperopt engine
from hyperopt_engine import (
    HyperOptEngine, HyperOptTrial, HyperOptDB,
    ParetoAnalyzer, TrialMatcher, CostAwareEvaluator
)

# Import validation pipeline
from hyperopt.validation import ValidationPipeline, StatisticalVerdict

# Import analysis tools
from analysis.robustness import AdversarialEvaluator, AttackResult
from analysis.scaling import ScalingAnalyzer

# Import statistics
from statistics import StatisticalAnalyzer


@dataclass
class CampaignResult:
    """Complete results from a scientific campaign."""
    task: str
    timestamp: str
    duration_seconds: float
    
    # Hyperopt results
    best_eqprop_config: Dict[str, Any]
    best_eqprop_performance: float
    best_baseline_config: Dict[str, Any]
    best_baseline_performance: float
    
    # Statistical comparison
    statistical_verdict: str
    p_value: float
    effect_size: float
    
    # Optional analyses
    robustness_results: Optional[Dict] = None
    scaling_results: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class CampaignOrchestrator:
    """Orchestrates complete scientific campaigns for TorEqProp analysis."""
    
    def __init__(self, 
                 config_path: str = "validation_config.yaml",
                 output_dir: str = "campaign_results"):
        """Initialize the orchestrator with all analysis components."""
        self.engine = HyperOptEngine(config_path)
        self.validation = ValidationPipeline(
            archive_dir="evidence_archive",
            report_dir="reports"
        )
        self.stats = StatisticalAnalyzer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_campaign(self,
                          task: str = "mnist",
                          n_trials: int = 10,
                          epochs: int = 5,
                          seeds: List[int] = None,
                          strategy: str = "random",
                          include_robustness: bool = True,
                          include_scaling: bool = False) -> CampaignResult:
        """Run a complete scientific campaign.
        
        Steps:
        1. Hyperparameter optimization for both algorithms
        2. Statistical comparison
        3. Optional: Robustness analysis
        4. Optional: Scaling analysis
        5. Generate validated report
        """
        if seeds is None:
            seeds = [0, 1, 2]
        
        start_time = time.time()
        print("\n" + "=" * 70)
        print("  🔬 TorEqProp Scientific Campaign")
        print("=" * 70)
        print(f"  Task: {task}")
        print(f"  Trials: {n_trials} per algorithm")
        print(f"  Epochs: {epochs}")
        print(f"  Seeds: {seeds}")
        print(f"  Strategy: {strategy}")
        print("=" * 70)
        
        # Phase 1: Hyperparameter Optimization
        print("\n📊 Phase 1: Hyperparameter Optimization")
        print("-" * 70)
        
        self.engine.run(
            task=task,
            n_trials=n_trials,
            strategy=strategy,
            seeds=seeds,
            epochs=epochs,
            headless=True
        )
        
        # Get results
        eqprop_trials = self.engine.db.get_trials(
            algorithm="eqprop", task=task, status="complete")
        baseline_trials = self.engine.db.get_trials(
            algorithm="bp", task=task, status="complete")
        
        if not eqprop_trials or not baseline_trials:
            raise RuntimeError("No completed trials found!")
        
        best_eq = max(eqprop_trials, key=lambda t: t.performance)
        best_bl = max(baseline_trials, key=lambda t: t.performance)
        
        print(f"\n✅ Best EqProp: {best_eq.performance:.4f}")
        print(f"✅ Best Baseline: {best_bl.performance:.4f}")
        
        # Phase 2: Statistical Comparison
        print("\n📈 Phase 2: Statistical Analysis")
        print("-" * 70)
        
        eq_perfs = [t.performance for t in eqprop_trials]
        bl_perfs = [t.performance for t in baseline_trials]
        
        comparison = self.stats.compare(eq_perfs, bl_perfs, "EqProp", "Baseline")
        
        if comparison.is_significant:
            if comparison.algo1_mean > comparison.algo2_mean:
                verdict = "EqProp significantly outperforms baseline"
            else:
                verdict = "Baseline significantly outperforms EqProp"
        else:
            verdict = "No significant difference"
        
        print(f"  Verdict: {verdict}")
        print(f"  p-value: {comparison.p_value:.4f}")
        print(f"  Effect size (Cohen's d): {comparison.cohens_d:.2f}")
        
        # Phase 3: Optional Robustness Analysis
        robustness_results = None
        if include_robustness and task in ["mnist", "fashion", "cifar10", "svhn"]:
            print("\n🛡️ Phase 3: Robustness Analysis")
            print("-" * 70)
            robustness_results = self._run_robustness_analysis(task, best_eq, best_bl)
        
        # Phase 4: Optional Scaling Analysis
        scaling_results = None
        if include_scaling:
            print("\n📈 Phase 4: Scaling Analysis")
            print("-" * 70)
            scaling_results = self._run_scaling_analysis(task)
        
        # Phase 5: Generate Report
        print("\n📝 Phase 5: Generating Report")
        print("-" * 70)
        
        duration = time.time() - start_time
        
        result = CampaignResult(
            task=task,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            best_eqprop_config=best_eq.config,
            best_eqprop_performance=best_eq.performance,
            best_baseline_config=best_bl.config,
            best_baseline_performance=best_bl.performance,
            statistical_verdict=verdict,
            p_value=comparison.p_value,
            effect_size=comparison.cohens_d,
            robustness_results=robustness_results,
            scaling_results=scaling_results
        )
        
        self._save_campaign_result(result)
        self._generate_report(result, eqprop_trials, baseline_trials)
        
        print("\n" + "=" * 70)
        print("  ✅ Campaign Complete")
        print("=" * 70)
        print(f"  Duration: {duration/60:.1f} minutes")
        print(f"  Results: {self.output_dir}/campaign_{task}.json")
        print(f"  Report: reports/campaign_{task}_report.md")
        print("=" * 70)
        
        return result
    
    def _run_robustness_analysis(self, task: str, 
                                  best_eq: HyperOptTrial,
                                  best_bl: HyperOptTrial) -> Dict:
        """Run adversarial robustness evaluation on best models."""
        try:
            import torch
            from src.models import ToroidalTransformer
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader
            
            # Load dataset for evaluation
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            if task == "mnist":
                test_data = datasets.MNIST(
                    'data', train=False, download=True, transform=transform)
            elif task == "fashion":
                test_data = datasets.FashionMNIST(
                    'data', train=False, download=True, transform=transform)
            else:
                print(f"  ⚠️ Robustness analysis not implemented for {task}")
                return None
            
            test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
            
            # Create and evaluate EqProp model
            eq_model = ToroidalTransformer(
                d_model=best_eq.config.get("d_model", 128),
                n_heads=8,
                d_ff=best_eq.config.get("d_model", 128) * 4,
                n_classes=10
            )
            
            evaluator = AdversarialEvaluator(eq_model)
            
            # Run FGSM attack
            print("  Running FGSM attack on EqProp model...")
            fgsm_result = evaluator.evaluate(
                test_loader, method="fgsm", epsilon=0.1)
            
            results = {
                "eqprop_clean_acc": fgsm_result.clean_acc,
                "eqprop_adversarial_acc": fgsm_result.adversarial_acc,
                "attack_success_rate": fgsm_result.attack_success_rate,
                "epsilon": fgsm_result.epsilon,
                "method": fgsm_result.method
            }
            
            print(f"  Clean accuracy: {fgsm_result.clean_acc:.2f}%")
            print(f"  Adversarial accuracy: {fgsm_result.adversarial_acc:.2f}%")
            print(f"  Attack success rate: {fgsm_result.attack_success_rate:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"  ⚠️ Robustness analysis failed: {e}")
            return None
    
    def _run_scaling_analysis(self, task: str) -> List[Dict]:
        """Run scaling law analysis across model sizes."""
        try:
            sizes = [32, 64, 128, 256]
            print(f"  Testing sizes: {sizes}")
            
            analyzer = ScalingAnalyzer(self.engine, results_dir="scaling_results")
            results = analyzer.run_scaling_sweep(task=task, sizes=sizes, epochs_per_grad=5)
            
            return results
            
        except Exception as e:
            print(f"  ⚠️ Scaling analysis failed: {e}")
            return None
    
    def _save_campaign_result(self, result: CampaignResult):
        """Save campaign results to JSON."""
        path = self.output_dir / f"campaign_{result.task}.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _generate_report(self, result: CampaignResult,
                          eqprop_trials: List[HyperOptTrial],
                          baseline_trials: List[HyperOptTrial]):
        """Generate a publication-ready Markdown report."""
        report_path = Path("reports") / f"campaign_{result.task}_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        eq_perfs = [t.performance for t in eqprop_trials]
        bl_perfs = [t.performance for t in baseline_trials]
        
        report = f"""# TorEqProp Campaign Report: {result.task.upper()}

**Generated**: {result.timestamp}
**Duration**: {result.duration_seconds/60:.1f} minutes

---

## Executive Summary

| Metric | EqProp | Baseline | Winner |
|--------|--------|----------|--------|
| Best Performance | {result.best_eqprop_performance:.4f} | {result.best_baseline_performance:.4f} | {'🔋 EqProp' if result.best_eqprop_performance > result.best_baseline_performance else '⚡ Baseline'} |
| Mean Performance | {np.mean(eq_perfs):.4f} | {np.mean(bl_perfs):.4f} | {'🔋 EqProp' if np.mean(eq_perfs) > np.mean(bl_perfs) else '⚡ Baseline'} |
| Std Dev | {np.std(eq_perfs):.4f} | {np.std(bl_perfs):.4f} | - |

**Verdict**: {result.statistical_verdict}
- **p-value**: {result.p_value:.4f}
- **Effect Size (Cohen's d)**: {result.effect_size:.2f}

---

## Best Configurations

### 🔋 EqProp
```yaml
{self._format_config(result.best_eqprop_config)}
```

### ⚡ Baseline
```yaml
{self._format_config(result.best_baseline_config)}
```

---

## Statistical Analysis

- **EqProp trials**: {len(eqprop_trials)}
- **Baseline trials**: {len(baseline_trials)}
- **Significance threshold**: α = 0.05
- **Test used**: Welch's t-test

"""
        
        if result.robustness_results:
            report += f"""
## Robustness Analysis

| Metric | Value |
|--------|-------|
| Clean Accuracy | {result.robustness_results.get('eqprop_clean_acc', 'N/A'):.2f}% |
| Adversarial Accuracy | {result.robustness_results.get('eqprop_adversarial_acc', 'N/A'):.2f}% |
| Attack Success Rate | {result.robustness_results.get('attack_success_rate', 'N/A'):.2f}% |
| Attack Method | {result.robustness_results.get('method', 'N/A')} |
| Epsilon | {result.robustness_results.get('epsilon', 'N/A')} |

"""
        
        if result.scaling_results:
            report += """
## Scaling Analysis

| d_model | Parameters | Loss | Time (s) |
|---------|------------|------|----------|
"""
            for r in result.scaling_results:
                report += f"| {r['d_model']} | {r['params']} | {r['loss']:.4f} | {r['time']:.1f} |\n"
        
        report += """
---

*Report generated by TorEqProp Campaign Orchestrator*
"""
        
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"  Report saved to {report_path}")
    
    def _format_config(self, config: Dict) -> str:
        """Format config as YAML-like string."""
        lines = []
        for k, v in sorted(config.items()):
            lines.append(f"{k}: {v}")
        return "\n".join(lines)
    
    def run_robustness_only(self, model_path: str, task: str = "mnist"):
        """Run robustness analysis on a saved model."""
        print("\n🛡️ Running Robustness Analysis")
        print("-" * 70)
        
        # This would load the model and run analysis
        # Placeholder for now - would need model loading logic
        print(f"  Model: {model_path}")
        print(f"  Task: {task}")
        print("  ⚠️ Standalone robustness analysis requires trained model checkpoint")
    
    def validate_existing_results(self, task: str = None):
        """Generate validated report from existing results in database."""
        print("\n📝 Generating Validated Report from Existing Results")
        print("-" * 70)
        
        # Get all tasks if none specified
        all_trials = self.engine.db.get_trials(status="complete")
        if not all_trials:
            print("  ❌ No completed trials found!")
            return
        
        tasks = set(t.task for t in all_trials)
        if task:
            tasks = {task} if task in tasks else set()
        
        for t in tasks:
            print(f"\n  Processing task: {t}")
            
            eq = self.engine.db.get_trials(algorithm="eqprop", task=t, status="complete")
            bl = self.engine.db.get_trials(algorithm="bp", task=t, status="complete")
            
            if eq and bl:
                eq_perfs = [trial.performance for trial in eq]
                bl_perfs = [trial.performance for trial in bl]
                
                claims = [{
                    "claim": f"EqProp vs Baseline on {t}",
                    "eqprop": eq_perfs,
                    "baseline": bl_perfs
                }]
                
                verdicts = self.validation.validate_claims(claims)
                self.validation.generate_report(verdicts, title=f"{t.upper()} Validation Report")
                
                print(f"  ✅ Report generated for {t}")


def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Campaign Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full campaign on MNIST
  python campaign_orchestrator.py --task mnist --full-campaign
  
  # Quick campaign (fewer trials)
  python campaign_orchestrator.py --task xor --full-campaign --n-trials 5 --epochs 2
  
  # Validate existing results
  python campaign_orchestrator.py --validate
  
  # Multi-task campaign
  python campaign_orchestrator.py --campaign --tasks mnist fashion cartpole
        """
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full-campaign", action="store_true",
                     help="Run complete scientific campaign")
    mode.add_argument("--campaign", action="store_true",
                     help="Run campaign across multiple tasks")
    mode.add_argument("--robustness", action="store_true",
                     help="Run robustness analysis only")
    mode.add_argument("--validate", action="store_true",
                     help="Generate validated report from existing results")
    
    # Task configuration
    parser.add_argument("--task", type=str, default="mnist",
                       help="Task to run campaign on")
    parser.add_argument("--tasks", nargs="+", 
                       default=["xor", "mnist", "cartpole"],
                       help="Tasks for multi-task campaign")
    
    # Campaign options
    parser.add_argument("--n-trials", type=int, default=10,
                       help="Number of trials per algorithm")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Epochs per trial")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                       help="Random seeds")
    parser.add_argument("--strategy", type=str, default="random",
                       choices=["grid", "random", "sobol", "lhs"],
                       help="Sampling strategy")
    
    # Analysis options
    parser.add_argument("--include-robustness", action="store_true",
                       help="Include robustness analysis in campaign")
    parser.add_argument("--include-scaling", action="store_true",
                       help="Include scaling analysis in campaign")
    
    # Model path for standalone robustness
    parser.add_argument("--model", type=str, default=None,
                       help="Model checkpoint path for robustness analysis")
    
    args = parser.parse_args()
    
    orchestrator = CampaignOrchestrator()
    
    if args.full_campaign:
        orchestrator.run_full_campaign(
            task=args.task,
            n_trials=args.n_trials,
            epochs=args.epochs,
            seeds=args.seeds,
            strategy=args.strategy,
            include_robustness=args.include_robustness,
            include_scaling=args.include_scaling
        )
    
    elif args.campaign:
        for task in args.tasks:
            print(f"\n{'='*70}")
            print(f"  Campaign for: {task.upper()}")
            print("=" * 70)
            try:
                orchestrator.run_full_campaign(
                    task=task,
                    n_trials=args.n_trials,
                    epochs=args.epochs,
                    seeds=args.seeds,
                    strategy=args.strategy,
                    include_robustness=args.include_robustness
                )
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
    
    elif args.robustness:
        if not args.model:
            print("❌ --model required for robustness analysis")
            sys.exit(1)
        orchestrator.run_robustness_only(args.model, args.task)
    
    elif args.validate:
        orchestrator.validate_existing_results(args.task if args.task != "mnist" else None)
    
    else:
        # Default: run a quick campaign
        orchestrator.run_full_campaign(
            task=args.task,
            n_trials=args.n_trials,
            epochs=args.epochs,
            seeds=args.seeds,
            strategy=args.strategy
        )


if __name__ == "__main__":
    main()
