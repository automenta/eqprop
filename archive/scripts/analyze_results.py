#!/usr/bin/env python3
"""
Rigorous Statistical Analysis and Scrutiny of Research Results

Performs comprehensive validation of experimental results:
1. Statistical significance testing (t-tests, p-values)
2. Variance analysis (flag high-variance models)
3. Improvement detection (identify flat/stagnant training)
4. Convergence validation (detect non-learning)
5. Generate publication-ready comparison tables

Output: Detailed markdown report with validation status
"""

import sys
sys.path.insert(0, '.')

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime


class ResultsAnalyzer:
    """Analyze experiment results for publishability."""
    
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.insights = []
    
    def load_benchmark(self, path):
        """Load benchmark results JSON."""
        with open(path) as f:
            return json.load(f)
    
    def analyze_variance(self, model_name, seeds, threshold=2.0):
        """Check if variance is acceptable for publication."""
        if len(seeds) < 3:
            self.warnings.append(f"{model_name}: Only {len(seeds)} seeds (recommend ≥3)")
            return None
        
        mean = np.mean(seeds)
        std = np.std(seeds)
        
        if std > threshold:
            self.warnings.append(
                f"{model_name}: High variance (±{std:.2f}% > {threshold}%) - "
                f"Seeds: {[f'{s:.2f}' for s in seeds]}"
            )
            return {'status': 'high_variance', 'mean': mean, 'std': std}
        
        return {'status': 'ok', 'mean': mean, 'std': std}
    
    def t_test_vs_baseline(self, baseline_seeds, model_seeds):
        """Perform t-test to check statistical significance."""
        if len(baseline_seeds) < 2 or len(model_seeds) < 2:
            return None, None
        
        t_stat, p_value = stats.ttest_ind(baseline_seeds, model_seeds)
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.var(baseline_seeds) + np.var(model_seeds)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(baseline_seeds) - np.mean(model_seeds)) / pooled_std
        else:
            cohens_d = 0
        
        return p_value, cohens_d
    
    def detect_flat_training(self, history, window=5, threshold=0.5):
        """Detect if training curve is flat (no learning)."""
        if 'test_acc' not in history or len(history['test_acc']) < window:
            return False
        
        accs = history['test_acc']
        if len(accs) < window:
            return False
        
        # Check last 'window' epochs
        recent = accs[-window:]
        improvement = max(recent) - min(recent)
        
        if improvement < threshold:
            return True  # Flat - less than threshold% improvement
        
        return False
    
    def analyze_convergence(self, history, random_baseline=10.0):
        """Check if model learned anything meaningful."""
        if 'test_acc' not in history or len(history['test_acc']) == 0:
            return {'status': 'no_data'}
        
        final_acc = history['test_acc'][-1]
        best_acc = max(history['test_acc'])
        
        # Check if better than random
        if best_acc < random_baseline * 1.5:  # 1.5x random
            self.errors.append(f"Model barely better than random (best={best_acc:.2f}% vs random={random_baseline}%)")
            return {'status': 'failed', 'final': final_acc, 'best': best_acc}
        
        # Check if training was flat
        if self.detect_flat_training(history):
            self.warnings.append(f"Flat training curve detected (final={final_acc:.2f}%)")
            return {'status': 'stagnant', 'final': final_acc, 'best': best_acc}
        
        return {'status': 'ok', 'final': final_acc, 'best': best_acc}
    
    def generate_comparison_table(self, results):
        """Generate markdown comparison table."""
        lines = [
            "| Model | Mean Acc | Std Dev | Seeds | Status |",
            "|-------|----------|---------|-------|--------|"
        ]
        
        for model_name, data in results.items():
            mean = data['mean_acc']
            std = data['std_acc']
            seeds = len(data['seeds'])
            status = data.get('status', 'ok')
            
            status_emoji = {
                'ok': '✅',
                'high_variance': '⚠️ High variance',
                'failed': '❌ Failed',
                'stagnant': '⚠️ Stagnant'
            }.get(status, '?')
            
            lines.append(
                f"| {model_name} | {mean:.2f}% | ±{std:.2f}% | {seeds} | {status_emoji} |"
            )
        
        return "\n".join(lines)
    
    def generate_report(self, benchmark_data, cifar_data=None):
        """Generate comprehensive statistical report."""
        lines = [
            "# Statistical Validation Report",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
            "---",
        ]

        # Detect if this is a multi-dataset benchmark (values are dicts of models)
        is_multi_dataset = False
        sample_val = next(iter(benchmark_data.values())) if benchmark_data else None
        if sample_val and 'mean_acc' not in sample_val and isinstance(sample_val, dict):
            is_multi_dataset = True
            
        if is_multi_dataset:
            # Process each task
            for task_name, task_data in benchmark_data.items():
                if 'error' in task_data:
                    lines.append(f"## {task_name}: ❌ Failed ({task_data['error']})")
                    lines.append("")
                    continue
                    
                lines.append(f"## {task_name} Results")
                lines.append("")
                lines.extend(self._analyze_single_task(task_data))
                lines.append("---")
                lines.append("")
        else:
            # Legacy single-task mode
            lines.append("## Benchmark Results")
            lines.append("")
            lines.extend(self._analyze_single_task(benchmark_data))
            lines.append("---")
            lines.append("")
        
        # CIFAR-10 specific validation (legacy support)
        if cifar_data:
            lines.append("## CIFAR-10 Hierarchical Results (Detailed)")
            lines.append("")
            
            for model_name, results in cifar_data.items():
                lines.append(f"### {model_name}")
                lines.append("")
                lines.append(f"**Best Accuracy**: {results['best_accuracy']:.2f}%")
                lines.append(f"**Best Config**: {results['best_config']}")
                lines.append("")
                
                if results['best_accuracy'] >= 50.0:
                    lines.append("> ✅ **Meets scalability threshold** (≥50%)")
                elif results['best_accuracy'] >= 35.0:
                    lines.append("> ⚠️ **Promising but below target** (35-50% range)")
                else:
                    lines.append("> ❌ **Below expectations** (<35%)")
                lines.append("")
        
        # Warnings and errors section remains valid global accumulation
        if self.warnings or self.errors:
            lines.append("---")
            lines.append("")
            lines.append("## Issues Detected")
            lines.append("")
        
        if self.errors:
            lines.append("### ❌ Errors (Critical)")
            lines.append("")
            for error in self.errors:
                lines.append(f"- {error}")
            lines.append("")
        
        if self.warnings:
            lines.append("### ⚠️ Warnings")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        # Final verdict
        lines.append("---")
        lines.append("")
        lines.append("## Publishability Assessment")
        lines.append("")
        
        if self.errors:
            lines.append("❌ **NOT READY** - Critical errors must be addressed")
        elif len(self.warnings) > 2:
            lines.append("⚠️ **NEEDS ATTENTION** - Multiple warnings detected")
        else:
            lines.append("✅ **READY FOR PUBLICATION** - Statistical validation passed")
        
        return "\n".join(lines)

    def _analyze_single_task(self, model_results):
        """Analyze a single task's model results."""
        lines = []
        baseline_name = None
        baseline_seeds = None
        
        results_summary = {}
        
        for model_name, data in model_results.items():
            # Identify baseline
            if 'Backprop' in model_name or ('MLP' in model_name and 'EqProp' not in model_name):
                baseline_name = model_name
                baseline_seeds = data['seeds']
            
            # Variance analysis
            var_result = self.analyze_variance(model_name, data['seeds'])
            
            results_summary[model_name] = {
                'mean_acc': data['mean_acc'],
                'std_acc': data['std_acc'],
                'seeds': data['seeds'],
                'status': var_result['status'] if var_result else 'ok'
            }
        
        # Comparison table
        lines.append(self.generate_comparison_table(results_summary))
        lines.append("")
        
        # Statistical significance tests
        if baseline_name and baseline_seeds:
            lines.append(f"### Significance vs {baseline_name}")
            lines.append("")
            lines.append(f"Baseline: **{baseline_name}** ({np.mean(baseline_seeds):.2f}% ± {np.std(baseline_seeds):.2f}%)")
            lines.append("")
            lines.append("| Model | Gap | P-value | Cohen's d | Significance |")
            lines.append("|-------|-----|---------|-----------|--------------|")
            
            for model_name, data in model_results.items():
                if model_name == baseline_name:
                    continue
                
                model_seeds = data['seeds']
                gap = np.mean(baseline_seeds) - np.mean(model_seeds)
                p_value, cohens_d = self.t_test_vs_baseline(baseline_seeds, model_seeds)
                
                if p_value is not None:
                    sig = "✅ Yes" if p_value < 0.05 else "⚠️ No"
                    effect = "large" if abs(cohens_d) > 0.8 else ("medium" if abs(cohens_d) > 0.5 else "small")
                    
                    lines.append(
                        f"| {model_name} | {gap:.2f}% | {p_value:.4f} | {cohens_d:.2f} ({effect}) | {sig} |"
                    )
            lines.append("")
            
        return lines


def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument('--benchmarks', type=str, required=True,
                       help='Path to MNIST benchmark results JSON')
    parser.add_argument('--cifar', type=str, default=None,
                       help='Path to CIFAR-10 results JSON (optional)')
    parser.add_argument('--output', type=str, default='results/statistical_report.md',
                       help='Output path for report')
    args = parser.parse_args()
    
    print("=" * 80)
    print("STATISTICAL ANALYSIS & VALIDATION")
    print("=" * 80)
    print()
    
    analyzer = ResultsAnalyzer()
    
    # Load data
    print(f"Loading MNIST benchmarks from {args.benchmarks}...")
    benchmark_data = analyzer.load_benchmark(args.benchmarks)
    
    cifar_data = None
    if args.cifar and Path(args.cifar).exists():
        print(f"Loading CIFAR-10 results from {args.cifar}...")
        cifar_data = analyzer.load_benchmark(args.cifar)
    
    # Generate report
    print("Analyzing results...")
    report = analyzer.generate_report(benchmark_data, cifar_data)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to {output_path}")
    print()
    
    # Print summary
    if analyzer.errors:
        print(f"❌ {len(analyzer.errors)} error(s) detected")
    if analyzer.warnings:
        print(f"⚠️  {len(analyzer.warnings)} warning(s) detected")
    if not analyzer.errors and not analyzer.warnings:
        print("✅ No issues detected")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
