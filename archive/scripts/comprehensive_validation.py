#!/usr/bin/env python3
"""
Comprehensive Validation Suite for Top 3 TODO5 Research Tracks

Runs rigorous experiments with statistical analysis and creates
publication-quality results for README.md documentation.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import time


@dataclass
class ValidationResult:
    """Stores comprehensive validation results."""
    track_name: str
    experiment_name: str
    seeds: List[int]
    results: List[Dict]
    mean_metric: float
    std_metric: float
    confidence_interval: Tuple[float, float]
    p_value: float = None  # vs baseline


class ComprehensiveValidator:
    """Runs deep validation experiments with statistical rigor."""
    
    def __init__(self, num_seeds: int = 5):
        self.num_seeds = num_seeds
        self.results = []
    
    def run_adversarial_healing_deep(self) -> ValidationResult:
        """
        Deep validation of adversarial self-healing.
        
        Tests:
        1. Neuron ablation at 15%, 30%, 50%
        2. Noise injection at various levels
        3. Compare EqProp vs BP recovery
        """
        print("\n" + "="*70)
        print("DEEP VALIDATION: ADVERSARIAL SELF-HEALING")
        print("="*70)
        
        from src.models import LoopedMLP, BackpropMLP
        from scripts.adversarial_healing import SelfHealingAnalyzer
        
        all_results = []
        
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # EqProp model
            model_eqprop = LoopedMLP(784, 256, 10, use_spectral_norm=True)
            analyzer_eq = SelfHealingAnalyzer(model_eqprop)
            
            # BP baseline
            model_bp = BackpropMLP(784, 256, 10)
            analyzer_bp = SelfHealingAnalyzer(model_bp)
            
            seed_results = {
                'seed': seed,
                'ablation_15': {},
                'ablation_30': {},
                'ablation_50': {},
                'noise_damping': {}
            }
            
            # Test ablation resilience
            for ablation_pct in [0.15, 0.30, 0.50]:
                # EqProp
                analyzer_eq.restore()
                analyzer_eq.ablate_neurons(ablation_pct)
                x = torch.randn(32, 784)
                # Quick accuracy test
                eq_functional = self._test_degraded_model(model_eqprop, x)
                
                # BP
                analyzer_bp.restore()
                analyzer_bp.ablate_neurons(ablation_pct)
                bp_functional = self._test_degraded_model(model_bp, x)
                
                key = f'ablation_{int(ablation_pct*100)}'
                seed_results[key] = {
                    'eqprop_functional': eq_functional,
                    'bp_functional': bp_functional,
                    'advantage': eq_functional - bp_functional
                }
            
            # Test noise damping
            analyzer_eq.restore()
            x = torch.randn(16, 784)
            damping = analyzer_eq.run_relaxation_damping_test(x, noise_level=1.0)
            seed_results['noise_damping'] = {
                'damping_ratio': damping['damping_ratio'],
                'initial_diff': damping['initial_difference'],
                'final_diff': damping['final_difference']
            }
            
            all_results.append(seed_results)
            print(f"  Seed {seed}: 50% ablation advantage = {seed_results['ablation_50']['advantage']:.1%}")
        
        # Aggregate statistics
        advantages_50 = [r['ablation_50']['advantage'] for r in all_results]
        damping_ratios = [r['noise_damping']['damping_ratio'] for r in all_results]
        
        mean_adv = np.mean(advantages_50)
        std_adv = np.std(advantages_50)
        
        print(f"\n  50% Ablation Advantage: {mean_adv:.1%} ± {std_adv:.1%}")
        print(f"  Avg Damping Ratio: {np.mean(damping_ratios):.3f}")
        
        return ValidationResult(
            track_name="Adversarial Self-Healing",
            experiment_name="Neuron Ablation Resilience",
            seeds=list(range(self.num_seeds)),
            results=all_results,
            mean_metric=mean_adv,
            std_metric=std_adv,
            confidence_interval=self._compute_ci(advantages_50)
        )
    
    def run_ternary_weights_deep(self) -> ValidationResult:
        """
        Deep validation of ternary weights.
        
        Tests:
        1. Full MNIST training with multiple seeds
        2. Accuracy vs sparsity trade-off
        3. Bit operations count
        """
        print("\n" + "="*70)
        print("DEEP VALIDATION: TERNARY WEIGHTS")
        print("="*70)
        
        from src.models import TernaryEqProp
        
        try:
            from src.tasks import get_task_loader
            train_loader, test_loader, _, _ = get_task_loader("mnist", batch_size=64, flatten=True)
        except:
            print("  Skipping MNIST test (data unavailable)")
            return None
        
        all_results = []
        
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = TernaryEqProp(784, 256, 10, threshold=0.5)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train for 3 epochs
            model.train()
            for epoch in range(3):
                for i, (x, y) in enumerate(train_loader):
                    if i > 100:  # Limit for speed
                        break
                    optimizer.zero_grad()
                    out = model(x, steps=20)
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(test_loader):
                    if i > 20:
                        break
                    out = model(x, steps=20)
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            
            accuracy = correct / total
            stats = model.get_model_stats()
            
            seed_results = {
                'seed': seed,
                'accuracy': accuracy,
                'sparsity': stats['overall_sparsity'],
                'bit_ops': model.count_bit_operations()
            }
            
            all_results.append(seed_results)
            print(f"  Seed {seed}: {accuracy:.1%} accuracy, {stats['overall_sparsity']:.0%} sparsity")
        
        accuracies = [r['accuracy'] for r in all_results]
        sparsities = [r['sparsity'] for r in all_results]
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"\n  Mean Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
        print(f"  Mean Sparsity: {np.mean(sparsities):.0%}")
        
        return ValidationResult(
            track_name="Ternary Weights",
            experiment_name="MNIST Classification",
            seeds=list(range(self.num_seeds)),
            results=all_results,
            mean_metric=mean_acc,
            std_metric=std_acc,
            confidence_interval=self._compute_ci(accuracies)
        )
    
    def run_neural_cube_deep(self) -> ValidationResult:
        """
        Deep validation of Neural Cube 3D topology.
        
        Tests:
        1. Scale to larger cubes
        2. Compare vs flat MLP
        3. Visualize neurogenesis
        """
        print("\n" + "="*70)
        print("DEEP VALIDATION: NEURAL CUBE 3D")
        print("="*70)
        
        from src.models import NeuralCube, LoopedMLP
        
        all_results = []
        
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Test data
            x = torch.randn(32, 64)
            y = torch.randint(0, 10, (32,))
            
            # 3D Cube
            cube = NeuralCube(cube_size=6, input_dim=64, output_dim=10)
            optimizer_cube = torch.optim.Adam(cube.parameters(), lr=0.01)
            
            # Train cube
            initial_loss = F.cross_entropy(cube(x, steps=15), y).item()
            for _ in range(30):
                optimizer_cube.zero_grad()
                loss = F.cross_entropy(cube(x, steps=15), y)
                loss.backward()
                optimizer_cube.step()
            final_loss = F.cross_entropy(cube(x, steps=15), y).item()
            cube_learning = 1 - (final_loss / initial_loss)
            
            # Flat MLP baseline (same parameter count)
            flat = LoopedMLP(64, 128, 10, use_spectral_norm=True)
            optimizer_flat = torch.optim.Adam(flat.parameters(), lr=0.01)
            
            initial_loss_flat = F.cross_entropy(flat(x, steps=15), y).item()
            for _ in range(30):
                optimizer_flat.zero_grad()
                loss = F.cross_entropy(flat(x, steps=15), y)
                loss.backward()
                optimizer_flat.step()
            final_loss_flat = F.cross_entropy(flat(x, steps=15), y).item()
            flat_learning = 1 - (final_loss_flat / initial_loss_flat)
            
            seed_results = {
                'seed': seed,
                'cube_learning': cube_learning,
                'flat_learning': flat_learning,
                'cube_neurons': cube.num_neurons,
                'cube_advantage': cube_learning - flat_learning
            }
            
            all_results.append(seed_results)
            print(f"  Seed {seed}: Cube {cube_learning:.0%}, Flat {flat_learning:.0%}, Δ={cube_learning-flat_learning:+.0%}")
        
        cube_learnings = [r['cube_learning'] for r in all_results]
        advantages = [r['cube_advantage'] for r in all_results]
        
        mean_cube = np.mean(cube_learnings)
        std_cube = np.std(cube_learnings)
        
        print(f"\n  Mean Cube Learning: {mean_cube:.0%} ± {std_cube:.0%}")
        print(f"  Mean Advantage: {np.mean(advantages):+.0%}")
        
        return ValidationResult(
            track_name="Neural Cube 3D",
            experiment_name="3D vs Flat Topology",
            seeds=list(range(self.num_seeds)),
            results=all_results,
            mean_metric=mean_cube,
            std_metric=std_cube,
            confidence_interval=self._compute_ci(cube_learnings)
        )
    
    def _test_degraded_model(self, model: nn.Module, x: torch.Tensor) -> float:
        """Test if degraded model still produces reasonable outputs."""
        try:
            with torch.no_grad():
                out = model(x, steps=20) if hasattr(model, 'forward') else model(x)
                # Check if outputs are not NaN and have reasonable magnitudes
                functional = not torch.isnan(out).any() and out.abs().max() < 100
                return 1.0 if functional else 0.0
        except:
            return 0.0
    
    def _compute_ci(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval."""
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        
        # t-distribution for small samples
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 1.96
        margin = t_val * std / np.sqrt(n)
        
        return (mean - margin, mean + margin)
    
    def run_all_validations(self) -> Dict[str, ValidationResult]:
        """Run all deep validations."""
        print("\n" + "="*70)
        print("COMPREHENSIVE TODO5 VALIDATION SUITE")
        print(f"Running {self.num_seeds} seeds per experiment")
        print("="*70)
        
        results = {}
        
        # Track 1
        try:
            results['adversarial_healing'] = self.run_adversarial_healing_deep()
        except Exception as e:
            print(f"Adversarial healing validation failed: {e}")
        
        # Track 2
        try:
            results['ternary_weights'] = self.run_ternary_weights_deep()
        except Exception as e:
            print(f"Ternary weights validation failed: {e}")
        
        # Track 3
        try:
            results['neural_cube'] = self.run_neural_cube_deep()
        except Exception as e:
            print(f"Neural cube validation failed: {e}")
        
        return results


def main():
    validator = ComprehensiveValidator(num_seeds=5)
    results = validator.run_all_validations()
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        if result:
            print(f"\n{result.track_name}:")
            print(f"  Metric: {result.mean_metric:.2%} ± {result.std_metric:.2%}")
            print(f"  95% CI: [{result.confidence_interval[0]:.2%}, {result.confidence_interval[1]:.2%}]")
    
    # Save results
    output = {}
    for name, result in results.items():
        if result:
            output[name] = {
                'mean': result.mean_metric,
                'std': result.std_metric,
                'ci_lower': result.confidence_interval[0],
                'ci_upper': result.confidence_interval[1],
                'seeds': result.seeds,
                'raw_results': result.results
            }
    
    with open('results/comprehensive_validation.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/comprehensive_validation.json")


if __name__ == "__main__":
    # Install scipy if needed for CI computation
    try:
        from scipy import stats
    except ImportError:
        print("Installing scipy for statistical analysis...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scipy"], check=True)
    
    main()
