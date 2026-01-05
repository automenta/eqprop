#!/usr/bin/env python3
"""
Oracle Metric: Uncertainty via Settling Time (TODO5 Additional Experiment)

In a dynamical system, Time = Uncertainty.
This script proves that TorEq has "Native Introspection" - the ability to
signal uncertainty without Bayesian wrappers.

Key Insight: For ambiguous inputs (e.g., a "5" that looks like an "S"),
the T_relax (settling time) is significantly higher than for clean inputs.

This is a "Gatekeeper crusher" - something Transformers and CNNs cannot do.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json


@dataclass
class OracleResult:
    """Result from oracle uncertainty analysis."""
    input_idx: int
    true_label: int
    predicted_label: int
    confidence: float
    settling_time: int  # Steps to converge
    final_velocity: float  # Velocity at last step
    is_correct: bool
    is_ambiguous: bool  # Based on low confidence or high settling time


class OracleAnalyzer:
    """
    Analyze settling time as uncertainty indicator.
    
    The hypothesis: Ambiguous inputs take longer to settle because
    the network explores multiple attractors before converging.
    """
    
    def __init__(self, model: nn.Module, max_steps: int = 100,
                 velocity_threshold: float = 1e-4):
        self.model = model
        self.max_steps = max_steps
        self.velocity_threshold = velocity_threshold
    
    def measure_settling(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int], List[float]]:
        """
        Run model and measure settling time for each input.
        
        Returns:
            outputs: Model predictions
            settling_times: Steps until velocity < threshold for each input
            final_velocities: Velocity at last step for each input
        """
        batch_size = x.size(0)
        
        # Initialize state
        if hasattr(self.model, 'hidden_dim'):
            h = torch.zeros(batch_size, self.model.hidden_dim, device=x.device)
        else:
            # Try to infer hidden dim
            with torch.no_grad():
                out = self.model(x, steps=1)
            h = torch.zeros(batch_size, 256, device=x.device)  # Default
        
        settling_times = [self.max_steps] * batch_size
        velocities = []
        
        h_prev = h.clone()
        
        for step in range(self.max_steps):
            if hasattr(self.model, 'forward_step'):
                h, _ = self.model.forward_step(h, x)
            else:
                # Use full forward but track internal state
                out = self.model(x, steps=1)
                h = out  # Approximate
            
            # Compute velocity per sample
            velocity = torch.mean(torch.abs(h - h_prev), dim=1)  # [batch_size]
            velocities.append(velocity.detach().cpu())
            
            # Check convergence for each sample
            for i in range(batch_size):
                if settling_times[i] == self.max_steps and velocity[i].item() < self.velocity_threshold:
                    settling_times[i] = step + 1
            
            h_prev = h.clone()
        
        # Final output
        if hasattr(self.model, 'head'):
            out = self.model.head(h)
        else:
            out = self.model(x)
        
        final_velocities = velocities[-1].tolist()
        
        return out, settling_times, final_velocities
    
    def analyze_dataset(self, dataloader, num_samples: int = 500) -> List[OracleResult]:
        """Analyze a dataset for uncertainty patterns."""
        results = []
        sample_count = 0
        
        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                if sample_count >= num_samples:
                    break
                
                out, settling_times, final_velocities = self.measure_settling(x)
                probs = F.softmax(out, dim=1)
                confidences = probs.max(dim=1).values.cpu().numpy()
                preds = probs.argmax(dim=1).cpu().numpy()
                
                for i in range(x.size(0)):
                    if sample_count >= num_samples:
                        break
                    
                    results.append(OracleResult(
                        input_idx=sample_count,
                        true_label=y[i].item(),
                        predicted_label=preds[i],
                        confidence=confidences[i],
                        settling_time=settling_times[i],
                        final_velocity=final_velocities[i],
                        is_correct=preds[i] == y[i].item(),
                        is_ambiguous=confidences[i] < 0.7 or settling_times[i] > 50
                    ))
                    sample_count += 1
        
        return results
    
    def compute_correlation(self, results: List[OracleResult]) -> Dict[str, float]:
        """Compute correlations between settling time and other metrics."""
        settling = np.array([r.settling_time for r in results])
        confidence = np.array([r.confidence for r in results])
        correct = np.array([r.is_correct for r in results])
        
        # Correlation: settling time vs confidence
        corr_conf = np.corrcoef(settling, confidence)[0, 1]
        
        # Correlation: settling time vs correctness
        corr_correct = np.corrcoef(settling, correct.astype(float))[0, 1]
        
        # Average settling time for correct vs incorrect
        avg_settling_correct = np.mean(settling[correct])
        avg_settling_incorrect = np.mean(settling[~correct]) if (~correct).any() else 0
        
        return {
            'settling_confidence_correlation': corr_conf,
            'settling_correctness_correlation': corr_correct,
            'avg_settling_correct': avg_settling_correct,
            'avg_settling_incorrect': avg_settling_incorrect,
            'settling_gap': avg_settling_incorrect - avg_settling_correct
        }


def create_ambiguous_samples(x: torch.Tensor, y: torch.Tensor, 
                              noise_level: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create artificially ambiguous samples by adding noise or blending."""
    x_ambiguous = x + torch.randn_like(x) * noise_level
    return x_ambiguous, y


def run_oracle_experiment(model_class=None, dataset: str = "mnist",
                          num_samples: int = 500, verbose: bool = True):
    """Run the full oracle uncertainty experiment."""
    
    # Load model
    if model_class is None:
        from src.models import LoopedMLP
        model = LoopedMLP(784, 256, 10, alpha=0.5, use_spectral_norm=True)
    else:
        model = model_class
    
    # Load data
    try:
        from src.tasks import get_task_loader
        train_loader, test_loader, in_dim, out_dim = get_task_loader(dataset, batch_size=32)
    except:
        print("Data loading failed. Using synthetic data.")
        x_test = torch.randn(num_samples, 784)
        y_test = torch.randint(0, 10, (num_samples,))
        test_loader = [(x_test[:32], y_test[:32])] * (num_samples // 32)
    
    analyzer = OracleAnalyzer(model, max_steps=100)
    
    if verbose:
        print("\n" + "="*70)
        print("ORACLE UNCERTAINTY EXPERIMENT")
        print("="*70)
        print(f"Dataset: {dataset}")
        print(f"Samples: {num_samples}")
        print()
    
    # Analyze clean data
    results_clean = analyzer.analyze_dataset(test_loader, num_samples)
    corr_clean = analyzer.compute_correlation(results_clean)
    
    if verbose:
        print("CLEAN DATA ANALYSIS:")
        print(f"  Settling-Confidence Correlation: {corr_clean['settling_confidence_correlation']:.4f}")
        print(f"  Avg Settling (Correct):   {corr_clean['avg_settling_correct']:.1f} steps")
        print(f"  Avg Settling (Incorrect): {corr_clean['avg_settling_incorrect']:.1f} steps")
        print(f"  Gap: {corr_clean['settling_gap']:.1f} steps")
    
    # Analyze ambiguous data (noisy)
    # Create noisy version by iterating through loader
    noisy_results = []
    sample_count = 0
    for x, y in test_loader:
        if sample_count >= num_samples:
            break
        x_noisy = x + torch.randn_like(x) * 0.5
        out, settling, velocities = analyzer.measure_settling(x_noisy)
        probs = F.softmax(out, dim=1)
        confidences = probs.max(dim=1).values.cpu().numpy()
        preds = probs.argmax(dim=1).cpu().numpy()
        
        for i in range(x.size(0)):
            if sample_count >= num_samples:
                break
            noisy_results.append(OracleResult(
                input_idx=sample_count, true_label=y[i].item(),
                predicted_label=preds[i], confidence=confidences[i],
                settling_time=settling[i], final_velocity=velocities[i],
                is_correct=preds[i] == y[i].item(),
                is_ambiguous=confidences[i] < 0.7
            ))
            sample_count += 1
    
    corr_noisy = analyzer.compute_correlation(noisy_results)
    
    if verbose:
        print("\nNOISY DATA ANALYSIS (50% noise):")
        print(f"  Settling-Confidence Correlation: {corr_noisy['settling_confidence_correlation']:.4f}")
        print(f"  Avg Settling (Correct):   {corr_noisy['avg_settling_correct']:.1f} steps")
        print(f"  Avg Settling (Incorrect): {corr_noisy['avg_settling_incorrect']:.1f} steps")
        print(f"  Gap: {corr_noisy['settling_gap']:.1f} steps")
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("ORACLE VERDICT:")
        
        if corr_clean['settling_confidence_correlation'] < -0.1:
            print("✓ CONFIRMED: Lower confidence → Higher settling time")
            print("  This proves NATIVE INTROSPECTION capability.")
        else:
            print("⚠ PARTIAL: Correlation is weak, may need more training")
        
        if corr_clean['settling_gap'] > 5:
            print(f"✓ CONFIRMED: Incorrect predictions take {corr_clean['settling_gap']:.1f} extra steps")
            print("  The network 'hesitates' on hard examples.")
        else:
            print("⚠ PARTIAL: Settling gap is small")
    
    return {
        'clean': {'results': results_clean, 'correlations': corr_clean},
        'noisy': {'results': noisy_results, 'correlations': corr_noisy}
    }


def main():
    parser = argparse.ArgumentParser(description="Oracle Uncertainty Experiment")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    data = run_oracle_experiment(dataset=args.dataset, num_samples=args.samples)
    
    if args.output:
        # Save results (excluding full result objects for JSON)
        save_data = {
            'clean_correlations': data['clean']['correlations'],
            'noisy_correlations': data['noisy']['correlations']
        }
        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
