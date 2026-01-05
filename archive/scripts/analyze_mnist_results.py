#!/usr/bin/env python3
"""
Comprehensive Analysis of MNIST Benchmark Results

Analyzes multi-seed MNIST benchmarks to extract insights about:
- Model performance and gaps to baseline
- Statistical significance and variance
- Training efficiency
- Publication readiness
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats


def analyze_mnist_benchmarks(results_path):
    """Analyze MNIST benchmark results and generate insights."""
    
    with open(results_path) as f:
        data = json.load(f)
    
    print("=" * 80)
    print("COMPREHENSIVE MNIST BENCHMARK ANALYSIS")
    print("=" * 80)
    print(f"Data source: {results_path}")
    print()
    
    # Extract model data
    models = {}
    for model_name, results in data.items():
        models[model_name] = {
            'mean': results['mean_acc'],
            'std': results['std_acc'],
            'seeds': results['seeds'],
            'time': results['mean_time'],
            'min': min(results['seeds']),
            'max': max(results['seeds']),
            'range': max(results['seeds']) - min(results['seeds'])
        }
    
    # 1. Performance Summary
    print("## 1. PERFORMANCE SUMMARY")
    print("-" * 80)
    print(f"{'Model':<25} {'Mean':>12} {'Std Dev':>12} {'Range':>12} {'Time':>10}")
    print("-" * 80)
    
    for name, m in models.items():
        print(f"{name:<25} {m['mean']:>11.2f}% {m['std']:>11.2f}% "
              f"{m['range']:>11.2f}% {m['time']:>9.1f}s")
    print()
    
    # 2. Baseline Comparison
    print("## 2. COMPARISON TO BACKPROP BASELINE")
    print("-" * 80)
    
    baseline_name = 'BackpropMLP'
    if baseline_name in models:
        baseline = models[baseline_name]
        
        for name, m in models.items():
            if name == baseline_name:
                continue
            
            gap = baseline['mean'] - m['mean']
            gap_pct = (gap / baseline['mean']) * 100
            
            # Statistical significance test
            t_stat, p_value = stats.ttest_ind(baseline['seeds'], m['seeds'])
            sig = "✅ Significant (p<0.05)" if p_value < 0.05 else "⚠️  Not significant"
            
            print(f"{name}:")
            print(f"  Absolute gap: {gap:.2f}% ({gap_pct:.1f}% relative)")
            print(f"  P-value: {p_value:.4f} - {sig}")
            print(f"  Speed: {m['time']/baseline['time']:.1f}x slower")
            print()
    
    # 3. Stability Analysis
    print("## 3. STABILITY ANALYSIS")
    print("-" * 80)
    
    for name, m in models.items():
        if 'Backprop' in name:
            continue  # Skip baseline
        
        # Variance assessment
        if m['std'] < 0.5:
            stability = "✅ EXCELLENT"
        elif m['std'] < 1.0:
            stability = "✅ GOOD"
        elif m['std'] < 2.0:
            stability = "⚠️  MODERATE"
        else:
            stability = "❌ POOR"
        
        # Coefficient of variation
        cv = (m['std'] / m['mean']) * 100
        
        print(f"{name}:")
        print(f"  Std dev: ±{m['std']:.2f}% - {stability}")
        print(f"  Range: {m['min']:.2f}% to {m['max']:.2f}% (Δ={m['range']:.2f}%)")
        print(f"  Coefficient of variation: {cv:.2f}%")
        print(f"  Individual seeds: {[f'{s:.2f}%' for s in m['seeds']]}")
        print()
    
    # 4. Success Criteria Assessment
    print("## 4. SUCCESS CRITERIA ASSESSMENT (from TODO.md)")
    print("-" * 80)
    
    criteria = [
        ("MNIST accuracy ≥ 94%", lambda m: m['mean'] >= 94.0),
        ("Std deviation < 1%", lambda m: m['std'] < 1.0),
        ("Competitive with backprop (≤3% gap)", 
         lambda m: baseline['mean'] - m['mean'] <= 3.0 if baseline_name in models else False),
    ]
    
    for name, m in models.items():
        if 'Backprop' in name:
            continue
        
        print(f"{name}:")
        for criterion_name, check in criteria:
            result = check(m)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {criterion_name}: {status}")
        print()
    
    # 5. Key Insights
    print("## 5. KEY INSIGHTS & FINDINGS")
    print("-" * 80)
    
    # Find best EqProp model
    eqprop_models = {k: v for k, v in models.items() if 'Backprop' not in k}
    if eqprop_models:
        best_name = max(eqprop_models.keys(), key=lambda k: eqprop_models[k]['mean'])
        best = eqprop_models[best_name]
        most_stable_name = min(eqprop_models.keys(), key=lambda k: eqprop_models[k]['std'])
        most_stable = eqprop_models[most_stable_name]
        
        print(f"✓ Best performing EqProp model: {best_name}")
        print(f"  - Accuracy: {best['mean']:.2f}% ± {best['std']:.2f}%")
        print(f"  - Gap to baseline: {baseline['mean'] - best['mean']:.2f}%")
        print()
        
        print(f"✓ Most stable EqProp model: {most_stable_name}")
        print(f"  - Std dev: ±{most_stable['std']:.2f}%")
        print(f"  - Accuracy: {most_stable['mean']:.2f}%")
        print()
        
        # Check if any model meets all criteria
        all_pass = []
        for name, m in eqprop_models.items():
            if all(check(m) for _, check in criteria):
                all_pass.append(name)
        
        if all_pass:
            print(f"✓ Models meeting ALL success criteria: {', '.join(all_pass)}")
        else:
            print("⚠️  No models meet all success criteria")
    
    print()
    print("=" * 80)
    
    return models


def main():
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = 'results/competitive_benchmark_5seed.json'
    
    if not Path(results_path).exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)
    
    analyze_mnist_benchmarks(results_path)


if __name__ == "__main__":
    main()
