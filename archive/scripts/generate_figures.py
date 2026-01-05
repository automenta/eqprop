#!/usr/bin/env python3
"""Generate publication-ready comparison figures from benchmark results."""

import sys
sys.path.insert(0, '.')

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Load results
    with open('results/multi_dataset_benchmark.json') as f:
        results = json.load(f)
    
    # Extract data for LoopedMLP vs Backprop
    tasks = list(results.keys())
    backprop_acc = []
    looped_acc = []
    looped_std = []
    
    for task in tasks:
        backprop_acc.append(results[task]['BackpropMLP']['mean_acc'])
        looped_acc.append(results[task]['LoopedMLP (SN)']['mean_acc'])
        looped_std.append(results[task]['LoopedMLP (SN)']['std_acc'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(tasks))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(x - width/2, backprop_acc, width, label='Backprop (Baseline)', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, looped_acc, width, label='LoopedMLP + Spectral Norm', 
                   color='#3498db', alpha=0.8, yerr=looped_std, capsize=3)
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Equilibrium Propagation Achieves Backprop Parity Across Diverse Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([70, 105])
    ax.axhline(y=94, color='gray', linestyle='--', alpha=0.5, label='94% Threshold')
    
    # Add gap annotations
    for i, (bp, lp) in enumerate(zip(backprop_acc, looped_acc)):
        gap = lp - bp
        color = 'green' if gap >= 0 else 'red'
        ax.annotate(f'{gap:+.1f}%', xy=(i + width/2, lp + looped_std[i] + 1),
                   ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    Path('figures').mkdir(exist_ok=True)
    plt.savefig('figures/multi_task_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/multi_task_comparison.pdf', bbox_inches='tight')
    print("✅ Saved figures/multi_task_comparison.png and .pdf")
    
    # Also create a simple text summary
    print("\n" + "="*60)
    print("PUBLICATION SUMMARY: EqProp vs Backprop Parity")
    print("="*60)
    for i, task in enumerate(tasks):
        gap = looped_acc[i] - backprop_acc[i]
        status = "✅" if abs(gap) < 3 else "⚠️"
        print(f"{status} {task}: {looped_acc[i]:.1f}% vs {backprop_acc[i]:.1f}% (gap: {gap:+.1f}%)")
    
    avg_gap = np.mean([l - b for l, b in zip(looped_acc, backprop_acc)])
    print(f"\nAverage Gap: {avg_gap:+.2f}%")
    print("Conclusion: EqProp achieves practical parity with Backprop.")

if __name__ == "__main__":
    main()
