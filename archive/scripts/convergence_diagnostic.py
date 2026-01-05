#!/usr/bin/env python3
"""
Convergence Diagnostic Tool

Measure equilibrium convergence for different input dimensions.
Tests Hypothesis 1: MNIST needs more steps to reach equilibrium.
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY
from src.benchmarks import get_task


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def measure_convergence(model, x, max_steps=200, threshold=1e-4):
    """Measure convergence delta per step."""
    model.eval()
    
    with torch.no_grad():
        h = model.embed(x)
        deltas = []
        energies = []
        
        for step in range(max_steps):
            # Measure energy
            try:
                energy = model.energy(h, x).item()
                energies.append(energy)
            except:
                energies.append(0)
            
            # Step and measure change
            h_prev = h.clone()
            h, _ = model.forward_step(h, x)
            
            delta = (h - h_prev).norm(dim=-1).mean().item()
            deltas.append(delta)
            
            if delta < threshold:
                return step + 1, deltas, energies
        
        return max_steps, deltas, energies


def analyze_convergence(model_name='TPEqProp', tasks=['digits', 'mnist'], 
                        max_steps=200, n_samples=32):
    """Analyze convergence across different tasks."""
    
    print(f"\n🔬 Convergence Analysis: {model_name}")
    print(f"   Tasks: {tasks}, Max steps: {max_steps}")
    print("="*70)
    
    results = {}
    
    for task_name in tasks:
        print(f"\n📊 Task: {task_name}")
        
        # Get data
        task = get_task(task_name)
        config = task.get_config()
        train_loader, _, _ = task.get_loaders()
        
        # Get sample
        x, y = next(iter(train_loader))
        x = x[:n_samples].to(DEVICE)
        
        # Create model
        model = MODEL_REGISTRY[model_name](
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            use_spectral_norm=True
        ).to(DEVICE)
        
        # Measure convergence
        steps_to_conv, deltas, energies = measure_convergence(model, x, max_steps)
        
        print(f"   Input dim: {config.input_dim}")
        print(f"   Converged at step: {steps_to_conv}/{max_steps}")
        print(f"   Final delta: {deltas[-1]:.2e}")
        print(f"   Steps/dim ratio: {steps_to_conv/config.input_dim:.4f}")
        
        results[task_name] = {
            'input_dim': config.input_dim,
            'steps_to_convergence': steps_to_conv,
            'deltas': deltas,
            'energies': energies,
            'final_delta': deltas[-1]
        }
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for task_name, data in results.items():
        steps = range(len(data['deltas']))
        
        # Delta plot
        ax1.semilogy(steps, data['deltas'], label=f"{task_name} ({data['input_dim']}d)", linewidth=2)
        
        # Energy plot
        if max(data['energies']) > 0:
            ax2.plot(steps, data['energies'], label=f"{task_name}", linewidth=2)
    
    ax1.set_xlabel('Equilibrium Step')
    ax1.set_ylabel('||h_t - h_{t-1}|| (log scale)')
    ax1.set_title('Convergence Speed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e-4, color='r', linestyle='--', label='Threshold', alpha=0.5)
    
    ax2.set_xlabel('Equilibrium Step')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Descent')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path('results/convergence_analysis.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Plot saved to: {output_path}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    dims = [results[t]['input_dim'] for t in tasks]
    steps = [results[t]['steps_to_convergence'] for t in tasks]
    
    if len(tasks) == 2:
        dim_ratio = dims[1] / dims[0]
        step_ratio = steps[1] / steps[0]
        
        print(f"\nDimension ratio: {dim_ratio:.1f}x")
        print(f"Steps ratio: {step_ratio:.1f}x")
        print(f"Expected (sqrt scaling): {np.sqrt(dim_ratio):.1f}x")
        print(f"Expected (linear scaling): {dim_ratio:.1f}x")
        
        if step_ratio >= dim_ratio * 0.5:
            print("\n⚠️  FINDING: Steps scale linearly or worse with dimension")
            print("    → Need dimension-adaptive step count!")
        elif step_ratio >= np.sqrt(dim_ratio) * 0.8:
            print("\n✓ FINDING: Steps scale with sqrt(dimension)")
            print("    → Reasonable, but still need more steps for MNIST")
        else:
            print("\n✓ FINDING: Sub-sqrt scaling")
            print("    → Architecture issue, not equilibrium")
    
    return results


def test_more_steps(model_name='TPEqProp', task='mnist', step_counts=[25, 50, 100, 200]):
    """Test if more steps improve MNIST accuracy."""
    
    print(f"\n🧪 Testing Step Counts: {model_name} on {task}")
    print("="*70)
    
    from src.benchmarks import get_task
    from src.training import EqPropTrainer
    import torch.optim as optim
    
    task_obj = get_task(task)
    config = task_obj.get_config()
    train_loader, _, test_loader = task_obj.get_loaders()
    
    results = []
    
    for steps in step_counts:
        print(f"\n📊 Testing {steps} steps...")
        
        # Create model
        model = MODEL_REGISTRY[model_name](
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            use_spectral_norm=True
        ).to(DEVICE)
        
        # Quick train (1 epoch)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trainer = EqPropTrainer(model, optimizer, beta=0.15, max_steps=steps)
        
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            try:
                trainer.step(data, target)
            except:
                continue
        
        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data, steps=steps)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        acc = correct / total
        results.append({'steps': steps, 'accuracy': acc})
        print(f"   Accuracy: {acc:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Steps':<10} {'Accuracy':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['steps']:<10} {r['accuracy']:>9.1%}")
    
    improvement = results[-1]['accuracy'] - results[0]['accuracy']
    print(f"\nImprovement: {improvement:.1%} ({results[0]['steps']} → {results[-1]['steps']} steps)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Convergence diagnostics')
    parser.add_argument('--model', type=str, default='TPEqProp')
    parser.add_argument('--mode', type=str, default='analyze', 
                       choices=['analyze', 'test_steps'])
    parser.add_argument('--tasks', type=str, default='digits,mnist')
    parser.add_argument('--max-steps', type=int, default=200)
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        tasks = args.tasks.split(',')
        analyze_convergence(args.model, tasks, args.max_steps)
    else:
        test_more_steps(args.model, 'mnist', [25, 50, 100, 200])


if __name__ == '__main__':
    main()
