#!/usr/bin/env python3
"""
Quick MNIST Hyperparameter Optimization

Fast sweep using 2-3 epochs to identify most impactful parameters.
Tests: lr, beta, steps, hidden_dim, gamma
"""

import argparse
import json
from pathlib import Path
import itertools
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY
from src.training import EqPropTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_mnist_loaders(batch_size=64, limit_batches=None):
    """Get MNIST loaders, optionally limited."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    if limit_batches:
        # Limit dataset size for faster iteration
        train_subset = torch.utils.data.Subset(train_data, 
                                               range(min(limit_batches * batch_size, len(train_data))))
        test_subset = torch.utils.data.Subset(test_data,
                                              range(min(limit_batches * batch_size // 2, len(test_data))))
        train_data, test_data = train_subset, test_subset
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_and_eval(model_name, hidden_dim=256, lr=1e-3, beta=0.22, steps=25, 
                   gamma=None, epochs=2, batch_size=64, limit_batches=None, verbose=False):
    """Train and evaluate with given hyperparameters."""
    
    # Get data
    train_loader, test_loader = get_mnist_loaders(batch_size, limit_batches)
    
    # Create model
    model_class = MODEL_REGISTRY[model_name]
    
    # Handle gamma parameter
    model_kwargs = {
        'input_dim': 784,
        'hidden_dim': hidden_dim,
        'output_dim': 10,
        'use_spectral_norm': True
    }
    
    if gamma is not None:
        model_kwargs['gamma'] = gamma
    
    try:
        model = model_class(**model_kwargs).to(DEVICE)
    except TypeError:
        # Model doesn't accept some params
        model = model_class(784, hidden_dim, 10).to(DEVICE)
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=steps)
    
    best_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            try:
                trainer.step(data, target)
            except Exception:
                continue
        
        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                try:
                    output = model(data, steps=steps)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                except Exception:
                    continue
        
        acc = correct / max(total, 1)
        best_acc = max(best_acc, acc)
        
        if verbose:
            print(f"    Epoch {epoch+1}/{epochs}: {acc:.1%}")
    
    return best_acc


def grid_search(model_name='TPEqProp', epochs=2, limit_batches=100, verbose=True):
    """Run grid search over key hyperparameters."""
    
    print(f"\n🔍 MNIST Hyperparameter Search: {model_name}")
    print(f"   Epochs: {epochs}, Limited batches: {limit_batches}")
    print(f"   Device: {DEVICE}")
    print("="*70)
    
    # Parameter grid (focused on most impactful)
    param_grid = {
        'lr': [1e-3, 3e-3, 5e-3],
        'beta': [0.15, 0.22, 0.30],
        'steps': [20, 30, 40],
        'hidden_dim': [128, 256],
    }
    
    results = []
    
    # Grid search
    total_combos = np.prod([len(v) for v in param_grid.values()])
    combo_idx = 0
    
    for lr, beta, steps, hidden_dim in itertools.product(
        param_grid['lr'],
        param_grid['beta'],
        param_grid['steps'],
        param_grid['hidden_dim']
    ):
        combo_idx += 1
        
        if verbose:
            print(f"\n[{combo_idx}/{total_combos}] lr={lr:.0e}, β={beta:.2f}, steps={steps}, h={hidden_dim}")
        
        start = time.time()
        acc = train_and_eval(
            model_name, hidden_dim, lr, beta, steps,
            epochs=epochs, limit_batches=limit_batches, verbose=verbose
        )
        elapsed = time.time() - start
        
        result = {
            'lr': lr,
            'beta': beta,
            'steps': steps,
            'hidden_dim': hidden_dim,
            'accuracy': acc,
            'time_s': elapsed
        }
        results.append(result)
        
        if verbose:
            print(f"    → Accuracy: {acc:.1%}, Time: {elapsed:.1f}s")
    
    # Analyze results
    print("\n" + "="*70)
    print("TOP CONFIGURATIONS")
    print("="*70)
    
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"{i:2d}. {r['accuracy']:.1%} | lr={r['lr']:.0e}, β={r['beta']:.2f}, "
              f"steps={r['steps']}, h={r['hidden_dim']}")
    
    # Parameter importance analysis
    print("\n" + "="*70)
    print("PARAMETER IMPORTANCE")
    print("="*70)
    
    for param in ['lr', 'beta', 'steps', 'hidden_dim']:
        # Group by parameter value
        param_impact = {}
        for r in results:
            val = r[param]
            if val not in param_impact:
                param_impact[val] = []
            param_impact[val].append(r['accuracy'])
        
        # Compute mean accuracy per value
        param_means = {k: np.mean(v) for k, v in param_impact.items()}
        best = max(param_means.items(), key=lambda x: x[1])
        worst = min(param_means.items(), key=lambda x: x[1])
        gap = best[1] - worst[1]
        
        print(f"\n{param}:")
        print(f"  Best: {best[0]} → {best[1]:.1%}")
        print(f"  Worst: {worst[0]} → {worst[1]:.1%}")
        print(f"  Impact: {gap:.1%}")
    
    return results


def random_search(model_name='TPEqProp', epochs=2, n_trials=30, limit_batches=100):
    """Random search for faster exploration."""
    
    print(f"\n🎲 Random Search: {model_name} ({n_trials} trials)")
    print("="*70)
    
    results = []
    
    for trial in range(n_trials):
        # Sample random hyperparameters
        lr = 10 ** np.random.uniform(-3.5, -2)  # 10^-3.5 to 10^-2 (0.0003 to 0.01)
        beta = np.random.uniform(0.1, 0.35)
        steps = int(np.random.choice([15, 20, 25, 30, 40, 50]))
        hidden_dim = int(np.random.choice([128, 256, 512]))
        gamma = np.random.uniform(0.3, 0.7)
        
        print(f"\n[{trial+1}/{n_trials}] lr={lr:.1e}, β={beta:.2f}, steps={steps}, "
              f"h={hidden_dim}, γ={gamma:.2f}")
        
        start = time.time()
        acc = train_and_eval(
            model_name, hidden_dim, lr, beta, steps, gamma,
            epochs=epochs, limit_batches=limit_batches, verbose=True
        )
        elapsed = time.time() - start
        
        result = {
            'lr': lr,
            'beta': beta,
            'steps': steps,
            'hidden_dim': hidden_dim,
            'gamma': gamma,
            'accuracy': acc,
            'time_s': elapsed
        }
        results.append(result)
        
        print(f"    → {acc:.1%} ({elapsed:.1f}s)")
    
    # Print best
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n🏆 Best: {best['accuracy']:.1%}")
    print(f"   lr={best['lr']:.1e}, β={best['beta']:.2f}, steps={best['steps']}, "
          f"h={best['hidden_dim']}, γ={best['gamma']:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Quick MNIST hyperparameter optimization')
    parser.add_argument('--model', type=str, default='TPEqProp', help='Model to optimize')
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'random'],
                       help='Search mode')
    parser.add_argument('--epochs', type=int, default=2, help='Epochs per trial')
    parser.add_argument('--trials', type=int, default=30, help='Trials for random search')
    parser.add_argument('--limit-batches', type=int, default=100, 
                       help='Limit batches for speed (None for full dataset)')
    parser.add_argument('--output', type=str, default='results/mnist_hyperparam_sweep.json')
    args = parser.parse_args()
    
    if args.mode == 'grid':
        results = grid_search(args.model, args.epochs, args.limit_batches)
    else:
        results = random_search(args.model, args.epochs, args.trials, args.limit_batches)
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved to {args.output}")


if __name__ == '__main__':
    main()
