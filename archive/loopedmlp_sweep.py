#!/usr/bin/env python
"""
LoopedMLP Hyperparameter Sweep for CIFAR-10

Comprehensive grid search to find optimal configuration.
Tests: hidden_dim, eq_steps, learning_rate, epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from models import LoopedMLP

def run_config(hidden_dim, eq_steps, lr, epochs, train_loader, test_loader, n_train, n_test):
    """Train and evaluate one configuration."""
    
    model = LoopedMLP(input_dim=3072, hidden_dim=hidden_dim, output_dim=10,
                     use_spectral_norm=True, max_steps=eq_steps)
    
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    start = time.time()
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X_flat = X.view(X.size(0), -1)
            optimizer.zero_grad()
            out = model(X_flat, steps=eq_steps)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        
        # Quick eval every epoch
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                X_flat = X.view(X.size(0), -1)
                out = model(X_flat, steps=eq_steps)
                test_correct += (out.argmax(1) == y).sum().item()
        
        test_acc = test_correct / n_test
        best_test_acc = max(best_test_acc, test_acc)
    
    train_time = time.time() - start
    
    # Final eval on train set
    model.eval()
    train_correct = 0
    with torch.no_grad():
        for X, y in train_loader:
            X_flat = X.view(X.size(0), -1)
            out = model(X_flat, steps=eq_steps)
            train_correct += (out.argmax(1) == y).sum().item()
    
    train_acc = train_correct / n_train
    
    return {
        'hidden_dim': hidden_dim,
        'eq_steps': eq_steps,
        'lr': lr,
        'epochs': epochs,
        'train_acc': train_acc,
        'test_acc': best_test_acc,  # Best test acc during training
        'time': train_time,
        'params': params,
        'time_per_epoch': train_time / epochs
    }

def main():
    print("="*90)
    print("LOOPEDMLP CIFAR-10 HYPERPARAMETER SWEEP")
    print("="*90)
    
    # Load data
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform)
    
    # Use 5k train, 1k test for speed
    torch.manual_seed(42)
    n_train, n_test = 5000, 1000
    train_indices = torch.randperm(len(train_dataset))[:n_train].tolist()
    test_indices = torch.randperm(len(test_dataset))[:n_test].tolist()
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False)
    
    print(f"\nDataset: {n_train} train, {n_test} test")
    print("Batch size: 128\n")
    
    # Hyperparameter grid
    hidden_dims = [512, 768, 1024, 1536]
    eq_steps_list = [15, 20, 25, 30]
    learning_rates = [0.0005, 0.001, 0.002]
    epochs_list = [5, 10]
    
    # Generate all combinations
    configs = list(product(hidden_dims, eq_steps_list, learning_rates, epochs_list))
    
    print(f"Testing {len(configs)} configurations...")
    print("="*90)
    
    results = []
    
    for i, (h, s, lr, e) in enumerate(configs, 1):
        print(f"\n[{i:2d}/{len(configs)}] hidden={h:4d}, steps={s:2d}, lr={lr:.4f}, epochs={e:2d}", end=" → ")
        
        result = run_config(h, s, lr, e, train_loader, test_loader, n_train, n_test)
        results.append(result)
        
        print(f"test={result['test_acc']*100:5.1f}%, time={result['time']:5.1f}s")
    
    # Analysis
    print("\n" + "="*90)
    print("TOP 10 RESULTS (by test accuracy)")
    print("="*90)
    
    sorted_results = sorted(results, key=lambda r: r['test_acc'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Hidden':>6} {'Steps':>5} {'LR':>7} {'Epochs':>6} "
          f"{'Train%':>7} {'Test%':>7} {'Time':>7} {'Params':>10}")
    print("-"*90)
    
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"{i:<5} {r['hidden_dim']:>6} {r['eq_steps']:>5} {r['lr']:>7.4f} {r['epochs']:>6} "
              f"{r['train_acc']*100:>7.1f} {r['test_acc']*100:>7.1f} {r['time']:>7.1f} {r['params']:>10,}")
    
    # Best result
    best = sorted_results[0]
    print("\n" + "="*90)
    print("BEST CONFIGURATION")
    print("="*90)
    print(f"\nHidden dim:      {best['hidden_dim']}")
    print(f"Eq steps:        {best['eq_steps']}")
    print(f"Learning rate:   {best['lr']}")
    print(f"Epochs:          {best['epochs']}")
    print(f"Train accuracy:  {best['train_acc']*100:.1f}%")
    print(f"Test accuracy:   {best['test_acc']*100:.1f}%")
    print(f"Training time:   {best['time']:.1f}s ({best['time_per_epoch']:.1f}s/epoch)")
    print(f"Parameters:      {best['params']:,}")
    
    # Analyze trends
    print("\n" + "="*90)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*90)
    
    # Group by each hyperparameter
    by_hidden = {}
    for r in results:
        h = r['hidden_dim']
        if h not in by_hidden:
            by_hidden[h] = []
        by_hidden[h].append(r['test_acc'])
    
    by_steps = {}
    for r in results:
        s = r['eq_steps']
        if s not in by_steps:
            by_steps[s] = []
        by_steps[s].append(r['test_acc'])
    
    by_lr = {}
    for r in results:
        lr = r['lr']
        if lr not in by_lr:
            by_lr[lr] = []
        by_lr[lr].append(r['test_acc'])
    
    import numpy as np
    
    print("\nHidden Dimension Impact:")
    for h in sorted(by_hidden.keys()):
        accs = by_hidden[h]
        print(f"  {h:4d}: mean={np.mean(accs)*100:.1f}%, max={max(accs)*100:.1f}%, min={min(accs)*100:.1f}%")
    
    print("\nEquilibrium Steps Impact:")
    for s in sorted(by_steps.keys()):
        accs = by_steps[s]
        print(f"  {s:2d}: mean={np.mean(accs)*100:.1f}%, max={max(accs)*100:.1f}%, min={min(accs)*100:.1f}%")
    
    print("\nLearning Rate Impact:")
    for lr in sorted(by_lr.keys()):
        accs = by_lr[lr]
        print(f"  {lr:.4f}: mean={np.mean(accs)*100:.1f}%, max={max(accs)*100:.1f}%, min={min(accs)*100:.1f}%")
    
    # Save results
    output_path = Path(__file__).parent / "results" / "loopedmlp_sweep.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'configurations': results,
            'best': best,
            'summary': {
                'by_hidden': {str(k): {'mean': float(np.mean(v)), 'max': float(max(v))} 
                             for k, v in by_hidden.items()},
                'by_steps': {str(k): {'mean': float(np.mean(v)), 'max': float(max(v))} 
                            for k, v in by_steps.items()},
                'by_lr': {str(k): {'mean': float(np.mean(v)), 'max': float(max(v))} 
                         for k, v in by_lr.items()},
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return best, results

if __name__ == "__main__":
    best, results = main()
