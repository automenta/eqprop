#!/usr/bin/env python
"""
Comprehensive CIFAR-10 Hyperparameter Search

Tests both ConvEqProp and LoopedMLP with various hyperparameters
to find the best configuration for each architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))
from models import ConvEqProp, LoopedMLP

@dataclass
class Result:
    arch: str
    hidden: int
    eq_steps: int
    lr: float
    epochs: int
    train_acc: float
    test_acc: float
    time: float
    params: int

def evaluate_model(model, loader, eq_steps, flatten=False):
    """Evaluate model accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            if flatten:
                X = X.view(X.size(0), -1)
            out = model(X, steps=eq_steps)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return correct / total

def train_and_evaluate(arch_type, hidden, eq_steps, lr, epochs, train_loader, test_loader):
    """Train and evaluate a single configuration."""
    
    # Create model
    if arch_type == "ConvEqProp":
        model = ConvEqProp(input_channels=3, hidden_channels=hidden, output_dim=10, 
                          use_spectral_norm=True)
        flatten = False
    else:  # LoopedMLP
        model = LoopedMLP(input_dim=3072, hidden_dim=hidden, output_dim=10,
                         use_spectral_norm=True, max_steps=eq_steps)
        flatten = True
    
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            if flatten:
                X = X.view(X.size(0), -1)
            optimizer.zero_grad()
            out = model(X, steps=eq_steps)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
    
    train_time = time.time() - start
    
    # Evaluate
    train_acc = evaluate_model(model, train_loader, eq_steps, flatten)
    test_acc = evaluate_model(model, test_loader, eq_steps, flatten)
    
    return Result(
        arch=arch_type,
        hidden=hidden,
        eq_steps=eq_steps,
        lr=lr,
        epochs=epochs,
        train_acc=train_acc,
        test_acc=test_acc,
        time=train_time,
        params=params
    )

def main():
    print("="*80)
    print("COMPREHENSIVE CIFAR-10 HYPERPARAMETER SEARCH")
    print("="*80)
    
    # Load data
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform)
    
    # Use subset for speed
    torch.manual_seed(42)
    n_train, n_test = 5000, 1000
    train_indices = torch.randperm(len(train_dataset))[:n_train].tolist()
    test_indices = torch.randperm(len(test_dataset))[:n_test].tolist()
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False)
    
    results = []
    
    # Hyperparameter grids
    configs = [
        # ConvEqProp configurations
        ("ConvEqProp", 32, 20, 0.001, 10),
        ("ConvEqProp", 48, 20, 0.001, 10),
        ("ConvEqProp", 64, 15, 0.001, 10),
        ("ConvEqProp", 64, 20, 0.0005, 10),
        ("ConvEqProp", 64, 20, 0.002, 10),
        
        # LoopedMLP configurations
        ("LoopedMLP", 512, 20, 0.001, 5),
        ("LoopedMLP", 768, 20, 0.001, 5),
        ("LoopedMLP", 1024, 20, 0.001, 5),
        ("LoopedMLP", 512, 25, 0.001, 5),
        ("LoopedMLP", 512, 15, 0.001, 5),
        ("LoopedMLP", 512, 20, 0.0005, 5),
        ("LoopedMLP", 512, 20, 0.002, 5),
    ]
    
    total = len(configs)
    for i, (arch, hidden, steps, lr, epochs) in enumerate(configs, 1):
        print(f"\n[{i}/{total}] Testing {arch}: hidden={hidden}, steps={steps}, lr={lr}, epochs={epochs}")
        
        result = train_and_evaluate(arch, hidden, steps, lr, epochs, train_loader, test_loader)
        results.append(result)
        
        print(f"  → Train: {result.train_acc*100:.1f}%, Test: {result.test_acc*100:.1f}%, "
              f"Time: {result.time:.1f}s, Params: {result.params:,}")
    
    # Analysis
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Best results per architecture
    conv_results = [r for r in results if r.arch == "ConvEqProp"]
    mlp_results = [r for r in results if r.arch == "LoopedMLP"]
    
    best_conv = max(conv_results, key=lambda r: r.test_acc)
    best_mlp = max(mlp_results, key=lambda r: r.test_acc)
    
    print("\nBest ConvEqProp:")
    print(f"  Config: hidden={best_conv.hidden}, steps={best_conv.eq_steps}, lr={best_conv.lr}")
    print(f"  Test Acc: {best_conv.test_acc*100:.1f}%")
    print(f"  Time: {best_conv.time:.1f}s")
    
    print("\nBest LoopedMLP:")
    print(f"  Config: hidden={best_mlp.hidden}, steps={best_mlp.eq_steps}, lr={best_mlp.lr}")
    print(f"  Test Acc: {best_mlp.test_acc*100:.1f}%")
    print(f"  Time: {best_mlp.time:.1f}s")
    
    # Detailed table
    print("\n" + "="*120)
    print("FULL RESULTS TABLE")
    print("="*120)
    print(f"{'Architecture':<12} {'Hidden':>6} {'Steps':>5} {'LR':>6} {'Epochs':>6} "
          f"{'Train%':>6} {'Test%':>6} {'Time(s)':>8} {'Params':>10}")
    print("-"*120)
    
    for r in sorted(results, key=lambda x: (x.arch, -x.test_acc)):
        print(f"{r.arch:<12} {r.hidden:>6} {r.eq_steps:>5} {r.lr:>6.4f} {r.epochs:>6} "
              f"{r.train_acc*100:>6.1f} {r.test_acc*100:>6.1f} {r.time:>8.1f} {r.params:>10,}")
    
    # Save results
    output_path = Path(__file__).parent / "results" / "hyperparam_search.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'results': [asdict(r) for r in results],
            'best_conv': asdict(best_conv),
            'best_mlp': asdict(best_mlp)
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    
    winner = "ConvEqProp" if best_conv.test_acc > best_mlp.test_acc else "LoopedMLP"
    margin = abs(best_conv.test_acc - best_mlp.test_acc) * 100
    
    print(f"\n🏆 Winner: {winner}")
    print(f"   Margin: {margin:.1f}%")
    print(f"\n   ConvEqProp best: {best_conv.test_acc*100:.1f}% (hidden={best_conv.hidden})")
    print(f"   LoopedMLP best:  {best_mlp.test_acc*100:.1f}% (hidden={best_mlp.hidden})")
    
    if margin < 2:
        print(f"\n   → Architectures perform similarly (< 2% difference)")
    
    # Training efficiency
    conv_speed = best_conv.time / best_conv.epochs
    mlp_speed = best_mlp.time / best_mlp.epochs
    
    print(f"\n📊 Training Speed:")
    print(f"   ConvEqProp: {conv_speed:.1f}s/epoch")
    print(f"   LoopedMLP:  {mlp_speed:.1f}s/epoch")
    print(f"   Speedup: {conv_speed/mlp_speed:.1f}× (LoopedMLP faster)" if mlp_speed < conv_speed 
          else f"   Speedup: {mlp_speed/conv_speed:.1f}× (ConvEqProp faster)")

if __name__ == "__main__":
    main()
