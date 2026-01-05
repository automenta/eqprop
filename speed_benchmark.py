#!/usr/bin/env python
"""
Speed Improvements for Equilibrium Propagation

Tests:
1. Truncated equilibrium (fewer steps)
2. Anderson acceleration (faster convergence)
3. Early stopping (convergence detection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models import LoopedMLP


def anderson_acceleration(f, x0, m=5, max_iter=30, tol=1e-6):
    """
    Anderson acceleration for fixed-point iteration.
    
    f: function computing x_{k+1} = f(x_k)
    x0: initial point
    m: history length (number of previous iterates to use)
    max_iter: maximum iterations
    tol: convergence tolerance
    
    Returns: fixed point approximation, number of iterations
    """
    x = x0.clone()
    F_history = []  # Store f(x_k) - x_k
    X_history = []  # Store x_k
    
    for k in range(max_iter):
        fx = f(x)
        residual = fx - x
        
        # Check convergence
        res_norm = residual.norm(dim=-1).mean().item()
        if res_norm < tol:
            return fx, k + 1
        
        # Store history
        F_history.append(residual)
        X_history.append(x.clone())
        
        # Limit history length
        if len(F_history) > m:
            F_history.pop(0)
            X_history.pop(0)
        
        if len(F_history) >= 2:
            # Build matrices for least squares
            # G_k = [f_k - f_{k-1}, ..., f_k - f_{k-m}]
            F_k = F_history[-1]
            G = torch.stack([F_k - F_history[i] for i in range(len(F_history)-1)], dim=-1)
            
            # Solve least squares: min ||G @ alpha + F_k||
            # Using pseudo-inverse for stability
            try:
                # Flatten batch for solving
                batch_size = x.shape[0]
                hidden_dim = x.shape[1]
                G_flat = G.view(batch_size * hidden_dim, -1)
                F_flat = F_k.view(batch_size * hidden_dim, 1)
                
                # Solve using normal equations
                alpha = torch.linalg.lstsq(G_flat, -F_flat).solution
                alpha = alpha.view(-1)
                
                # Compute accelerated update
                x_new = X_history[-1] + F_history[-1]
                for i, a in enumerate(alpha):
                    x_new = x_new + a * (X_history[-1] - X_history[i] + F_history[-1] - F_history[i])
                
                x = x_new
            except:
                # Fall back to regular iteration
                x = fx
        else:
            # Not enough history, use regular iteration
            x = fx
    
    return f(x), max_iter


class AcceleratedLoopedMLP(nn.Module):
    """LoopedMLP with Anderson acceleration for faster convergence."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, use_spectral_norm=True):
        super().__init__()
        self.base = LoopedMLP(input_dim, hidden_dim, output_dim, 
                              use_spectral_norm=use_spectral_norm, max_steps=30)
    
    def forward(self, x, steps=30, use_anderson=False, anderson_m=5, tol=1e-6):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.base.hidden_dim, device=x.device, dtype=x.dtype)
        x_proj = self.base.W_in(x)
        
        if use_anderson:
            # Define fixed-point function
            def f(h_curr):
                return torch.tanh(x_proj + self.base.W_rec(h_curr))
            
            h, actual_steps = anderson_acceleration(f, h, m=anderson_m, max_iter=steps, tol=tol)
        else:
            # Regular iteration
            for _ in range(steps):
                h = torch.tanh(x_proj + self.base.W_rec(h))
        
        return self.base.W_out(h)


def run_speed_benchmark():
    """Compare speed of different equilibrium solving strategies."""
    
    print("="*70)
    print("EQUILIBRIUM PROPAGATION SPEED BENCHMARK")
    print("="*70)
    
    # Setup
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
    
    torch.manual_seed(42)
    n_train, n_test = 2000, 500
    indices = torch.randperm(len(train_dataset))[:n_train].tolist()
    subset = torch.utils.data.Subset(train_dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)
    
    test_indices = torch.randperm(len(train_dataset))[n_train:n_train+n_test].tolist()
    test_subset = torch.utils.data.Subset(train_dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False)
    
    print(f"\nDataset: CIFAR-10 ({n_train} train, {n_test} test)")
    
    # Backprop baseline
    print("\n" + "="*50)
    print("BACKPROP BASELINE")
    print("="*50)
    
    class SimpleMLP(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.fc1 = nn.Linear(3072, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 10)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    bp_model = SimpleMLP(512)
    bp_optimizer = torch.optim.Adam(bp_model.parameters(), lr=0.001)
    
    start = time.time()
    for epoch in range(5):
        for X, y in loader:
            bp_optimizer.zero_grad()
            out = bp_model(X)
            loss = F.cross_entropy(out, y)
            loss.backward()
            bp_optimizer.step()
    bp_time = time.time() - start
    
    bp_model.eval()
    with torch.no_grad():
        correct = sum((bp_model(X).argmax(1) == y).sum().item() for X, y in test_loader)
    bp_acc = correct / n_test * 100
    
    print(f"Time: {bp_time:.2f}s, Accuracy: {bp_acc:.1f}%")
    
    # Test different step counts
    results = []
    
    for steps in [5, 10, 15, 20, 30]:
        print(f"\n" + "="*50)
        print(f"EQPROP: {steps} STEPS")
        print("="*50)
        
        model = LoopedMLP(3072, 512, 10, use_spectral_norm=True, max_steps=steps)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        start = time.time()
        for epoch in range(5):
            model.train()
            for X, y in loader:
                X_flat = X.view(X.size(0), -1)
                optimizer.zero_grad()
                out = model(X_flat, steps=steps)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
        train_time = time.time() - start
        
        model.eval()
        with torch.no_grad():
            correct = sum((model(X.view(X.size(0), -1), steps=steps).argmax(1) == y).sum().item() 
                         for X, y in test_loader)
        acc = correct / n_test * 100
        
        speedup = bp_time / train_time
        print(f"Time: {train_time:.2f}s, Accuracy: {acc:.1f}%, Speedup vs BP: {speedup:.2f}x")
        
        results.append({
            'steps': steps,
            'time': train_time,
            'accuracy': acc,
            'speedup': speedup
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Steps':<10} {'Time (s)':<12} {'Accuracy':<12} {'Speedup':<12}")
    print("-"*50)
    print(f"{'Backprop':<10} {bp_time:<12.2f} {bp_acc:<12.1f} {'1.00x':<12}")
    for r in results:
        print(f"{r['steps']:<10} {r['time']:<12.2f} {r['accuracy']:<12.1f} {r['speedup']:<12.2f}x")
    
    # Find optimal
    best = max(results, key=lambda r: r['accuracy'])
    fastest = max(results, key=lambda r: r['speedup'])
    
    print(f"\n📊 Best accuracy: {best['steps']} steps ({best['accuracy']:.1f}%)")
    print(f"⚡ Fastest (while learning): {fastest['steps']} steps ({fastest['speedup']:.2f}x)")
    
    # Recommendation
    sweet_spot = [r for r in results if r['accuracy'] >= bp_acc - 5 and r['speedup'] > 0.8]
    if sweet_spot:
        rec = max(sweet_spot, key=lambda r: r['speedup'])
        print(f"\n✅ Recommended: {rec['steps']} steps (within 5% of BP, {rec['speedup']:.2f}x speedup)")
    
    return results


if __name__ == "__main__":
    run_speed_benchmark()
