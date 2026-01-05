#!/usr/bin/env python
"""
Architecture Experiments for EqProp

Tests various architectures on CIFAR-10:
1. Deeper MLP (multiple hidden layers)
2. Residual connections
3. Different activations (ReLU, GELU)
4. Wider vs deeper trade-offs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from pathlib import Path
from torch.nn.utils.parametrizations import spectral_norm

sys.path.insert(0, str(Path(__file__).parent))


class DeepLoopedMLP(nn.Module):
    """Multi-layer equilibrium network."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, 
                 use_spectral_norm=True, max_steps=20):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        
        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim)
        if use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
        
        # Multiple recurrent layers (each layer recurs with itself)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            if use_spectral_norm:
                layer = spectral_norm(layer)
            self.layers.append(layer)
        
        # Output projection
        self.W_out = nn.Linear(hidden_dim, output_dim)
        if use_spectral_norm:
            self.W_out = spectral_norm(self.W_out)
    
    def forward(self, x, steps=None):
        steps = steps or self.max_steps
        batch_size = x.shape[0]
        
        # Initialize all hidden states
        h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) 
             for _ in range(self.num_layers)]
        
        x_proj = self.W_in(x)
        
        for _ in range(steps):
            # Each layer takes input from previous layer's hidden state
            h_new = []
            for i, layer in enumerate(self.layers):
                if i == 0:
                    h_new.append(torch.tanh(x_proj + layer(h[i])))
                else:
                    h_new.append(torch.tanh(h[i-1] + layer(h[i])))
            h = h_new
        
        return self.W_out(h[-1])


class ResidualLoopedMLP(nn.Module):
    """Equilibrium network with residual connections."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 use_spectral_norm=True, max_steps=20, residual_weight=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.residual_weight = residual_weight
        
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, output_dim)
        
        if use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)
    
    def forward(self, x, steps=None):
        steps = steps or self.max_steps
        batch_size = x.shape[0]
        
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        x_proj = self.W_in(x)
        
        for _ in range(steps):
            h_new = torch.tanh(x_proj + self.W_rec(h))
            h = self.residual_weight * h + (1 - self.residual_weight) * h_new
        
        return self.W_out(h)


class GELULoopedMLP(nn.Module):
    """Equilibrium network with GELU activation."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 use_spectral_norm=True, max_steps=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, output_dim)
        
        # Note: GELU has Lipschitz ~1.1, so spectral norm is more important
        if use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)
    
    def forward(self, x, steps=None):
        steps = steps or self.max_steps
        batch_size = x.shape[0]
        
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        x_proj = self.W_in(x)
        
        for _ in range(steps):
            h = F.gelu(x_proj + self.W_rec(h))
        
        return self.W_out(h)


def train_and_eval(model, train_loader, test_loader, epochs=5, steps=15):
    """Train and evaluate a model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X_flat = X.view(X.size(0), -1)
            optimizer.zero_grad()
            out = model(X_flat, steps=steps)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X_flat = X.view(X.size(0), -1)
            out = model(X_flat, steps=steps)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    
    return correct / total * 100, train_time


def run_architecture_experiments():
    """Test different architectures on CIFAR-10."""
    
    print("="*70)
    print("ARCHITECTURE EXPERIMENTS")
    print("="*70)
    
    from torchvision import datasets, transforms
    from models import LoopedMLP
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform)
    
    torch.manual_seed(42)
    n_train, n_test = 3000, 1000
    
    train_indices = torch.randperm(len(train_dataset))[:n_train].tolist()
    test_indices = torch.randperm(len(test_dataset))[:n_test].tolist()
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False)
    
    print(f"\nDataset: CIFAR-10 ({n_train} train, {n_test} test)")
    
    results = []
    
    # 1. Baseline LoopedMLP
    print("\n" + "-"*50)
    print("1. Baseline LoopedMLP (hidden=512, steps=15)")
    print("-"*50)
    model = LoopedMLP(3072, 512, 10, use_spectral_norm=True, max_steps=15)
    acc, t = train_and_eval(model, train_loader, test_loader, epochs=10, steps=15)
    print(f"   Accuracy: {acc:.1f}%, Time: {t:.1f}s")
    results.append(("Baseline (512, 15 steps)", acc, t, sum(p.numel() for p in model.parameters())))
    
    # 2. Wider model
    print("\n" + "-"*50)
    print("2. Wider LoopedMLP (hidden=1024, steps=15)")
    print("-"*50)
    model = LoopedMLP(3072, 1024, 10, use_spectral_norm=True, max_steps=15)
    acc, t = train_and_eval(model, train_loader, test_loader, epochs=10, steps=15)
    print(f"   Accuracy: {acc:.1f}%, Time: {t:.1f}s")
    results.append(("Wider (1024, 15 steps)", acc, t, sum(p.numel() for p in model.parameters())))
    
    # 3. Deep multi-layer
    print("\n" + "-"*50)
    print("3. DeepLoopedMLP (3 layers, hidden=512)")
    print("-"*50)
    model = DeepLoopedMLP(3072, 512, 10, num_layers=3, max_steps=15)
    acc, t = train_and_eval(model, train_loader, test_loader, epochs=10, steps=15)
    print(f"   Accuracy: {acc:.1f}%, Time: {t:.1f}s")
    results.append(("Deep (3 layers)", acc, t, sum(p.numel() for p in model.parameters())))
    
    # 4. Residual connections
    print("\n" + "-"*50)
    print("4. ResidualLoopedMLP (residual_weight=0.5)")
    print("-"*50)
    model = ResidualLoopedMLP(3072, 512, 10, max_steps=15, residual_weight=0.5)
    acc, t = train_and_eval(model, train_loader, test_loader, epochs=10, steps=15)
    print(f"   Accuracy: {acc:.1f}%, Time: {t:.1f}s")
    results.append(("Residual (0.5)", acc, t, sum(p.numel() for p in model.parameters())))
    
    # 5. GELU activation
    print("\n" + "-"*50)
    print("5. GELULoopedMLP")
    print("-"*50)
    model = GELULoopedMLP(3072, 512, 10, max_steps=15)
    acc, t = train_and_eval(model, train_loader, test_loader, epochs=10, steps=15)
    print(f"   Accuracy: {acc:.1f}%, Time: {t:.1f}s")
    results.append(("GELU activation", acc, t, sum(p.numel() for p in model.parameters())))
    
    # 6. Truncated steps (optimal from speed benchmark)
    print("\n" + "-"*50)
    print("6. Truncated LoopedMLP (hidden=512, steps=5)")
    print("-"*50)
    model = LoopedMLP(3072, 512, 10, use_spectral_norm=True, max_steps=5)
    acc, t = train_and_eval(model, train_loader, test_loader, epochs=10, steps=5)
    print(f"   Accuracy: {acc:.1f}%, Time: {t:.1f}s")
    results.append(("Truncated (5 steps)", acc, t, sum(p.numel() for p in model.parameters())))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Architecture':<25} {'Accuracy':<12} {'Time (s)':<12} {'Params':<12}")
    print("-"*70)
    for name, acc, t, params in sorted(results, key=lambda x: -x[1]):
        print(f"{name:<25} {acc:<12.1f} {t:<12.1f} {params:<12,}")
    
    best = max(results, key=lambda x: x[1])
    fastest = min(results, key=lambda x: x[2])
    
    print(f"\n🏆 Best accuracy: {best[0]} ({best[1]:.1f}%)")
    print(f"⚡ Fastest: {fastest[0]} ({fastest[2]:.1f}s)")
    
    return results


if __name__ == "__main__":
    run_architecture_experiments()
