#!/usr/bin/env python3
"""
Quick IDEA Test Framework

Uses sklearn digits (8x8) for fast iteration:
- 1797 samples, 64 features, 10 classes
- ~100x faster than MNIST
- Good for hyperparameter sweeping

Usage:
    python scripts/quick_idea_test.py --models all --epochs 5
    python scripts/quick_idea_test.py --sweep --model SpectralTorEqProp
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY, list_models
from src.training import EqPropTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Digits dataset config
INPUT_DIM = 64   # 8x8 images
OUTPUT_DIM = 10  # 10 digits


def get_digits_loaders(batch_size=32, test_size=0.2, seed=42):
    """Load sklearn digits dataset as PyTorch dataloaders."""
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader


def get_model(model_name, hidden_dim=64, use_spectral_norm=True):
    """Create model for digits dataset."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown: {model_name}. Available: {list_models()}")
    
    ModelClass = MODEL_REGISTRY[model_name]
    try:
        return ModelClass(INPUT_DIM, hidden_dim, OUTPUT_DIM, use_spectral_norm=use_spectral_norm)
    except TypeError:
        return ModelClass(INPUT_DIM, hidden_dim, OUTPUT_DIM)


def train_epoch(model, trainer, train_loader):
    """Train one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        try:
            info = trainer.step(data, target)
            total_loss += info['loss']
        except Exception:
            continue
        
        with torch.no_grad():
            output = model(data, steps=15)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return total_loss / max(len(train_loader), 1), correct / max(total, 1)


def evaluate(model, test_loader, steps=15):
    """Evaluate model."""
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
    
    return correct / max(total, 1)


def test_model(model_name, epochs=5, hidden_dim=64, lr=1e-3, beta=0.22, 
               seed=42, verbose=True):
    """Test a single model configuration."""
    torch.manual_seed(seed)
    
    train_loader, test_loader = get_digits_loaders(seed=seed)
    model = get_model(model_name, hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=20)
    
    best_acc = 0
    epoch_times = []
    
    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_epoch(model, trainer, train_loader)
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        
        test_acc = evaluate(model, test_loader)
        best_acc = max(best_acc, test_acc)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, "
                  f"Train={train_acc:.1%}, Test={test_acc:.1%}, Time={epoch_time*1000:.0f}ms")
    
    return {
        'model': model_name,
        'best_accuracy': best_acc,
        'final_accuracy': test_acc,
        'mean_epoch_time_ms': sum(epoch_times) / len(epoch_times) * 1000,
        'params': sum(p.numel() for p in model.parameters()),
        'config': {'hidden_dim': hidden_dim, 'lr': lr, 'beta': beta}
    }


def run_sweep(model_name, epochs=3, seeds=3):
    """Sweep hyperparameters for a model."""
    print(f"\n🔍 Hyperparameter sweep for {model_name}")
    print("="*60)
    
    # Parameter grid
    hidden_dims = [32, 64, 128]
    lrs = [1e-3, 3e-3]
    betas = [0.15, 0.22, 0.30]
    
    results = []
    
    for hidden_dim, lr, beta in itertools.product(hidden_dims, lrs, betas):
        accs = []
        for seed in range(42, 42 + seeds):
            result = test_model(model_name, epochs, hidden_dim, lr, beta, seed, verbose=False)
            accs.append(result['best_accuracy'])
        
        mean_acc = sum(accs) / len(accs)
        results.append({
            'hidden_dim': hidden_dim,
            'lr': lr,
            'beta': beta,
            'mean_accuracy': mean_acc,
            'std': (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
        })
        
        print(f"  h={hidden_dim:3d}, lr={lr:.0e}, β={beta:.2f} → {mean_acc:.1%} ± {results[-1]['std']:.1%}")
    
    # Best config
    best = max(results, key=lambda x: x['mean_accuracy'])
    print(f"\n  🏆 Best: h={best['hidden_dim']}, lr={best['lr']:.0e}, β={best['beta']:.2f} → {best['mean_accuracy']:.1%}")
    
    return results


def run_comparison(models, epochs=5, hidden_dim=64, seeds=1):
    """Compare multiple models."""
    print(f"\n🧪 Quick IDEA Comparison (digits 8x8)")
    print(f"   Models: {len(models)}, Epochs: {epochs}, Device: {DEVICE}")
    print("="*60)
    
    results = {}
    
    for model_name in models:
        print(f"\n📊 {model_name}")
        
        accs = []
        times = []
        for seed in range(42, 42 + seeds):
            r = test_model(model_name, epochs, hidden_dim, seed=seed)
            accs.append(r['best_accuracy'])
            times.append(r['mean_epoch_time_ms'])
        
        results[model_name] = {
            'mean_accuracy': sum(accs) / len(accs),
            'mean_time_ms': sum(times) / len(times),
            'params': r['params']
        }
    
    # Print rankings
    print("\n" + "="*70)
    print("RANKINGS")
    print("="*70)
    
    sorted_models = sorted(results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
    baseline_acc = results.get('ModernEqProp', sorted_models[-1][1])['mean_accuracy']
    
    for rank, (name, data) in enumerate(sorted_models, 1):
        delta = (data['mean_accuracy'] - baseline_acc) * 100
        delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal}{rank}. {name:<20} {data['mean_accuracy']:.1%} ({delta_str:>6}) "
              f"{data['mean_time_ms']:>5.0f}ms {data['params']:>8,} params")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Quick IDEA testing on digits')
    parser.add_argument('--models', type=str, default='all', help='Models to test')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--seeds', type=int, default=1, help='Number of seeds')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--model', type=str, default='SpectralTorEqProp', help='Model for sweep')
    parser.add_argument('--output', type=str, default='results/quick_ideas.json')
    args = parser.parse_args()
    
    if args.sweep:
        results = run_sweep(args.model, args.epochs, args.seeds)
    else:
        if args.models == 'all':
            models = ['ModernEqProp', 'SpectralTorEqProp', 'DiffTorEqProp', 'TPEqProp',
                      'TorEqODEProp', 'TCEP', 'MSTEP', 'TEPSSR', 'HTSEP']
        elif args.models == 'top':
            models = ['ModernEqProp', 'SpectralTorEqProp', 'TPEqProp']
        else:
            models = [m.strip() for m in args.models.split(',')]
        
        results = run_comparison(models, args.epochs, args.hidden_dim, args.seeds)
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved to {args.output}")


if __name__ == '__main__':
    main()
