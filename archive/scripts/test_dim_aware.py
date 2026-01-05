#!/usr/bin/env python3
"""
Test Dimension-Aware Models

Evaluate algorithmic interventions for handling high-dimensional inputs.
"""

import argparse
import json
from pathlib import Path
import torch
import torch.optim as optim
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY
from src.training import EqPropTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_mnist_data(batch_size=64, limit_batches=None):
    """Get MNIST data."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    if limit_batches:
        # Convert to list for limiting
        train_list = []
        for i, batch in enumerate(train_loader):
            if i >= limit_batches:
                break
            train_list.append(batch)
        
        test_list = []
        for i, batch in enumerate(test_loader):
            if i >= limit_batches // 2:
                break
            test_list.append(batch)
        
        return train_list, test_list
    
    return train_loader, test_loader


def test_model(model_name, epochs=3, hidden_dim=128, target_dim=128, limit_batches=100):
    """Test a dimension-aware model on MNIST."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    torch.manual_seed(42)
    
    # Get data
    train_data, test_data = get_mnist_data(limit_batches=limit_batches)
    
    # Create model with appropriate args
    if model_name in ['EmbeddingEqProp', 'ProjectedEqProp']:
        model = MODEL_REGISTRY[model_name](
            input_dim=784,
            hidden_dim=hidden_dim,
            output_dim=10,
            target_dim=target_dim,
            use_spectral_norm=True
        ).to(DEVICE)
    elif model_name == 'DimensionScaledEqProp':
        model = MODEL_REGISTRY[model_name](
            input_dim=784,
            hidden_dim=hidden_dim,
            output_dim=10,
            reference_dim=64,  # Digits dimension
            use_spectral_norm=True
        ).to(DEVICE)
    else:
        model = MODEL_REGISTRY[model_name](
            input_dim=784,
            hidden_dim=hidden_dim,
            output_dim=10,
            use_spectral_norm=True
        ).to(DEVICE)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = EqPropTrainer(model, optimizer, beta=0.22, max_steps=30)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_correct = train_total = 0
        
        for data, target in train_data:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            try:
                trainer.step(data, target)
            except Exception as e:
                print(f"  Training error: {e}")
                continue
            
            with torch.no_grad():
                output = model(data, steps=25)
                pred = output.argmax(dim=1)
                train_correct += (pred == target).sum().item()
                train_total += target.size(0)
        
        train_acc = train_correct / max(train_total, 1)
        print(f"  Epoch {epoch+1}/{epochs}: Train={train_acc:.1%}")
    
    train_time = time.time() - start_time
    
    # Evaluate
    model.eval()
    test_correct = test_total = 0
    
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(DEVICE), target.to(DEVICE)
            try:
                output = model(data, steps=25)
                pred = output.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)
            except Exception:
                continue
    
    test_acc = test_correct / max(test_total, 1)
    
    print(f"\n  ✅ Test Accuracy: {test_acc:.1%}, Time: {train_time:.1f}s")
    
    return {
        'model': model_name,
        'test_acc': test_acc,
        'params': param_count,
        'time_s': train_time
    }


def compare_dim_aware_models(epochs=3, limit_batches=100, target_dim=128):
    """Compare all dimension-aware models."""
    
    print(f"\n🧪 Dimension-Aware Model Comparison")
    print(f"   Target dim: {target_dim}, Epochs: {epochs}, Device: {DEVICE}")
    
    models = [
        'TPEqProp',  # Baseline
        'DimensionScaledEqProp',
        'EmbeddingEqProp',
        'ProjectedEqProp',
    ]
    
    results = []
    
    for model_name in models:
        try:
            result = test_model(model_name, epochs, target_dim=target_dim, 
                               limit_batches=limit_batches)
            results.append(result)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':>10} {'vs TPEqProp':>12} {'Params':>12}")
    print("-"*70)
    
    baseline_acc = results[0]['test_acc'] if results else 0
    
    for r in results:
        delta = r['test_acc'] - baseline_acc
        delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        print(f"{r['model']:<25} {r['test_acc']:>9.1%} {delta_str:>12} {r['params']:>11,}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--target-dim', type=int, default=128)
    parser.add_argument('--limit-batches', type=int, default=100)
    parser.add_argument('--output', type=str, default='results/dim_aware_comparison.json')
    args = parser.parse_args()
    
    if args.compare or args.model is None:
        results = compare_dim_aware_models(args.epochs, args.limit_batches, args.target_dim)
    else:
        result = test_model(args.model, args.epochs, target_dim=args.target_dim,
                           limit_batches=args.limit_batches)
        results = [result]
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved to {args.output}")


if __name__ == '__main__':
    main()
