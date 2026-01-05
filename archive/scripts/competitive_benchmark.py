#!/usr/bin/env python3
"""Run competitive benchmark: EqProp models vs Backprop baseline.

Uses insights from analysis:
- Spectral norm enabled for stability
- Optimal beta = 0.22
- max_steps = 25
- Train for sufficient epochs to show convergence
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim
import time
import json
import argparse
import numpy as np

from src.models import LoopedMLP, ToroidalMLP, ModernEqProp, BackpropMLP
from src.models.conv_eqprop import ConvEqProp
from src.training import EqPropTrainer
from src.tasks import get_task_loader


def train_eqprop(model, train_loader, test_loader, epochs=50, lr=0.001, beta=0.22, max_steps=25):
    """Train EqProp model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=max_steps)
    
    history = {'train_loss': [], 'test_acc': [], 'convergence_rate': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_converged = 0
        total_batches = 0
        
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            metrics = trainer.step(x, y)
            total_loss += metrics['loss']
            total_converged += int(metrics.get('converged_free', False))
            total_batches += 1
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x, steps=max_steps)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100. * correct / total
        conv_rate = total_converged / total_batches
        
        history['train_loss'].append(total_loss / total_batches)
        history['test_acc'].append(acc)
        history['convergence_rate'].append(conv_rate)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={total_loss/total_batches:.4f}, "
                  f"Acc={acc:.2f}%, Conv={conv_rate:.1%}")
    
    elapsed = time.time() - start_time
    return history, elapsed


def train_backprop(model, train_loader, test_loader, epochs=50, lr=0.001):
    """Train Backprop baseline."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'test_acc': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100. * correct / total
        history['train_loss'].append(total_loss / total_batches)
        history['test_acc'].append(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={total_loss/total_batches:.4f}, Acc={acc:.2f}%")
    
    elapsed = time.time() - start_time
    return history, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=1, help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default='digits', help='Dataset name')
    parser.add_argument('--models', type=str, default='ModernEqProp,LoopedMLP',
                       help='Comma-separated list of models to test (or "all" for all models)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"COMPETITIVE BENCHMARK: EqProp vs Backprop ({args.seeds} seeds)")
    print("=" * 70)
    
    # Configuration
    epochs = args.epochs
    hidden_dim = 256
    lr = 0.001
    beta = 0.22  # Optimal from analysis
    max_steps = 25  # Most models converge by step 25
    
    # Load data
    train_loader, test_loader, input_dim, output_dim = get_task_loader(
        args.dataset, batch_size=64, dataset_size=10000
    )
    
    results = {}
    
    # Test models
    if args.dataset == 'cifar10':
        print(">> Mode: CIFAR-10 (Using ConvEqProp)")
        all_configs = [
             ("ConvEqProp (SN)", lambda: ConvEqProp(input_channels=3, hidden_channels=64, output_dim=10, use_spectral_norm=True)),
        ]
    else:
        all_configs = [
            ("BackpropMLP", lambda: BackpropMLP(input_dim, hidden_dim, output_dim, depth=2)),
            ("LoopedMLP (SN)", lambda: LoopedMLP(input_dim, hidden_dim, output_dim, 
                                                  symmetric=True, use_spectral_norm=True)),
            ("ToroidalMLP (SN)", lambda: ToroidalMLP(input_dim, hidden_dim, output_dim, 
                                                      use_spectral_norm=True)),
            ("ModernEqProp (SN)", lambda: ModernEqProp(input_dim, hidden_dim, output_dim,
                                                        use_spectral_norm=True)),
        ]
    
    # Filter models based on --models argument
    if args.models.lower() != 'all':
        requested_models = [m.strip() for m in args.models.split(',')]
        # Always include BackpropMLP as baseline
        if 'BackpropMLP' not in requested_models and args.dataset != 'cifar10':
            requested_models.insert(0, 'BackpropMLP')
        
        filtered_configs = []
        for name, model_fn in all_configs:
            # Check if this model matches any requested model
            model_base_name = name.replace(' (SN)', '').replace(' ', '')
            if any(req.replace(' (SN)', '').replace(' ', '') in model_base_name for req in requested_models):
                filtered_configs.append((name, model_fn))
        
        configs = filtered_configs
        print(f"\n>> Testing models: {[name for name, _ in configs]}")
    else:
        configs = all_configs
        print(f"\n>> Testing all models")
    
    print()
    
    for name, model_fn in configs:
        print(f"\n## {name}")
        print("-" * 50)
        
        seed_accuracies = []
        seed_times = []
        all_histories = []
        
        for seed in range(42, 42 + args.seeds):
            print(f"  Seed {seed}...")
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            model = model_fn().cuda()
            
            if "Backprop" in name:
                history, elapsed = train_backprop(model, train_loader, test_loader, 
                                                 epochs=epochs, lr=lr)
            else:
                history, elapsed = train_eqprop(model, train_loader, test_loader, 
                                               epochs=epochs, lr=lr, beta=beta, max_steps=max_steps)
            
            final_acc = history['test_acc'][-1]
            seed_accuracies.append(final_acc)
            seed_times.append(elapsed)
            all_histories.append(history)
            
            print(f"    Acc: {final_acc:.2f}%, Time: {elapsed:.1f}s")

        # Aggregate stats
        mean_acc = np.mean(seed_accuracies)
        std_acc = np.std(seed_accuracies)
        mean_time = np.mean(seed_times)
        
        results[name] = {
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_time': mean_time,
            'seeds': seed_accuracies,
            'histories': all_histories
        }
        
        print(f"  Result: {mean_acc:.2f} ± {std_acc:.2f}% (Mean Time: {mean_time:.1f}s)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} | {'Mean Acc':>15} | {'Mean Time':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<20} | {r['mean_acc']:>6.2f} ± {r['std_acc']:<5.2f}% | {r['mean_time']:>8.1f}s")
    
    # Save results
    output_file = f'/tmp/competitive_benchmark_{args.seeds}seed.json'
    
    # helper for serialization
    def make_serializable(obj):
        if hasattr(obj, 'item'): return obj.item()
        if hasattr(obj, '__iter__') and not isinstance(obj, str): return [make_serializable(x) for x in obj]
        if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
        return obj

    with open(output_file, 'w') as f:
        # Convert history lists to serializable format
        # Complex nesting: list of dicts of lists
        save_results = {}
        for k, v in results.items():
             save_results[k] = {
                 'mean_acc': float(v['mean_acc']),
                 'std_acc': float(v['std_acc']),
                 'mean_time': float(v['mean_time']),
                 'seeds': list(v['seeds']),
                 'histories': make_serializable(v['histories'])
             }
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (will be slow)")
    main()
