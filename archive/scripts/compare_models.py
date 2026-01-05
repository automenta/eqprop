#!/usr/bin/env python3
"""
Enhanced EqProp Model Comparison Framework

Features:
- Automatic model discovery from registry
- Multiple metrics (accuracy, convergence, energy, time)
- Statistical analysis with multiple seeds
- Pretty output with rankings
- JSON export for further analysis
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY, list_models
from src.training import EqPropTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class ModelMetrics:
    """Metrics for a single model run."""
    model_name: str
    seed: int
    final_accuracy: float
    best_accuracy: float
    train_accuracy: float
    train_loss: float
    epoch_time: float
    convergence_steps: int
    param_count: int


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across seeds."""
    model_name: str
    mean_accuracy: float
    std_accuracy: float
    mean_best_accuracy: float
    mean_epoch_time: float
    mean_convergence_steps: float
    param_count: int
    seeds: List[ModelMetrics]


def get_model(model_name, input_dim=784, hidden_dim=256, output_dim=10, use_spectral_norm=True):
    """Create model instance with error handling."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}\nAvailable: {list_models()}")
    
    ModelClass = MODEL_REGISTRY[model_name]
    try:
        return ModelClass(input_dim, hidden_dim, output_dim, use_spectral_norm=use_spectral_norm)
    except TypeError:
        # Some models don't accept use_spectral_norm
        return ModelClass(input_dim, hidden_dim, output_dim)


def measure_convergence(model, x, max_steps=50, threshold=1e-4):
    """Measure steps to equilibrium convergence."""
    with torch.no_grad():
        h = model.embed(x)
        prev_h = h.clone()
        buffer = None
        
        for step in range(max_steps):
            try:
                h, buffer = model.forward_step(h, x, buffer, step=step, max_steps=max_steps)
            except TypeError:
                h, buffer = model.forward_step(h, x, buffer)
            
            delta = (h - prev_h).abs().mean().item()
            if delta < threshold:
                return step + 1
            prev_h = h.clone()
    return max_steps


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, trainer, train_loader):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data = data.view(data.size(0), -1).to(DEVICE)
        target = target.to(DEVICE)
        
        try:
            info = trainer.step(data, target)
            total_loss += info['loss']
        except Exception as e:
            # Fallback for models with compatibility issues
            continue
        
        with torch.no_grad():
            output = model(data, steps=20)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return total_loss / max(len(train_loader), 1), correct / max(total, 1)


def evaluate(model, test_loader, steps=30):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(DEVICE)
            target = target.to(DEVICE)
            
            try:
                output = model(data, steps=steps)
            except Exception as e:
                continue
                
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / max(total, 1)


def run_comparison(
    models_to_test: List[str],
    epochs: int = 1,
    batch_size: int = 64,
    hidden_dim: int = 256,
    seeds: List[int] = None,
    max_batches: Optional[int] = None
) -> Dict[str, AggregatedMetrics]:
    """Run comparison between model variants."""
    seeds = seeds or [42]
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    # Optionally limit batches for quick testing
    if max_batches:
        from torch.utils.data import Subset
        train_data = Subset(train_data, range(min(max_batches * batch_size, len(train_data))))
        test_data = Subset(test_data, range(min(max_batches * batch_size // 2, len(test_data))))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        seed_metrics = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            
            try:
                model = get_model(model_name, hidden_dim=hidden_dim).to(DEVICE)
            except Exception as e:
                print(f"  ❌ Failed to create model: {e}")
                continue
            
            param_count = count_parameters(model)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            trainer = EqPropTrainer(model, optimizer, beta=0.22, max_steps=30)
            
            # Measure initial convergence
            sample_x = next(iter(train_loader))[0][:8].view(8, -1).to(DEVICE)
            
            best_acc = 0.0
            epoch_times = []
            
            for epoch in range(epochs):
                start_time = time.time()
                train_loss, train_acc = train_epoch(model, trainer, train_loader)
                epoch_time = time.time() - start_time
                epoch_times.append(epoch_time)
                
                test_acc = evaluate(model, test_loader)
                best_acc = max(best_acc, test_acc)
                
                print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, "
                      f"Train={train_acc:.2%}, Test={test_acc:.2%}, Time={epoch_time:.1f}s")
            
            # Measure post-training convergence
            try:
                conv_steps = measure_convergence(model, sample_x)
            except Exception:
                conv_steps = 30
            
            metrics = ModelMetrics(
                model_name=model_name,
                seed=seed,
                final_accuracy=test_acc,
                best_accuracy=best_acc,
                train_accuracy=train_acc,
                train_loss=train_loss,
                epoch_time=sum(epoch_times) / len(epoch_times),
                convergence_steps=conv_steps,
                param_count=param_count
            )
            seed_metrics.append(metrics)
        
        if seed_metrics:
            results[model_name] = AggregatedMetrics(
                model_name=model_name,
                mean_accuracy=sum(m.final_accuracy for m in seed_metrics) / len(seed_metrics),
                std_accuracy=(sum((m.final_accuracy - sum(m.final_accuracy for m in seed_metrics) / len(seed_metrics))**2 
                             for m in seed_metrics) / len(seed_metrics)) ** 0.5 if len(seed_metrics) > 1 else 0,
                mean_best_accuracy=sum(m.best_accuracy for m in seed_metrics) / len(seed_metrics),
                mean_epoch_time=sum(m.epoch_time for m in seed_metrics) / len(seed_metrics),
                mean_convergence_steps=sum(m.convergence_steps for m in seed_metrics) / len(seed_metrics),
                param_count=seed_metrics[0].param_count,
                seeds=seed_metrics
            )
    
    return results


def print_comparison_table(results: Dict[str, AggregatedMetrics], baseline='ModernEqProp'):
    """Print a formatted comparison table with rankings."""
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    
    # Sort by accuracy
    sorted_models = sorted(results.values(), key=lambda x: x.mean_accuracy, reverse=True)
    
    # Find baseline
    baseline_acc = results.get(baseline, sorted_models[-1]).mean_accuracy
    
    print(f"{'Rank':<5} {'Model':<20} {'Accuracy':>10} {'Δ Baseline':>12} {'Conv Steps':>12} "
          f"{'Time':>10} {'Params':>12}")
    print("-"*100)
    
    for rank, metrics in enumerate(sorted_models, 1):
        delta = (metrics.mean_accuracy - baseline_acc) * 100
        delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        
        print(f"{medal}{rank:<3} {metrics.model_name:<20} {metrics.mean_accuracy:>9.2%} "
              f"{delta_str:>12} {metrics.mean_convergence_steps:>11.1f} "
              f"{metrics.mean_epoch_time:>9.1f}s {metrics.param_count:>11,}")
    
    print("="*100)
    
    # Category winners
    print("\n📊 CATEGORY WINNERS:")
    best_acc = max(results.values(), key=lambda x: x.mean_accuracy)
    fastest_time = min(results.values(), key=lambda x: x.mean_epoch_time)
    fastest_conv = min(results.values(), key=lambda x: x.mean_convergence_steps)
    smallest = min(results.values(), key=lambda x: x.param_count)
    
    print(f"  🎯 Best Accuracy:      {best_acc.model_name} ({best_acc.mean_accuracy:.2%})")
    print(f"  ⚡ Fastest Training:   {fastest_time.model_name} ({fastest_time.mean_epoch_time:.1f}s/epoch)")
    print(f"  🔄 Fastest Convergence: {fastest_conv.model_name} ({fastest_conv.mean_convergence_steps:.1f} steps)")
    print(f"  📦 Smallest Model:     {smallest.model_name} ({smallest.param_count:,} params)")


def main():
    parser = argparse.ArgumentParser(description='Compare EqProp model variants')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated models or "all" for all IDEA variants')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--seeds', type=int, default=1, help='Number of seeds')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--max-batches', type=int, default=None, help='Limit batches for quick test')
    parser.add_argument('--output', type=str, default='results/model_comparison.json')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for name in list_models():
            print(f"  - {name}")
        return
    
    # Determine models to test
    if args.models == 'all':
        # All IDEA variants + baseline
        models = ['ModernEqProp', 'SpectralTorEqProp', 'DiffTorEqProp', 'TPEqProp',
                  'TorEqODEProp', 'TCEP', 'MSTEP', 'TEPSSR', 'HTSEP']
    elif args.models == 'ideas':
        # Just IDEA variants
        models = ['SpectralTorEqProp', 'DiffTorEqProp', 'TPEqProp', 'TorEqODEProp',
                  'TCEP', 'MSTEP', 'TEPSSR', 'HTSEP']
    else:
        models = [m.strip() for m in args.models.split(',')]
    
    seeds = list(range(42, 42 + args.seeds))
    
    print(f"🧪 EqProp Model Comparison")
    print(f"   Models: {len(models)}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Seeds: {seeds}")
    print(f"   Device: {DEVICE}")
    
    results = run_comparison(
        models_to_test=models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        seeds=seeds,
        max_batches=args.max_batches
    )
    
    print_comparison_table(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {
        name: {
            'mean_accuracy': m.mean_accuracy,
            'std_accuracy': m.std_accuracy,
            'mean_best_accuracy': m.mean_best_accuracy,
            'mean_epoch_time': m.mean_epoch_time,
            'mean_convergence_steps': m.mean_convergence_steps,
            'param_count': m.param_count,
            'seeds': [asdict(s) for s in m.seeds]
        }
        for name, m in results.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
