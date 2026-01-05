#!/usr/bin/env python3
"""
Multi-Dataset Benchmark Suite for TorEqProp

Tests EqProp models across multiple datasets and tasks:
- Vision: MNIST, Fashion-MNIST (intermediate complexity)
- RL: CartPole, Acrobot (behavioral cloning)
- Hierarchical models: MSTEP, EnhancedMSTEP variants

Designed to run quickly (<2 hours total) while providing
comprehensive validation across task types.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

# Task loaders
from src.tasks import get_task_loader

# Models
from src.models.looped_mlp import LoopedMLP
from src.models.modern_eqprop import ModernEqProp
from src.models.mstep import MSTEP
from src.models.mstep_enhanced import EnhancedMSTEP
from src.models.bp_mlp import BackpropMLP
from src.training import EqPropTrainer


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    task_name: str
    input_dim: int
    output_dim: int
    hidden_dim: int
    epochs: int
    batch_size: int
    dataset_size: Optional[int]  # None = full dataset
    max_steps: int = 20
    lr: float = 0.001
    beta: float = 0.22


# Define benchmark configurations - optimized for speed and accuracy (updated with hyperparameter sweep results)
VISION_BENCHMARKS = [
    BenchmarkConfig("Digits (8x8)", "digits", 64, 10, 128, 30, 64, None, max_steps=30, lr=0.001, beta=0.22),
    BenchmarkConfig("MNIST", "mnist", 784, 10, 256, 20, 128, 5000, max_steps=30, lr=0.002, beta=0.22),
    BenchmarkConfig("Fashion-MNIST", "fashion-mnist", 784, 10, 256, 20, 128, 5000, max_steps=30, lr=0.002, beta=0.5),
]

RL_BENCHMARKS = [
    BenchmarkConfig("CartPole-BC", "cartpole", 4, 2, 64, 30, 64, 5000, max_steps=30, lr=0.001, beta=0.22),
    BenchmarkConfig("Acrobot-BC", "acrobot", 6, 3, 64, 30, 64, 5000, max_steps=30, lr=0.002, beta=0.5),
]


def get_model_configs(input_dim: int, hidden_dim: int, output_dim: int) -> Dict[str, Callable]:
    """Get model factory functions for benchmarking."""
    return {
        # Core models (proven on MNIST)
        "BackpropMLP": lambda: BackpropMLP(input_dim, hidden_dim, output_dim, depth=2),
        "LoopedMLP (SN)": lambda: LoopedMLP(input_dim, hidden_dim, output_dim, 
                                            symmetric=True, use_spectral_norm=True),
        "ModernEqProp (SN)": lambda: ModernEqProp(input_dim, hidden_dim, output_dim,
                                                   use_spectral_norm=True),
        # Hierarchical models
        "MSTEP (SN)": lambda: MSTEP(input_dim, hidden_dim, output_dim, 
                                    use_spectral_norm=True),
        "EnhancedMSTEP (SN)": lambda: EnhancedMSTEP(input_dim, hidden_dim, output_dim,
                                                     use_spectral_norm=True),
    }


def train_backprop(model, train_loader, epochs, lr):
    """Train a backprop model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()


def train_eqprop(model, train_loader, epochs, lr, beta, max_steps):
    """Train an EqProp model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=max_steps)
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            trainer.step(x, y)


def evaluate(model, test_loader) -> float:
    """Evaluate model accuracy."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Handle both regular forward and EqProp models
            if hasattr(model, 'forward_equilibrium'):
                out = model.forward_equilibrium(x, steps=25)
            else:
                out = model(x)
            
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return 100.0 * correct / total


def run_benchmark(config: BenchmarkConfig, model_filter: Optional[List[str]] = None, 
                  seeds: int = 1) -> Dict:
    """Run benchmark for a single configuration."""
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {config.name}")
    print(f"{'='*70}")
    print(f"Task: {config.task_name}, Input: {config.input_dim}, Output: {config.output_dim}")
    print(f"Epochs: {config.epochs}, LR: {config.lr}, Beta: {config.beta}")
    print()
    
    # Load data
    try:
        train_loader, test_loader, in_dim, out_dim = get_task_loader(
            config.task_name, 
            batch_size=config.batch_size,
            dataset_size=config.dataset_size or 10000
        )
        # Update config with actual dimensions
        config.input_dim = in_dim
        config.output_dim = out_dim
        print(f"Data loaded: {in_dim} -> {out_dim}")
    except Exception as e:
        print(f"❌ Failed to load {config.task_name}: {e}")
        return {"error": str(e)}
    
    # Get model configs
    model_configs = get_model_configs(config.input_dim, config.hidden_dim, config.output_dim)
    
    # Filter models if specified
    if model_filter:
        model_configs = {k: v for k, v in model_configs.items() 
                        if any(f.lower() in k.lower() for f in model_filter)}
    
    results = {}
    
    for model_name, model_fn in model_configs.items():
        print(f"\n  {model_name}")
        print(f"  {'-'*50}")
        
        seed_accs = []
        seed_times = []
        
        for seed in range(42, 42 + seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            try:
                model = model_fn()
                start = time.time()
                
                if 'Backprop' in model_name:
                    train_backprop(model, train_loader, config.epochs, config.lr)
                else:
                    train_eqprop(model, train_loader, config.epochs, 
                                config.lr, config.beta, config.max_steps)
                
                elapsed = time.time() - start
                acc = evaluate(model, test_loader)
                
                seed_accs.append(acc)
                seed_times.append(elapsed)
                print(f"    Seed {seed}: {acc:.2f}% ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"    Seed {seed}: ❌ Error: {e}")
                seed_accs.append(0.0)
                seed_times.append(0.0)
        
        if seed_accs:
            mean_acc = np.mean(seed_accs)
            std_acc = np.std(seed_accs) if len(seed_accs) > 1 else 0.0
            mean_time = np.mean(seed_times)
            
            results[model_name] = {
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'seeds': seed_accs,
                'mean_time': mean_time
            }
            
            print(f"  → Result: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Benchmark Suite")
    parser.add_argument('--seeds', type=int, default=2, help='Seeds per config')
    parser.add_argument('--vision-only', action='store_true', help='Only run vision tasks')
    parser.add_argument('--rl-only', action='store_true', help='Only run RL tasks')
    parser.add_argument('--models', type=str, default=None,
                       help='Comma-separated model filter (e.g., "Looped,Modern,MSTEP")')
    parser.add_argument('--quick', action='store_true', help='Quick mode: fewer epochs')
    parser.add_argument('--smoke-test', action='store_true', help='Sanity check: 1 epoch, 1 seed, all tasks')
    args = parser.parse_args()
    
    if args.smoke_test:
        print("⚡ SMOKE TEST MODE: 1 epoch, 1 seed, small subsets")
        args.seeds = 1
        args.quick = True
    
    print("=" * 80)
    print("TOREQPROP MULTI-DATASET BENCHMARK SUITE")
    print("=" * 80)
    print(f"Seeds: {args.seeds}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()
    
    model_filter = args.models.split(',') if args.models else None
    
    all_results = {}
    
    # Vision benchmarks
    if not args.rl_only:
        print("\n" + "=" * 80)
        print("VISION TASKS")
        print("=" * 80)
        
        for config in VISION_BENCHMARKS:
            if args.smoke_test:
                config.epochs = 1
                config.dataset_size = 500  # Tiny subset for smoke test
            elif args.quick:
                config.epochs = min(config.epochs, 10)
                config.dataset_size = min(config.dataset_size or 5000, 2000)
            
            results = run_benchmark(config, model_filter, args.seeds)
            all_results[config.name] = results
    
    # RL benchmarks
    if not args.vision_only:
        print("\n" + "=" * 80)
        print("REINFORCEMENT LEARNING TASKS (Behavioral Cloning)")
        print("=" * 80)
        
        for config in RL_BENCHMARKS:
            if args.smoke_test:
                config.epochs = 1
            elif args.quick:
                config.epochs = min(config.epochs, 10)
            
            results = run_benchmark(config, model_filter, args.seeds)
            all_results[config.name] = results
    
    # Summary
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS MATRIX")
    print("=" * 100)
    print()
    
    # Collect all model names
    all_models = set()
    for task_results in all_results.values():
        if isinstance(task_results, dict):
            all_models.update(task_results.keys())
    sorted_models = sorted(list(all_models))
    
    # Print Header
    header = f"{'Task':<20}"
    for m in sorted_models:
        header += f" | {m[:15]:<15}"
    print(header)
    print("-" * len(header))
    
    # Print Rows
    for task_name in all_results:
        row = f"{task_name:<20}"
        task_data = all_results[task_name]
        
        if 'error' in task_data:
            print(f"{task_name:<20} | ERROR: {task_data['error']}")
            continue
            
        for m in sorted_models:
            if m in task_data:
                res = task_data[m]
                acc = res['mean_acc']
                std = res['std_acc']
                row += f" | {acc:>6.2f}%"
            else:
                row += f" | {'-':>15}"
        print(row)
    
    # Save results
    output_path = Path('results/multi_dataset_benchmark.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print()
    print(f"Results saved to: {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
