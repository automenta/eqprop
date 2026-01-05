#!/usr/bin/env python3
"""
Benchmark Suite Runner

Run comprehensive benchmarks across tasks and models.
Includes baseline comparisons and statistical analysis.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import get_task, TASK_REGISTRY, TaskResult
from src.models import MODEL_REGISTRY, get_model
from src.training import EqPropTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleMLPBaseline(nn.Module):
    """Simple MLP baseline with backprop."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, **kwargs):
        return self.net(x)


def train_eqprop_model(model, train_loader, val_loader, config, verbose=False):
    """Train an EqProp model."""
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    trainer = EqPropTrainer(model, optimizer, beta=config.beta, max_steps=30)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = train_correct = train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            try:
                info = trainer.step(data, target)
                train_loss += info['loss']
            except Exception:
                continue
            
            with torch.no_grad():
                output = model(data, steps=25)  # More steps for MNIST
                pred = output.argmax(dim=1)
                train_correct += (pred == target).sum().item()
                train_total += target.size(0)
        
        train_acc = train_correct / max(train_total, 1)
        
        # Validate
        model.eval()
        val_correct = val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data, steps=25)
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        val_acc = val_correct / max(val_total, 1)
        best_val_acc = max(best_val_acc, val_acc)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{config.epochs}: Train={train_acc:.1%}, Val={val_acc:.1%}")
    
    return best_val_acc


def train_baseline_model(model, train_loader, val_loader, config, verbose=False):
    """Train a baseline model with standard backprop."""
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = train_correct = train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
        
        train_acc = train_correct / max(train_total, 1)
        
        # Validate
        model.eval()
        val_correct = val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        val_acc = val_correct / max(val_total, 1)
        best_val_acc = max(best_val_acc, val_acc)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{config.epochs}: Train={train_acc:.1%}, Val={val_acc:.1%}")
    
    return best_val_acc


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            try:
                output = model(data, steps=25)
            except TypeError:
                output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / max(total, 1)


def run_benchmark(task_name, model_name, seeds=3, verbose=True):
    """Run benchmark for a task-model pair."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}, Model: {model_name}")
        print(f"{'='*60}")
    
    results = []
    
    for seed_idx, seed in enumerate(range(42, 42 + seeds)):
        if verbose:
            print(f"\n  Seed {seed_idx+1}/{seeds} (seed={seed})")
        
        torch.manual_seed(seed)
        
        # Get task
        task = get_task(task_name, seed=seed)
        config = task.get_config()
        train_loader, val_loader, test_loader = task.get_loaders()
        
        # Create model
        if model_name == 'MLP':
            model = SimpleMLPBaseline(config.input_dim, config.hidden_dim, config.output_dim).to(DEVICE)
            is_baseline = True
        else:
            model = get_model(model_name, input_dim=config.input_dim, 
                            hidden_dim=config.hidden_dim, output_dim=config.output_dim).to(DEVICE)
            is_baseline = False
        
        # Train
        start_time = time.time()
        
        if is_baseline:
            val_acc = train_baseline_model(model, train_loader, val_loader, config, verbose)
        else:
            val_acc = train_eqprop_model(model, train_loader, val_loader, config, verbose)
        
        train_time = time.time() - start_time
        
        # Test
        test_acc = evaluate_model(model, test_loader)
        
        if verbose:
            print(f"    Val={val_acc:.1%}, Test={test_acc:.1%}, Time={train_time:.1f}s")
        
        results.append({
            'seed': seed,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_time_s': train_time,
            'params': sum(p.numel() for p in model.parameters())
        })
    
    # Aggregate
    test_accs = [r['test_acc'] for r in results]
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    mean_time = np.mean([r['train_time_s'] for r in results])
    params = results[0]['params']
    
    if verbose:
        print(f"\n  📊 Summary: Test={mean_acc:.1%} ± {std_acc:.1%}, Time={mean_time:.1f}s, Params={params:,}")
    
    return {
        'task': task_name,
        'model': model_name,
        'mean_test_acc': mean_acc,
        'std_test_acc': std_acc,
        'mean_time_s': mean_time,
        'params': params,
        'seeds': results
    }


def run_suite(tasks: List[str], models: List[str], seeds=3, output_file='results/benchmark_suite.json'):
    """Run full benchmark suite."""
    
    print(f"\n🧪 Benchmark Suite")
    print(f"   Tasks: {tasks}")
    print(f"   Models: {models}")
    print(f"   Seeds: {seeds}")
    print(f"   Device: {DEVICE}")
    
    all_results = []
    
    for task in tasks:
        for model in models:
            result = run_benchmark(task, model, seeds, verbose=True)
            all_results.append(result)
    
    # Print summary table
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)
    print(f"{'Task':<15} {'Model':<20} {'Accuracy':>12} {'Time (s)':>10} {'Params':>12}")
    print("-"*90)
    
    for r in all_results:
        print(f"{r['task']:<15} {r['model']:<20} {r['mean_test_acc']:>11.1%} {r['mean_time_s']:>9.1f} {r['params']:>11,}")
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run benchmark suite')
    parser.add_argument('--tasks', type=str, default='digits,mnist',
                       help='Tasks to run (comma-separated or "all")')
    parser.add_argument('--models', type=str, default='MLP,TPEqProp,SpectralTorEqProp,MSTEP',
                       help='Models to test (comma-separated)')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--output', type=str, default='results/benchmark_suite.json')
    args = parser.parse_args()
    
    tasks = list(TASK_REGISTRY.keys()) if args.tasks == 'all' else args.tasks.split(',')
    models = args.models.split(',')
    
    run_suite(tasks, models, args.seeds, args.output)


if __name__ == '__main__':
    main()
