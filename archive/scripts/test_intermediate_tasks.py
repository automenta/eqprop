#!/usr/bin/env python3
"""
Test Intermediate Tasks

Evaluate models on intermediate-scale tasks to understand dimension scaling.
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
from src.intermediate_tasks import get_intermediate_task
from src.training import EqPropTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_and_eval(model_name, task_name, epochs=10, verbose=True):
    """Train and evaluate on intermediate task."""
    
    print(f"\n{'='*70}")
    print(f"Task: {task_name}, Model: {model_name}")
    print(f"{'='*70}")
    
    # Get task
    task = get_intermediate_task(task_name)
    config = task.get_config()
    train_loader, val_loader, test_loader = task.get_loaders()
    
    print(f"Input dim: {config.input_dim}, Output: {config.output_dim} classes")
    print(f"Train samples: {len(train_loader.dataset)}")
    
    # Create model
    torch.manual_seed(42)
    model = MODEL_REGISTRY[model_name](
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        use_spectral_norm=True
    ).to(DEVICE)
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    trainer = EqPropTrainer(model, optimizer, beta=config.beta, max_steps=30)
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(epochs):
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
                output = model(data, steps=25)
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
            print(f"Epoch {epoch+1}/{epochs}: Train={train_acc:.1%}, Val={val_acc:.1%}")
    
    total_time = time.time() - start_time
    
    # Test
    model.eval()
    test_correct = test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data, steps=25)
            pred = output.argmax(dim=1)
            test_correct += (pred == target).sum().item()
            test_total += target.size(0)
    
    test_acc = test_correct / test_total
    
    print(f"\n✅ Final: Test={test_acc:.1%}, Best Val={best_val_acc:.1%}, Time={total_time:.1f}s")
    
    return {
        'task': task_name,
        'model': model_name,
        'input_dim': config.input_dim,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'train_time_s': total_time,
        'params': sum(p.numel() for p in model.parameters())
    }


def dimension_scaling_experiment(model_name='TPEqProp', epochs=10):
    """Test model across different input dimensions."""
    
    print(f"\n🔬 Dimension Scaling: {model_name}")
    print("="*70)
    
    # Tasks in order of increasing dimension
    tasks = [
        ('digits', 64),
        ('mnist_14x14', 196),
        ('mnist', 784)
    ]
    
    results = []
    
    for task_name, expected_dim in tasks:
        if task_name == 'digits':
            from src.benchmarks import get_task
            task = get_task(task_name)
        else:
            task = get_intermediate_task(task_name)
        
        result = train_and_eval(model_name, task_name, epochs)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("DIMENSION SCALING SUMMARY")
    print("="*70)
    print(f"{'Task':<15} {'Input Dim':>10} {'Test Acc':>10} {'Δ vs Digits':>12}")
    print("-"*70)
    
    baseline_acc = results[0]['test_acc']
    
    for r in results:
        delta = r['test_acc'] - baseline_acc
        delta_str = f"{delta:+.1%}"
        print(f"{r['task']:<15} {r['input_dim']:>10} {r['test_acc']:>9.1%} {delta_str:>12}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test intermediate tasks')
    parser.add_argument('--task', type=str, default='mnist_14x14',
                       help='Task: mnist_14x14, mnist_binary')
    parser.add_argument('--model', type=str, default='TPEqProp')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--scaling', action='store_true',
                       help='Run dimension scaling experiment')
    parser.add_argument('--output', type=str, default='results/intermediate_tasks.json')
    args = parser.parse_args()
    
    if args.scaling:
        results = dimension_scaling_experiment(args.model, args.epochs)
    else:
        result = train_and_eval(args.model, args.task, args.epochs)
        results = [result]
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved to {args.output}")


if __name__ == '__main__':
    main()
