#!/usr/bin/env python3
"""
Quick MNIST Steps Test

Test if more equilibrium steps improve MNIST accuracy.
Simple version without matplotlib dependency.
"""

import argparse
from pathlib import Path
import torch
import torch.optim as optim

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY
from src.benchmarks import get_task
from src.training import EqPropTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_step_counts(model_name='TPEqProp', step_counts=[25, 50, 100], epochs=2, limit_batches=100):
    """Test different step counts on MNIST."""
    
    print(f"\n🧪 Testing Step Counts: {model_name}")
    print(f"   Steps: {step_counts}, Epochs: {epochs}, Batches: {limit_batches}")
    print(f"   Device: {DEVICE}")
    print("="*70)
    
    # Get MNIST data
    task = get_task('mnist')
    config = task.get_config()
    train_loader, _, test_loader = task.get_loaders()
    
    # Limit data for speed
    if limit_batches:
        train_subset = []
        for i, batch in enumerate(train_loader):
            if i >= limit_batches:
                break
            train_subset.append(batch)
        
        test_subset = []
        for i, batch in enumerate(test_loader):
            if i >= limit_batches // 2:
                break
            test_subset.append(batch)
    
    results = []
    
    for steps in step_counts:
        print(f"\n📊 Testing {steps} steps...")
        
        torch.manual_seed(42)
        
        # Create model
        model = MODEL_REGISTRY[model_name](
            input_dim=784,
            hidden_dim=128,  # Use smaller for speed
            output_dim=10,
            use_spectral_norm=True
        ).to(DEVICE)
        
        # Train
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trainer = EqPropTrainer(model, optimizer, beta=0.15, max_steps=steps)
        
        for epoch in range(epochs):
            model.train()
            data_source = train_subset if limit_batches else train_loader
            
            for batch_data in (data_source if limit_batches else train_loader):
                data, target = batch_data
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                if not limit_batches:
                    # Need to check batch limit
                    pass
                
                try:
                    trainer.step(data, target)
                except Exception as e:
                    continue
            
            print(f"  Epoch {epoch+1}/{epochs} done")
        
        # Evaluate
        model.eval()
        correct = total = 0
        
        with torch.no_grad():
            data_source = test_subset if limit_batches else test_loader
            
            for batch_data in data_source:
                data, target = batch_data
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                try:
                    output = model(data, steps=steps)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                except Exception:
                    continue
        
        acc = correct / max(total, 1)
        results.append({'steps': steps, 'accuracy': acc})
        print(f"  → Accuracy: {acc:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Steps':<10} {'Accuracy':>12} {'vs 25 steps':>15}")
    print("-"*70)
    
    baseline_acc = results[0]['accuracy']
    
    for r in results:
        delta = r['accuracy'] - baseline_acc
        delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        print(f"{r['steps']:<10} {r['accuracy']:>11.1%} {delta_str:>15}")
    
    improvement = results[-1]['accuracy'] - results[0]['accuracy']
    
    print("\n" + "="*70)
    if improvement > 0.02:
        print(f"✅ CONFIRMED: More steps help! (+{improvement:.1%})")
        print(f"   Hypothesis 1 (equilibrium) likely correct")
    elif improvement > 0:
        print(f"⚠️  Small improvement (+{improvement:.1%})")
        print(f"   May need even more steps or different approach")
    else:
        print(f"❌ No improvement")
        print(f"   Hypothesis 3 (architecture) more likely")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TPEqProp')
    parser.add_argument('--steps', type=str, default='25,50,100,150',
                       help='Comma-separated step counts')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--limit-batches', type=int, default=100)
    args = parser.parse_args()
    
    steps = [int(s) for s in args.steps.split(',')]
    
    test_step_counts(args.model, steps, args.epochs, args.limit_batches)


if __name__ == '__main__':
    main()
