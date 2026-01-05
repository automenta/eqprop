#!/usr/bin/env python3
"""
Test ModernEqProp on MNIST

Following RL success, test if ModernEqProp's simplicity helps vision too.
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY
from src.training import EqPropTrainer
from src.benchmarks import get_task


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_mnist(model_name='ModernEqProp', epochs=10, hidden_dim=256, steps=30):
    """Test model on MNIST."""
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name} on MNIST")
    print(f"  Hidden: {hidden_dim}, Steps: {steps}, Epochs: {epochs}")
    print(f"{'='*60}")
    
    # Get MNIST
    task = get_task('mnist')
    config = task.get_config()
    train_loader, val_loader, test_loader = task.get_loaders()
    
    # Create model
    torch.manual_seed(42)
    model = MODEL_REGISTRY[model_name](
        input_dim=784,
        hidden_dim=hidden_dim,
        output_dim=10,
        use_spectral_norm=False  # Try without for simplicity
    ).to(DEVICE)
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = EqPropTrainer(model, optimizer, beta=0.22, max_steps=steps)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            try:
                trainer.step(data, target)
            except Exception as e:
                if batch_idx == 0:
                    print(f"  Training error: {e}")
                continue
        
        # Eval on validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data, steps=steps)
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)
        
        print(f"  Epoch {epoch+1}/{epochs}: Val={val_acc:.1%}")
    
    # Test
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data, steps=steps)
            pred = output.argmax(dim=1)
            test_correct += (pred == target).sum().item()
            test_total += target.size(0)
    
    test_acc = test_correct / test_total
    
    print(f"\n✅ Test: {test_acc:.1%}, Best Val: {best_val_acc:.1%}")
    
    return {
        'model': model_name,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ModernEqProp')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    
    result = test_mnist(args.model, args.epochs, args.hidden_dim, args.steps)
    
    print(f"\nFinal result: {result['test_acc']:.1%}")


if __name__ == '__main__':
    main()
