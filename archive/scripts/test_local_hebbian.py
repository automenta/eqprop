#!/usr/bin/env python3
"""Test LocalHebbianUpdate integration in trainer."""

import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim

from src.models import LoopedMLP
from src.training import EqPropTrainer
from src.training.updates import LocalHebbianUpdate
from src.tasks import get_task_loader


def quick_test():
    """Quick smoke test of LocalHebbianUpdate."""
    print("=" * 70)
    print("LocalHebbianUpdate Integration Test")
    print("=" * 70)
    
    train_loader, test_loader, input_dim, output_dim = get_task_loader(
        'digits', batch_size=64, dataset_size=500
    )
    
    # Create model
    model = LoopedMLP(input_dim, 128, output_dim,
                     symmetric=True, use_spectral_norm=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer with LocalHebbianUpdate
    update_strategy = LocalHebbianUpdate(beta=0.22)
    trainer = EqPropTrainer(model, optimizer, beta=0.22, max_steps=25,
                           update_strategy=update_strategy)
    
    print("\nTraining with LocalHebbianUpdate...")
    print("-" * 50)
    
    # Train for a few epochs
    for epoch in range(5):
        total_loss = 0
        total_batches = 0
        
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            metrics = trainer.step(x, y)
            total_loss += metrics['loss']
            total_batches += 1
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x, steps=25)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100. * correct / total
        print(f"  Epoch {epoch+1}: Loss={total_loss/total_batches:.4f}, Acc={acc:.2f}%")
    
    print("\n" + "=" * 70)
    print("✓ Integration test passed!")
    print("=" * 70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)
    quick_test()
