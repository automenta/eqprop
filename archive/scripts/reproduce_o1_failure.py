#!/usr/bin/env python3
"""
Reproduction script for O(1) memory training failure.
Effectively a unit test for LocalHebbianUpdate.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, '.')

from src.models import ModernEqProp
from src.training import EqPropTrainer
from src.training.updates import LocalHebbianUpdate

def get_dummy_data(samples=1000):
    """Generate simple linearly separable data."""
    torch.manual_seed(42)
    X = torch.randn(samples, 784)
    # Simple rule: if sum of first 10 pixels > 0, class 1, else 0
    y = (X[:, :10].sum(dim=1) > 0).long()
    return X, y

def test_local_hebbian_learning():
    print("="*60)
    print("TESTING LOCAL HEBBIAN UPDATE LEARNING")
    print("="*60)
    
    # 1. Setup Data
    X, y = get_dummy_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=100, shuffle=True) # Increased batch size for better Hebbian stats
    
    # 2. Setup Model (ModernEqProp)
    model = ModernEqProp(
        input_dim=784,
        hidden_dim=128,
        output_dim=2,
        use_spectral_norm=True
    )
    
    # 3. Setup Trainer with LocalHebbianUpdate
    optimizer = optim.Adam(model.parameters(), lr=0.002) # Slightly higher LR
    
    # Use beta=0.22 as standard
    hebbian_update = LocalHebbianUpdate(beta=0.22)
    trainer = EqPropTrainer(
        model, 
        optimizer, 
        beta=0.22, 
        max_steps=20,
        update_strategy=hebbian_update
    )
    
    print("\nTraining for 15 epochs...")
    accuracies = []
    
    for epoch in range(1, 16):
        total_acc = 0
        batches = 0
        
        for x_batch, y_batch in loader:
            metrics = trainer.step(x_batch, y_batch)
            
            # Compute accuracy manually
            with torch.no_grad():
                logits = model(x_batch)
                preds = logits.argmax(dim=1)
                acc = (preds == y_batch).float().mean().item()
            
            total_acc += acc
            batches += 1
            
        avg_acc = total_acc / batches
        accuracies.append(avg_acc)
        print(f"Epoch {epoch}: Accuracy = {avg_acc*100:.2f}%")
        
    final_acc = accuracies[-1] * 100
    print(f"\nFinal Accuracy: {final_acc:.2f}%")
    
    # Validation
    if final_acc > 70.0:
        print("\n✅ SUCCESS: Model is learning with LocalHebbianUpdate!")
        return True
    else:
        print("\n❌ FAILURE: Model is not learning (random chance is 50%).")
        print("   Likely causes: broken update rule, sign error, or state mismatch.")
        return False

if __name__ == "__main__":
    success = test_local_hebbian_learning()
    sys.exit(0 if success else 1)
