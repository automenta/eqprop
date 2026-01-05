#!/usr/bin/env python3
"""Debug LocalHebbianUpdate dimension issues."""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn

from src.models import LoopedMLP
from src.training.updates import LocalHebbianUpdate


def debug_hebbian():
    """Debug the Hebbian update computation."""
    print("=" * 70)
    print("Debugging LocalHebbianUpdate")
    print("=" * 70)
    
    # Create a simple model
    model = LoopedMLP(20, 64, 10, symmetric=True).cuda()
    
    # Create update strategy
    hebbian = LocalHebbianUpdate(beta=0.22)
    hebbian.register_hooks(model)
    
    # Create dummy data
    x = torch.randn(32, 20).cuda()
    
    # Simulate free phase
    print("\n1. Free phase forward pass...")
    hebbian.phase = 'free'
    _ = model(x, steps=1)
    
    print(f"   Captured activations (free): {len(hebbian.activations_free)} layers")
    for name, act in hebbian.activations_free.items():
        print(f"     {name}: {act.shape}")
    
    # Simulate nudged phase
    print("\n2. Nudged phase forward pass...")
    hebbian.phase = 'nudged'
    _ = model(x, steps=1)
    
    print(f"   Captured activations (nudged): {len(hebbian.activations_nudged)} layers")
    for name, act in hebbian.activations_nudged.items():
        print(f"     {name}: {act.shape}")
    
    # Compute Hebbian updates
    print("\n3. Computing Hebbian updates...")
    try:
        weight_updates = hebbian.compute_hebbian_updates()
        print(f"   Computed updates for {len(weight_updates)} layers")
        for name, update in weight_updates.items():
            print(f"     {name}: {update.shape}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Check model parameter shapes
    print("\n4. Model parameter shapes...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"   {name}.weight: {module.weight.shape}")
    
    # Compare
    print("\n5. Dimension analysis...")
    if weight_updates:
        for name in weight_updates:
            update_shape = weight_updates[name].shape
            # Find matching module
            for mod_name, module in model.named_modules():
                if mod_name == name and isinstance(module, nn.Linear):
                    weight_shape = module.weight.shape
                    print(f"   {name}:")
                    print(f"     Update:  {update_shape}")
                    print(f"     Weight:  {weight_shape}")
                    print(f"     Match:   {update_shape == weight_shape}")


if __name__ == "__main__":
    debug_hebbian()
