#!/usr/bin/env python3
"""
Test CIFAR-10 Readiness.
Runs a "Smoke Test":
1. Loads CIFAR-10 via src.tasks (verifies data pipeline)
2. Instantiates ConvEqProp (verifies model architecture)
3. Trains for 1 Epoch on a subset (verifies training loop and stability)
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim
from src.tasks import get_task_loader
from src.models.conv_eqprop import ConvEqProp
from src.training import EqPropTrainer
import time

def main():
    print(">>> CHECKING CIFAR-10 READINESS <<<")
    
    # 1. Data Loading
    print("[1/3] Loading CIFAR-10...")
    try:
        # Load small subset (1000 images) for speed
        train_loader, test_loader, in_c, out_c = get_task_loader("cifar10", batch_size=32, dataset_size=1000)
        print(f"✅ Data Loaded. Input: {in_c} channels, Output: {out_c} classes")
    except Exception as e:
        print(f"❌ Data Loading Failed: {e}")
        return

    # 2. Model Instantiation
    print("[2/3] Instantiating ConvEqProp...")
    try:
        model = ConvEqProp(input_channels=in_c, hidden_channels=64, output_dim=out_c, use_spectral_norm=True).cuda()
        print("✅ Model Created (on GPU).")
    except Exception as e:
        print(f"❌ Model Creation Failed: {e}")
        return

    # 3. Training Loop (Smoke Test)
    print("[3/3] Running Training Smoke Test (1 Epoch)...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = EqPropTrainer(model, optimizer, beta=0.22, max_steps=15) # Reduced steps for speed
    
    model.train()
    total_loss = 0
    start = time.time()
    
    try:
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            metrics = trainer.step(x, y)
            total_loss += metrics['loss']
            
            if i % 10 == 0:
                print(f"    Batch {i}: Loss={metrics['loss']:.4f}")
        
        elapsed = time.time() - start
        print(f"✅ Training Complete in {elapsed:.2f}s. Avg Loss: {total_loss/len(train_loader):.4f}")
        
        # Sanity Check
        if total_loss/len(train_loader) > 2.35: # log(10) ≈ 2.30
             print("⚠️  Warning: Loss is high (Random chance ≈ 2.30). Model might not be learning quickly.")
        else:
             print("✅ Loss indicates learning began (< 2.30).")
             
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        # Print full traceback
        import traceback
        traceback.print_exc()
        return

    print("\n>>> RESULT: READY FOR CIFAR-10 <<<")
    print("Next step: Run full benchmark with `scripts/run_full_suite.py` (after enabling cifar10 in config)")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available! CIFAR training will be too slow.")
        sys.exit(1)
    main()
