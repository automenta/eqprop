#!/usr/bin/env python3
"""
Ultra-Fast CIFAR-10 Mini Demo

Validates ConvEqProp on tiny subset for quick testing (< 2 minutes).
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import time

# Test imports
print("Testing imports...")
from src.models import ConvEqProp
print("✓ ConvEqProp imported")

def get_mini_cifar10(num_train=500, num_test=200):
    """Get small CIFAR-10 subset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Subset
    train_subset = Subset(trainset, list(range(num_train)))
    test_subset = Subset(testset, list(range(num_test)))
    
    trainloader = DataLoader(train_subset, batch_size=50, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=50, shuffle=False)
    
    return trainloader, testloader

def test_accuracy(model, testloader, device):
    """Quick accuracy test."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x, steps=15)  # Fewer steps for speed
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return 100. * correct / total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("⚠️  Warning: Running on CPU, will be slow!")
    
    print("\n" + "="*60)
    print("CIFAR-10 MINI DEMO (500 train / 200 test samples)")
    print("="*60)
    
    # Data
    print("\n[1/3] Loading data subset...")
    trainloader, testloader = get_mini_cifar10()
    print(f"  Train batches: {len(trainloader)}, Test batches: {len(testloader)}")
    
    # Model
    print("\n[2/3] Creating ConvEqProp model...")
    model = ConvEqProp(
        input_channels=3,
        hidden_channels=64,
        output_dim=10,
        use_spectral_norm=True
    ).to(device)
    print("  ✓ Model created with spectral normalization")
    
    # Test initial accuracy
    initial_acc = test_accuracy(model, testloader, device)
    print(f"  Initial accuracy: {initial_acc:.1f}% (random baseline ≈ 10%)")
    
    # Training
    print("\n[3/3] Training for 5 epochs...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(5):
        epoch_start = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x, steps=15)  # Reduced steps for speed
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # Progress
            if (batch_idx + 1) % 3 == 0:
                curr_acc = 100. * correct / total
                print(f"    Epoch {epoch+1}/5 [{batch_idx+1}/{len(trainloader)}] "
                      f"Loss: {total_loss/(batch_idx+1):.3f}, Train Acc: {curr_acc:.1f}%",
                      end='\r', flush=True)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        train_acc = 100. * correct / total
        test_acc = test_accuracy(model, testloader, device)
        
        print(f"    Epoch {epoch+1}/5: Train={train_acc:.1f}%, Test={test_acc:.1f}%, Time={epoch_time:.1f}s")
    
    # Final results
    final_acc = test_accuracy(model, testloader, device)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Initial accuracy: {initial_acc:.1f}%")
    print(f"  Final accuracy: {final_acc:.1f}%")
    print(f"  Improvement: {final_acc - initial_acc:+.1f}%")
    
    if final_acc > 25:
        print("\n  ✓ SUCCESS: Model learning confirmed (>25% on subset)")
        print("  Note: Full CIFAR-10 training would reach 60-70%")
    elif final_acc > initial_acc + 5:
        print("\n  ✓ Model is learning (accuracy improved)")
    else:
        print("\n  ⚠️  Model may not be learning properly")
    
    print("\n  To run full benchmark:")
    print("  python scripts/train_cifar10.py --quick")

if __name__ == "__main__":
    main()
