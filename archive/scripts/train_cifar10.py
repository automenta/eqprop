#!/usr/bin/env python3
"""
CIFAR-10 Comprehensive Training with ConvEqProp

Runs full CIFAR-10 experiments with statistical rigor:
- Multiple seeds
- Convergence analysis
- Comparison to standard CNN baseline
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import time
import json


@dataclass
class CIFAR10Result:
    """Results from CIFAR-10 training."""
    seed: int
    model_type: str
    final_train_acc: float
    final_test_acc: float
    epochs: int
    training_time: float
    convergence_epoch: int  # Epoch where test acc > 40%


class StandardCNN(nn.Module):
    """Baseline CNN for comparison."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CIFAR10Trainer:
    """Trainer for CIFAR-10 experiments."""
    
    def __init__(self, device='cuda', num_seeds=3):
        self.device = device
        self.num_seeds = num_seeds
        self.results = []
    
    def get_cifar10_loaders(self, batch_size=128):
        """Get CIFAR-10 data loaders."""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    def train_epoch(self, model, trainloader, optimizer, is_eqprop=False, epoch=0, total_epochs=1):
        """Train for one epoch with progress feedback."""
        import sys
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        num_batches = len(trainloader)
        
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            if is_eqprop:
                out = model(x, steps=20)  # ConvEqProp
            else:
                out = model(x)  # Standard CNN
            
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # Progress feedback every 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start_time
                batches_done = batch_idx + 1
                batches_left = num_batches - batches_done
                time_per_batch = elapsed / batches_done
                eta = time_per_batch * batches_left
                
                curr_loss = total_loss / batches_done
                curr_acc = 100. * correct / total
                
                print(f"\r  Epoch {epoch+1}/{total_epochs} [{batches_done}/{num_batches}] "
                      f"Loss: {curr_loss:.3f}, Acc: {curr_acc:.1f}%, ETA: {eta:.0f}s",
                      end='', flush=True)
        
        print()  # New line after progress
        return total_loss / len(trainloader), 100. * correct / total
    
    def test(self, model, testloader, is_eqprop=False):
        """Test accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)
                
                if is_eqprop:
                    out = model(x, steps=20)
                else:
                    out = model(x)
                
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return 100. * correct / total
    
    def run_experiment(self, model_type='conv_eqprop', seed=0, epochs=20, mini=False):
        """Run one experiment.
        
        Args:
            mini: If True, use tiny subset for ultra-fast demo (< 1 min)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"\n{'='*60}")
        print(f"{model_type.upper()} - Seed {seed}")
        print(f"{'='*60}")
        
        # Create model
        if model_type == 'conv_eqprop':
            from src.models import ConvEqProp
            model = ConvEqProp(
                input_channels=3,
                hidden_channels=64,
                output_dim=10,
                use_spectral_norm=True
            ).to(self.device)
            is_eqprop = True
        else:
            model = StandardCNN(num_classes=10).to(self.device)
            is_eqprop = False
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Data
        if mini:
            # Ultra-fast mini demo: 500 train, 200 test samples
            trainloader, testloader = self.get_cifar10_loaders(batch_size=50)
            # Limit to first few batches
            print("  [Mini Mode: 500 train / 200 test samples for quick validation]")
        else:
            trainloader, testloader = self.get_cifar10_loaders()
        
        # Training
        print(f"\n  Starting training... (Estimated: {epochs * 60 if not mini else epochs * 10}s total)")
        start_time = time.time()
        best_test_acc = 0
        convergence_epoch = epochs
        
        train_loader_limited = list(trainloader)[:10] if mini else trainloader
        test_loader_limited = list(testloader)[:4] if mini else testloader
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                model, train_loader_limited, optimizer, is_eqprop,
                epoch=epoch, total_epochs=epochs
            )
            test_acc = self.test(model, test_loader_limited, is_eqprop)
            scheduler.step()
            
            # Track convergence (40% is reasonable for CIFAR-10)
            if test_acc > 40 and convergence_epoch == epochs:
                convergence_epoch = epoch + 1
            
            best_test_acc = max(best_test_acc, test_acc)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:2d}/{epochs}: "
                      f"Train={train_acc:.1f}%, Test={test_acc:.1f}%, "
                      f"Best={best_test_acc:.1f}%")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_train_acc = train_acc
        final_test_acc = self.test(model, testloader, is_eqprop)
        
        result = CIFAR10Result(
            seed=seed,
            model_type=model_type,
            final_train_acc=final_train_acc,
            final_test_acc=final_test_acc,
            epochs=epochs,
            training_time=training_time,
            convergence_epoch=convergence_epoch
        )
        
        print(f"\n  Final Test Accuracy: {final_test_acc:.1f}%")
        print(f"  Convergence Epoch: {convergence_epoch}/{epochs}")
        print(f"  Training Time: {training_time:.1f}s")
        
        return result
    
    def run_full_comparison(self, epochs=20):
        """Run full comparison: ConvEqProp vs Standard CNN."""
        print("\n" + "="*70)
        print("CIFAR-10 COMPREHENSIVE EVALUATION")
        print(f"Seeds: {self.num_seeds}, Epochs: {epochs}")
        print("="*70)
        
        results = {'conv_eqprop': [], 'standard_cnn': []}
        
        # Run ConvEqProp
        for seed in range(self.num_seeds):
            result = self.run_experiment('conv_eqprop', seed=seed, epochs=epochs)
            results['conv_eqprop'].append(result)
        
        # Run Standard CNN baseline
        for seed in range(self.num_seeds):
            result = self.run_experiment('standard_cnn', seed=seed, epochs=epochs)
            results['standard_cnn'].append(result)
        
        # Summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict[str, List[CIFAR10Result]]):
        """Print comparison summary."""
        print("\n" + "="*70)
        print("SUMMARY: ConvEqProp vs Standard CNN on CIFAR-10")
        print("="*70)
        
        for model_type in ['conv_eqprop', 'standard_cnn']:
            res_list = results[model_type]
            
            test_accs = [r.final_test_acc for r in res_list]
            conv_epochs = [r.convergence_epoch for r in res_list]
            times = [r.training_time for r in res_list]
            
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            mean_conv = np.mean(conv_epochs)
            mean_time = np.mean(times)
            
            print(f"\n{model_type.upper().replace('_', ' ')}:")
            print(f"  Test Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%")
            print(f"  Convergence: Epoch {mean_conv:.1f}/{res_list[0].epochs}")
            print(f"  Training Time: {mean_time:.1f}s per run")
        
        # Gap analysis
        eqprop_acc = np.mean([r.final_test_acc for r in results['conv_eqprop']])
        cnn_acc = np.mean([r.final_test_acc for r in results['standard_cnn']])
        gap = eqprop_acc - cnn_acc
        
        print(f"\nGap: {gap:+.1f}% (EqProp vs CNN)")
        
        if abs(gap) < 5:
            print("✓ On-par performance (gap < 5%)")
        else:
            print("⚠ Significant gap detected")


def quick_demo(epochs=10):
    """Quick demo with 1 seed."""
    trainer = CIFAR10Trainer(num_seeds=1)
    
    print("Running quick CIFAR-10 demo (1 seed, 10 epochs)...")
    result = trainer.run_experiment('conv_eqprop', seed=0, epochs=epochs)
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CIFAR-10 with ConvEqProp")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Quick demo mode")
    parser.add_argument("--output", type=str, default="results/cifar10_results.json")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, training will be slow!")
    
    if args.quick:
        quick_demo(epochs=10)
    else:
        trainer = CIFAR10Trainer(num_seeds=args.seeds)
        results = trainer.run_full_comparison(epochs=args.epochs)
        
        # Save results
        import os
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        save_data = {}
        for model_type, res_list in results.items():
            save_data[model_type] = [
                {
                    'seed': r.seed,
                    'final_test_acc': r.final_test_acc,
                    'convergence_epoch': r.convergence_epoch,
                    'training_time': r.training_time
                }
                for r in res_list
            ]
        
        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
