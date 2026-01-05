#!/usr/bin/env python3
"""
CIFAR-10 Hierarchical Model Evaluation with Hyperparameter Sweep

Comprehensive testing of hierarchical/convolutional EqProp models on CIFAR-10
with systematic hyperparameter optimization.

Models tested:
- ConvEqProp (baseline convolutional)
- EnhancedMSTEP (hierarchical multi-scale)

Hyperparameters swept:
- beta: nudge strength
- learning_rate: optimizer learning rate
- hidden_channels/dim: model capacity
- max_steps: equilibrium iteration budget

Output: results/cifar10_hierarchical_sweep.json
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim
import argparse
import json
import time
import numpy as np
from itertools import product
from pathlib import Path

from src.tasks import get_task_loader
from src.models.conv_eqprop import ConvEqProp
from src.models.mstep_enhanced import EnhancedMSTEP
from src.training import EqPropTrainer


def train_model(model, train_loader, test_loader, epochs, lr, beta, max_steps):
    """Train a model and return final accuracy + metrics."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=max_steps)
    
    history = {'train_loss': [], 'test_acc': [], 'convergence_rate': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_converged = 0
        total_batches = 0
        
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            metrics = trainer.step(x, y)
            total_loss += metrics['loss']
            total_converged += int(metrics.get('converged_free', False))
            total_batches += 1
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x, steps=max_steps)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100. * correct / total
        conv_rate = total_converged / total_batches
        avg_loss = total_loss / total_batches
        
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(acc)
        history['convergence_rate'].append(conv_rate)
        
        # Check for early stopping (no learning)
        if epoch > 5 and all(a < 15.0 for a in history['test_acc'][-3:]):
            print(f"      Early stopping at epoch {epoch+1} - no learning detected (acc <15%)")
            break
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Acc={acc:.2f}%, Conv={conv_rate:.1%}")
    
    elapsed = time.time() - start_time
    final_acc = history['test_acc'][-1]
    best_acc = max(history['test_acc'])
    
    return {
        'final_acc': final_acc,
        'best_acc': best_acc,
        'training_time': elapsed,
        'history': history,
        'converged': final_acc > 15.0  # Did it learn anything?
    }


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Hierarchical Model Sweep")
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--beta-values', type=str, default='0.18,0.20,0.22,0.25',
                       help='Comma-separated beta values')
    parser.add_argument('--lr-values', type=str, default='0.0005,0.001,0.002',
                       help='Comma-separated learning rates')
    parser.add_argument('--hidden-values', type=str, default='64,128',
                       help='Comma-separated hidden channel/dim values')
    parser.add_argument('--steps-values', type=str, default='15,25',
                       help='Comma-separated max_steps values')
    parser.add_argument('--models', type=str, default='ConvEqProp,EnhancedMSTEP',
                       help='Comma-separated model names')
    args = parser.parse_args()
    
    # Parse hyperparameter grids
    betas = [float(v) for v in args.beta_values.split(',')]
    lrs = [float(v) for v in args.lr_values.split(',')]
    hidden_vals = [int(v) for v in args.hidden_values.split(',')]
    steps_vals = [int(v) for v in args.steps_values.split(',')]
    model_names = args.models.split(',')
    
    print("=" * 80)
    print("CIFAR-10 HIERARCHICAL MODEL SWEEP")
    print("=" * 80)
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Models: {model_names}")
    print(f"Beta values: {betas}")
    print(f"Learning rates: {lrs}")
    print(f"Hidden sizes: {hidden_vals}")
    print(f"Max steps: {steps_vals}")
    print()
    
    # Load CIFAR-10
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_channels, output_dim = get_task_loader(
        'cifar10', batch_size=128, dataset_size=None  # Use full dataset
    )
    print(f"Dataset loaded: {input_channels} channels, {output_dim} classes")
    print()
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}\n")
        
        # Hyperparameter grid for this model
        if 'Conv' in model_name:
            # ConvEqProp uses hidden_channels
            grid = list(product(betas, lrs, hidden_vals, steps_vals))
        else:
            # EnhancedMSTEP uses hidden_dim (will be 4x for input embedding)
            grid = list(product(betas, lrs, [h*4 for h in hidden_vals], steps_vals))
        
        print(f"Total configurations to test: {len(grid)} × {args.seeds} seeds = {len(grid) * args.seeds} runs")
        print()
        
        best_config = None
        best_score = 0
        
        config_results = []
        
        for config_idx, (beta, lr, hidden, max_steps) in enumerate(grid, 1):
            print(f"[{config_idx}/{len(grid)}] Testing: β={beta}, lr={lr}, hidden={hidden}, steps={max_steps}")
            
            seed_accs = []
            seed_results = []
            
            for seed in range(42, 42 + args.seeds):
                print(f"  Seed {seed}...")
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                # Create model
                if model_name == 'ConvEqProp':
                    model = ConvEqProp(
                        input_channels=input_channels,
                        hidden_channels=hidden,
                        output_dim=output_dim,
                        use_spectral_norm=True
                    ).cuda()
                elif model_name == 'EnhancedMSTEP':
                    model = EnhancedMSTEP(
                        input_dim=input_channels * 32 * 32,  # Flattened CIFAR
                        hidden_dim=hidden,
                        output_dim=output_dim,
                        use_spectral_norm=True
                    ).cuda()
                else:
                    print(f"Unknown model: {model_name}")
                    continue
                
                # Train
                result = train_model(model, train_loader, test_loader, 
                                   args.epochs, lr, beta, max_steps)
                
                seed_accs.append(result['final_acc'])
                seed_results.append(result)
                
                print(f"      Final: {result['final_acc']:.2f}%, Best: {result['best_acc']:.2f}%")
            
            # Aggregate across seeds
            mean_acc = np.mean(seed_accs)
            std_acc = np.std(seed_accs)
            
            config_results.append({
                'config': {'beta': beta, 'lr': lr, 'hidden': hidden, 'max_steps': max_steps},
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'seeds': seed_accs,
                'seed_details': seed_results
            })
            
            print(f"  --> Mean: {mean_acc:.2f} ± {std_acc:.2f}%")
            
            if mean_acc > best_score:
                best_score = mean_acc
                best_config = {'beta': beta, 'lr': lr, 'hidden': hidden, 'max_steps': max_steps}
            
            print()
        
        # Summary for this model
        all_results[model_name] = {
            'best_config': best_config,
            'best_accuracy': best_score,
            'all_configs': config_results
        }
        
        print(f"\n{'='*80}")
        print(f"{model_name} BEST RESULT:")
        print(f"{'='*80}")
        print(f"Config: {best_config}")
        print(f"Accuracy: {best_score:.2f}%")
        print()
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    for model_name, results in all_results.items():
        print(f"{model_name:30} Best: {results['best_accuracy']:.2f}%")
        print(f"  Config: {results['best_config']}")
        print()
    
    # Save results
    output_file = 'results/cifar10_hierarchical_sweep.json'
    Path('results').mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Also save best config separately
    best_configs = {name: res['best_config'] for name, res in all_results.items()}
    with open('results/cifar10_best_config.json', 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    print(f"Best configs saved to results/cifar10_best_config.json")
    print("=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available! CIFAR-10 training requires GPU.")
        sys.exit(1)
    main()
