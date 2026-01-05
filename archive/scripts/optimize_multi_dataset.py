import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
from pathlib import Path
from src.tasks import get_task_loader
from src.models import (
    BackpropMLP, 
    LoopedMLP, 
    ModernEqProp, 
    MSTEP, 
    EnhancedMSTEP
)
from src.training import EqPropTrainer
from scripts.multi_dataset_benchmark import BenchmarkConfig, VISION_BENCHMARKS, RL_BENCHMARKS, train_backprop, train_eqprop

# Define search space - reduced for speed
PARAM_GRID = {
    'lr': [0.001, 0.002],
    'beta': [0.22, 0.5],
    'max_steps': [15, 30]
}

def optimize_task(config, model_cls, seeds=1, device='cpu'):
    print(f"\nOptimization for {config.name} - {model_cls.__name__}")
    
    best_acc = 0.0
    best_params = {}
    
    # Grid search
    total_configs = len(PARAM_GRID['lr']) * len(PARAM_GRID['beta']) * len(PARAM_GRID['max_steps'])
    current = 0
    
    for lr in PARAM_GRID['lr']:
        for beta in PARAM_GRID['beta']:
            for steps in PARAM_GRID['max_steps']:
                current += 1
                print(f"[{current}/{total_configs}] Testing LR={lr}, Beta={beta}, Steps={steps}...", end="", flush=True)
                
                # Run quick evaluation (fewer epochs for speed)
                eval_epochs = 5 
                avg_acc = 0.0
                
                try:
                    for seed in range(seeds):
                        torch.manual_seed(seed)
                        
                        # Load data
                        # Recieves (train_loader, test_loader, input_dim, output_dim)
                        # We ignore dims as they are in config
                        train_loader, test_loader, _, _ = get_task_loader(
                            config.task_name, 
                            batch_size=config.batch_size,
                            dataset_size=1000  # Subsample for speed
                        )
                        
                        # Init model
                        model = model_cls(
                            config.input_dim, 
                            config.hidden_dim, 
                            config.output_dim, 
                            use_spectral_norm=True
                        ).to(device)
                        
                        # Train
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=steps)
                        
                        for _ in range(eval_epochs):
                            model.train()
                            for x, y in train_loader:
                                x, y = x.to(device), y.to(device)
                                trainer.step(x, y)
                        
                        # Evaluate
                        model.eval()
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for x, y in test_loader:
                                x, y = x.to(device), y.to(device)
                                out = model(x)
                                pred = out.argmax(dim=1)
                                correct += (pred == y).sum().item()
                                total += y.size(0)
                        
                        avg_acc += (correct / total) * 100
                    
                    avg_acc /= seeds
                    print(f" Acc: {avg_acc:.2f}%")
                    
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        best_params = {'lr': lr, 'beta': beta, 'max_steps': steps}
                        print(f"  >>> NEW BEST: {best_acc:.2f}%")
                        
                except Exception as e:
                    print(f" Failed: {e}")
                    
    return best_acc, best_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='all', help='Comma-separated task names or "all"')
    parser.add_argument('--models', type=str, default='LoopedMLP', help='Comma-separated model names')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running optimization on {device}")
    
    # Select tasks
    tasks = []
    all_configs = VISION_BENCHMARKS + RL_BENCHMARKS
    if args.tasks == 'all':
        tasks = all_configs
    else:
        target_names = args.tasks.lower().split(',')
        tasks = [c for c in all_configs if any(t in c.name.lower() for t in target_names)]
    
    # Select models
    model_classes = {
        'LoopedMLP': LoopedMLP,
        'ModernEqProp': ModernEqProp,
        'MSTEP': MSTEP,
        'EnhancedMSTEP': EnhancedMSTEP
    }
    models_to_test = [model_classes[m] for m in args.models.split(',') if m in model_classes]
    
    results = {}
    
    for config in tasks:
        results[config.name] = {}
        for model_cls in models_to_test:
            acc, params = optimize_task(config, model_cls, seeds=1, device=device)
            results[config.name][model_cls.__name__] = {
                'best_acc': acc,
                'params': params
            }
            
    # Save optimized config
    with open('results/optimized_hyperparameters.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\nOPTIMIZATION COMPLETE. Best parameters saved.")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
