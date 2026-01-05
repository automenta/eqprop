#!/usr/bin/env python3
"""Run analysis on trained models to verify theoretical guarantees.

This script:
1. Trains each model briefly on a task
2. Re-runs analytical validation
3. Compares trained vs untrained metrics

Usage:
    python scripts/analyze_trained.py --task xor --epochs 10
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '.')

from src.models import LoopedMLP, ToroidalMLP, ModernEqProp, GatedMLP
from src.training import EqPropTrainer, EquilibriumSolver
from src.analysis import IterationAnalyzer, XORTask, MemorizationTask, AttractorTask


def train_model(model, task, epochs=10, lr=0.01, beta=0.22):
    """Train model on task using EqProp."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=30)
    
    X, Y = task.generate_data(500, device='cpu')
    
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        batch_size = 32
        total_loss = 0
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]
            
            metrics = trainer.step(x_batch, y_batch)
            total_loss += metrics['loss']
        
        if epoch % 5 == 0:
            # Evaluate
            model.eval()
            with torch.no_grad():
                output = model(X, steps=30)
                preds = output.argmax(dim=-1)
                acc = (preds == Y).float().mean().item()
            print(f"  Epoch {epoch}: loss={total_loss:.4f}, acc={acc:.2%}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Analyze Trained Models")
    parser.add_argument("--task", type=str, default="xor", choices=["xor", "memorization", "attractor"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--models", type=str, default="LoopedMLP,ModernEqProp")
    
    args = parser.parse_args()
    
    # Create task
    if args.task == "xor":
        task = XORTask(n_bits=4)
    elif args.task == "memorization":
        task = MemorizationTask(n_patterns=5, input_dim=20, output_dim=5)
    else:
        task = AttractorTask(n_attractors=3, input_dim=20)
    
    model_factories = {
        "LoopedMLP": lambda: LoopedMLP(task.input_dim, 64, task.output_dim, symmetric=True),
        "ToroidalMLP": lambda: ToroidalMLP(task.input_dim, 64, task.output_dim),
        "ModernEqProp": lambda: ModernEqProp(task.input_dim, 64, task.output_dim),
    }
    
    model_names = [m.strip() for m in args.models.split(",")]
    
    print("=" * 60)
    print(f"Trained Model Analysis - Task: {args.task}")
    print("=" * 60)
    
    results = {}
    
    for name in model_names:
        if name not in model_factories:
            print(f"Unknown model: {name}")
            continue
        
        print(f"\n## {name}")
        print("-" * 40)
        
        # Analyze untrained
        model_untrained = model_factories[name]()
        analyzer = IterationAnalyzer(model_untrained, max_steps=50)
        report_untrained = analyzer.analyze_task(task, n_samples=100, validate_theory=True)
        
        print(f"UNTRAINED: Conv={report_untrained.trajectory.converged}, "
              f"L={report_untrained.theoretical.lipschitz_constant:.3f}, "
              f"Acc={report_untrained.task_accuracy:.2%}")
        
        # Train
        print("Training...")
        model_trained = model_factories[name]()
        model_trained = train_model(model_trained, task, epochs=args.epochs)
        
        # Analyze trained
        analyzer_trained = IterationAnalyzer(model_trained, max_steps=50)
        report_trained = analyzer_trained.analyze_task(task, n_samples=100, validate_theory=True)
        
        print(f"TRAINED:   Conv={report_trained.trajectory.converged}, "
              f"L={report_trained.theoretical.lipschitz_constant:.3f}, "
              f"Acc={report_trained.task_accuracy:.2%}")
        
        # Compare
        energy_improved = (report_trained.theoretical.energy_descent_valid and 
                          not report_untrained.theoretical.energy_descent_valid)
        
        results[name] = {
            "untrained": report_untrained.to_dict(),
            "trained": report_trained.to_dict(),
            "energy_improved": energy_improved
        }
        
        print(f"Energy descent improved: {energy_improved}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
