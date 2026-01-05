#!/usr/bin/env python3
"""
Training script for algorithmic reasoning tasks.

Tests the adaptive compute hypothesis: do harder instances need more iterations?

Usage:
    python train_algorithmic.py --task parity --seq-len 8 --epochs 10
    python train_algorithmic.py --task addition --n-digits 4 --epochs 20
    python train_algorithmic.py --task reversal --seq-len 6 --epochs 15
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.algorithmic_tasks import (
    ParityDataset, ReversalDataset, CopyDataset, AdditionDataset,
    get_algorithmic_loader, TASK_INFO
)
from src.config import TorEqPropConfig
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
from src.trainer import EqPropTrainer


def create_model(input_dim: int, num_classes: int, config):
    """Create model for algorithmic tasks."""
    embedding = nn.Linear(input_dim, config.d_model)
    
    model = LoopedTransformerBlock(
        config.d_model,
        config.n_heads,
        config.d_ff,
        dropout=config.dropout,
        attention_type=config.attention_type,
        symmetric=config.symmetric
    )
    
    output_head = nn.Linear(config.d_model, num_classes)
    
    return embedding, model, output_head


def train_epoch(trainer, embedding, train_loader, device, log_iterations=False):
    """Train for one epoch, optionally logging per-sample iterations."""
    trainer.model.train()
    embedding.train()
    trainer.output_head.train()
    
    total_loss = 0
    total_acc = 0
    all_iterations = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Handle multi-output tasks (reversal, copy)
        if len(target.shape) > 1:
            # For sequence output, just use first position for now
            target = target[:, 0]
        
        # Embed and add sequence dimension
        x_emb = embedding(data).unsqueeze(0)
        
        metrics = trainer.train_step(x_emb, target)
        
        total_loss += metrics["loss"]
        total_acc += metrics["accuracy"]
        
        if log_iterations:
            all_iterations.append({
                "batch": batch_idx,
                "iters_free": metrics["iters_free"],
                "iters_nudged": metrics["iters_nudged"]
            })
    
    n_batches = len(train_loader)
    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "iterations": all_iterations
    }


def evaluate(solver, model, embedding, output_head, test_loader, device, analyze_difficulty=False):
    """Evaluate on test set, optionally analyzing iteration vs difficulty."""
    model.eval()
    embedding.eval()
    output_head.eval()
    
    total_acc = 0
    total_loss = 0
    difficulty_analysis = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            if len(target.shape) > 1:
                target = target[:, 0]
            
            x_emb = embedding(data).unsqueeze(0)
            h0 = torch.zeros_like(x_emb)
            
            # Track iterations for analysis
            h_fixed, iters = solver.solve(model, h0, x_emb)
            
            y_pred = output_head(h_fixed.mean(dim=0))
            
            loss = nn.functional.cross_entropy(y_pred, target)
            acc = (y_pred.argmax(-1) == target).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            if analyze_difficulty:
                # For parity: difficulty = number of 1s
                # For addition: difficulty = number of carries
                difficulty_analysis.append({
                    "batch": batch_idx,
                    "iterations": iters,
                    "accuracy": acc.item(),
                    "loss": loss.item()
                })
    
    n_batches = len(test_loader)
    return {
        "test_loss": total_loss / n_batches,
        "test_accuracy": total_acc / n_batches,
        "difficulty_analysis": difficulty_analysis
    }


def analyze_adaptive_compute(results: list, task: str) -> dict:
    """Analyze correlation between difficulty and iterations."""
    if not results:
        return {"correlation": None, "insight": "No data"}
    
    iterations = [r["iterations"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    
    # Simple analysis
    mean_iters = sum(iterations) / len(iterations)
    std_iters = (sum((i - mean_iters)**2 for i in iterations) / len(iterations)) ** 0.5
    
    insight = f"Iterations: {mean_iters:.1f} ± {std_iters:.2f}"
    
    if std_iters > 0.5:
        insight += " - VARIANCE DETECTED (adaptive?)"
    else:
        insight += " - Uniform convergence"
    
    return {
        "mean_iterations": mean_iters,
        "std_iterations": std_iters,
        "has_variance": std_iters > 0.5,
        "insight": insight
    }


def main():
    parser = argparse.ArgumentParser(description="Train on algorithmic reasoning tasks")
    
    # Task settings
    parser.add_argument("--task", type=str, default="parity",
                        choices=["parity", "reversal", "copy", "addition"],
                        help="Task to train on")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length")
    parser.add_argument("--n-digits", type=int, default=4, help="Number of digits for addition")
    parser.add_argument("--n-samples", type=int, default=10000, help="Training samples")
    
    # Model settings
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=256, help="FFN dimension")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.22, help="EqProp beta")
    parser.add_argument("--max-iters", type=int, default=30, help="Equilibrium iterations")
    parser.add_argument("--tol", type=float, default=1e-4, help="Equilibrium tolerance")
    parser.add_argument("--damping", type=float, default=0.8, help="Equilibrium damping")
    
    # Analysis
    parser.add_argument("--analyze-difficulty", action="store_true",
                        help="Analyze iteration count vs difficulty")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"🧮 Algorithmic Task Training: {args.task.upper()}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Task: {args.task} (seq_len={args.seq_len})")
    print(f"Model: d={args.d_model}, heads={args.n_heads}")
    print(f"Training: {args.epochs} epochs, lr={args.lr}, β={args.beta}")
    print(f"Hypothesis: {TASK_INFO[args.task]['hypothesis']}")
    print(f"{'='*70}\n")
    
    # Create data loaders
    if args.task == "parity":
        train_dataset = ParityDataset(args.n_samples, args.seq_len, seed=42)
        test_dataset = ParityDataset(args.n_samples // 5, args.seq_len, seed=1337)
        input_dim = args.seq_len
        num_classes = 2
    elif args.task == "reversal":
        train_dataset = ReversalDataset(args.n_samples, args.seq_len, seed=42)
        test_dataset = ReversalDataset(args.n_samples // 5, args.seq_len, seed=1337)
        input_dim = args.seq_len * 10
        num_classes = 10
    elif args.task == "copy":
        train_dataset = CopyDataset(args.n_samples, args.seq_len, seed=42)
        test_dataset = CopyDataset(args.n_samples // 5, args.seq_len, seed=1337)
        input_dim = args.seq_len * 10
        num_classes = 10
    elif args.task == "addition":
        train_dataset = AdditionDataset(args.n_samples, args.n_digits, seed=42)
        test_dataset = AdditionDataset(args.n_samples // 5, args.n_digits, seed=1337)
        input_dim = 2 * args.n_digits
        num_classes = 10 ** args.n_digits
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Data: {len(train_dataset)} train, {len(test_dataset)} test")
    print(f"Input dim: {input_dim}, Classes: {num_classes}")
    
    # Create model
    class ConfigStub:
        def __init__(self):
            self.d_model = args.d_model
            self.n_heads = args.n_heads
            self.d_ff = args.d_ff
            self.dropout = 0.1
            self.attention_type = "linear"
            self.symmetric = False
            
    config = ConfigStub()
    embedding, model, output_head = create_model(input_dim, num_classes, config)
    
    embedding = embedding.to(device)
    model = model.to(device)
    output_head = output_head.to(device)
    
    # Create solver and trainer
    solver = EquilibriumSolver(max_iters=args.max_iters, tol=args.tol, damping=args.damping)
    trainer = EqPropTrainer(model, solver, output_head, beta=args.beta, lr=args.lr)
    trainer.optimizer.add_param_group({'params': embedding.parameters()})
    
    # Training loop
    best_acc = 0
    results_log = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_metrics = train_epoch(trainer, embedding, train_loader, device)
        test_metrics = evaluate(
            solver, model, embedding, output_head, test_loader, device,
            analyze_difficulty=args.analyze_difficulty
        )
        
        epoch_time = time.time() - start_time
        
        # Log
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Test Loss: {test_metrics['test_loss']:.4f}, Test Acc: {test_metrics['test_accuracy']:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        results_log.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["test_loss"],
            "test_accuracy": test_metrics["test_accuracy"]
        })
        
        if test_metrics["test_accuracy"] > best_acc:
            best_acc = test_metrics["test_accuracy"]
    
    # Final analysis
    print(f"\n{'='*70}")
    print(f"📊 RESULTS")
    print(f"{'='*70}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Final Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    
    if args.analyze_difficulty and test_metrics["difficulty_analysis"]:
        analysis = analyze_adaptive_compute(test_metrics["difficulty_analysis"], args.task)
        print(f"Adaptive Compute Analysis: {analysis['insight']}")
    
    # Save results
    output_dir = Path("logs/algorithmic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "task": args.task,
            "config": vars(args),
            "best_accuracy": best_acc,
            "results": results_log
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
