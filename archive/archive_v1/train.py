"""Unified training script for TorEqProp on multiple datasets."""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from pathlib import Path

from src.config import TorEqPropConfig
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
from src.trainer import EqPropTrainer

# Global PyTorch optimizations
torch.backends.cudnn.benchmark = True  # Optimize conv/attention for fixed input sizes
torch.set_float32_matmul_precision('high')  # Use TensorCores if available


def get_data_loaders(config: TorEqPropConfig):
    """Create data loaders based on dataset name."""
    from src.datasets import get_data_loaders as _get_loaders
    return _get_loaders(
        config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )


def create_model(config: TorEqPropConfig, input_dim: int, num_classes: int):
    """Create model components based on config."""
    embedding = nn.Linear(input_dim, config.d_model).to(config.device)
    
    model = LoopedTransformerBlock(
        config.d_model,
        config.n_heads,
        config.d_ff,
        dropout=config.dropout,
        attention_type=config.attention_type,
        symmetric=config.symmetric
    ).to(config.device)
    
    output_head = nn.Linear(config.d_model, num_classes).to(config.device)
    
    # Apply torch.compile if requested
    if config.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        embedding = torch.compile(embedding)
        output_head = torch.compile(output_head)
    
    return embedding, model, output_head


def train_epoch(trainer, embedding, train_loader, config, epoch):
    """Train for one epoch."""
    trainer.model.train()
    embedding.train()
    trainer.output_head.train()
    
    total_loss = 0
    total_acc = 0
    total_iters_free = 0
    total_iters_nudged = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(config.device), target.to(config.device)
        
        # Project input: [batch, input_dim] -> [1, batch, d_model]
        x_emb = embedding(data).unsqueeze(0)
        
        metrics = trainer.train_step(x_emb, target)
        
        total_loss += metrics["loss"]
        total_acc += metrics["accuracy"]
        total_iters_free += metrics["iters_free"]
        total_iters_nudged += metrics["iters_nudged"]
        
        if batch_idx % config.log_interval == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {metrics['loss']:.4f} Acc: {metrics['accuracy']:.4f} "
                  f"Iters: {metrics['iters_free']}/{metrics['iters_nudged']}")
    
    n_batches = len(train_loader)
    metrics = {
        "train/loss": total_loss / n_batches,
        "train/accuracy": total_acc / n_batches,
        "train/iters_free": total_iters_free / n_batches,
        "train/iters_nudged": total_iters_nudged / n_batches,
        "train/epoch_time": time.time() - start_time
    }
    
    return metrics


def evaluate(solver, model, embedding, output_head, test_loader, config):
    """Evaluate on test set."""
    model.eval()
    embedding.eval()
    output_head.eval()
    
    test_acc = 0
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            x_emb = embedding(data).unsqueeze(0)
            h0 = torch.zeros_like(x_emb)
            h_fixed, _ = solver.solve(model, h0, x_emb)
            y_pred = output_head(h_fixed.mean(dim=0))
            
            test_loss += nn.functional.cross_entropy(y_pred, target).item()
            test_acc += (y_pred.argmax(-1) == target).float().mean().item()
    
    n_batches = len(test_loader)
    return {
        "test/loss": test_loss / n_batches,
        "test/accuracy": test_acc / n_batches
    }


def main():
    # Parse config
    config = TorEqPropConfig.from_args()
    
    # Initialize wandb if requested
    if config.wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.to_dict(),
                name=f"{config.dataset}_d{config.d_model}_beta{config.beta}"
            )
        except ImportError:
            print("Warning: wandb not installed, logging disabled")
            config.wandb = False
    
    print("="*70)
    print(f"Training TorEqProp on {config.dataset.upper()}")
    print("="*70)
    print(f"Device: {config.device}")
    print(f"Model: d_model={config.d_model}, n_heads={config.n_heads}, d_ff={config.d_ff}")
    print(f"Attention: {config.attention_type}, Symmetric: {config.symmetric}")
    print(f"Beta: {config.beta}, LR: {config.lr}, Damping: {config.damping}")
    print(f"Compile: {config.compile}, Wandb: {config.wandb}")
    print("="*70)
    
    # Setup data
    train_loader, test_loader, input_dim, num_classes = get_data_loaders(config)
    
    # Create model
    embedding, model, output_head = create_model(config, input_dim, num_classes)
    
    # Create solver and trainer
    solver = EquilibriumSolver(
        max_iters=config.max_iters,
        tol=config.tol,
        damping=config.damping
    )
    
    # Create β schedule if annealing is enabled
    beta_schedule = None
    if config.beta_anneal:
        # Linear anneal from 0.3 to final beta over epochs
        beta_start = 0.3
        beta_end = config.beta
        beta_schedule = lambda epoch: beta_start + (beta_end - beta_start) * (epoch / max(1, config.epochs - 1))
    
    trainer = EqPropTrainer(
        model, 
        solver, 
        output_head,
        beta=config.beta if not config.beta_anneal else 0.3,  # Start high if annealing
        lr=config.lr,
        update_mode=config.update_mode,
        beta_schedule=beta_schedule
    )
    
    # Add embedding to optimizer
    trainer.optimizer.add_param_group({'params': embedding.parameters()})
    
    # Training loop
    best_acc = 0
    for epoch in range(config.epochs):
        # Update β if annealing
        trainer.update_beta(epoch)
        if config.beta_anneal:
            print(f"Beta for epoch {epoch}: {trainer.beta:.4f}")
        # Train
        train_metrics = train_epoch(trainer, embedding, train_loader, config, epoch)
        
        # Evaluate
        test_metrics = evaluate(solver, model, embedding, output_head, test_loader, config)
        
        # Log
        all_metrics = {**train_metrics, **test_metrics, "epoch": epoch}
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['train/loss']:.4f}, Train Acc: {train_metrics['train/accuracy']:.4f}")
        print(f"  Test Loss: {test_metrics['test/loss']:.4f}, Test Acc: {test_metrics['test/accuracy']:.4f}")
        print(f"  Avg Iters: {train_metrics['train/iters_free']:.1f}/{train_metrics['train/iters_nudged']:.1f}")
        print(f"  Time: {train_metrics['train/epoch_time']:.1f}s")
        
        if config.wandb:
            wandb.log(all_metrics)
        
        # Save best model (only if checkpointing enabled)
        if config.save_checkpoint and test_metrics["test/accuracy"] > best_acc:
            best_acc = test_metrics["test/accuracy"]
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                "config": config.to_dict(),
                "embedding": embedding.state_dict(),
                "model": model.state_dict(),
                "output_head": output_head.state_dict(),
                "test_acc": best_acc
            }, f"checkpoints/best_{config.dataset}.pt")
        elif test_metrics["test/accuracy"] > best_acc:
            best_acc = test_metrics["test/accuracy"]
    
    print("\n" + "="*70)
    print(f"Training complete! Best test accuracy: {best_acc:.4f}")
    print("="*70)
    
    if config.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
