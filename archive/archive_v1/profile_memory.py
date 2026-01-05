"""Memory profiling script comparing EqProp vs BP training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
from src.trainer import EqPropTrainer
import gc


def profile_eqprop(config, x, y, model, embedding, output_head, solver):
    """Profile memory for EqProp training step."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    trainer = EqPropTrainer(model, solver, output_head, beta=config["beta"], lr=config["lr"])
    trainer.optimizer.add_param_group({'params': embedding.parameters()})
    
    # Warm-up
    x_emb = embedding(x).unsqueeze(0)
    _ = trainer.train_step(x_emb, y)
    
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    
    return peak_memory


def profile_bp(config, x, y, model, embedding, output_head, solver):
    """Profile memory for BP training step."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + list(model.parameters()) + list(output_head.parameters()),
        lr=config["lr"]
    )
    
    # Warm-up
    optimizer.zero_grad()
    x_emb = embedding(x).unsqueeze(0)
    h0 = torch.zeros_like(x_emb)
    h_fixed, _ = solver.solve(model, h0, x_emb)
    y_pred = output_head(h_fixed.mean(dim=0))
    loss = F.cross_entropy(y_pred, y)
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    
    return peak_memory


import argparse

def run_profiling():
    if not torch.cuda.is_available():
        print("CUDA not available. Memory profiling requires GPU.")
        return
    
    device = "cuda"
    
    # Parse args
    parser = argparse.ArgumentParser(description="Memory profiling")
    parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    parser.add_argument("--max-iters", type=int, default=100, help="Max iterations")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    
    args = parser.parse_args()
    
    # Load a batch of real data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Determine configs to run
    if args.d_model is not None:
        # Run specific config (e.g. for smoke test)
        configs = [
            {"d_model": args.d_model, "n_heads": 4, "d_ff": 4 * args.d_model, "batch_size": args.batch_size}
        ]
        max_iters = args.max_iters
    else:
        # Run standard suite
        configs = [
            {"d_model": 64, "n_heads": 4, "d_ff": 256, "batch_size": 128},
            {"d_model": 128, "n_heads": 4, "d_ff": 512, "batch_size": 128},
            {"d_model": 256, "n_heads": 8, "d_ff": 1024, "batch_size": 64},
            {"d_model": 512, "n_heads": 8, "d_ff": 2048, "batch_size": 32},
        ]
        max_iters = 50
    
    print("=" * 70)
    print("Memory Profiling: EqProp vs BP")
    print("=" * 70)
    print(f"{'d_model':>8} {'batch':>6} {'EqProp (MB)':>12} {'BP (MB)':>12} {'Ratio':>8}")
    print("-" * 70)
    
    results = []
    
    for cfg in configs:
        config = {
            **cfg,
            "max_iters": max_iters,
            "damping": 0.9,
            "beta": 0.1,
            "lr": 1e-3,
        }
        
        # Create models fresh each time
        torch.cuda.empty_cache()
        gc.collect()
        
        embedding = nn.Linear(784, config["d_model"]).to(device)
        model = LoopedTransformerBlock(
            config["d_model"], config["n_heads"], config["d_ff"],
            attention_type='linear', symmetric=False
        ).to(device)
        output_head = nn.Linear(config["d_model"], 10).to(device)
        solver = EquilibriumSolver(max_iters=config["max_iters"], tol=1e-5, damping=config["damping"])
        
        # Get batch
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        
        # Profile EqProp
        torch.cuda.empty_cache()
        eqprop_mem = profile_eqprop(config, x, y, model, embedding, output_head, solver)
        
        # Recreate models for fair comparison
        torch.cuda.empty_cache()
        gc.collect()
        
        embedding = nn.Linear(784, config["d_model"]).to(device)
        model = LoopedTransformerBlock(
            config["d_model"], config["n_heads"], config["d_ff"],
            attention_type='linear', symmetric=False
        ).to(device)
        output_head = nn.Linear(config["d_model"], 10).to(device)
        
        # Profile BP
        bp_mem = profile_bp(config, x, y, model, embedding, output_head, solver)
        
        # Avoid division by zero
        if bp_mem == 0:
            ratio = 1.0
        else:
            ratio = eqprop_mem / bp_mem
        
        print(f"{config['d_model']:>8} {config['batch_size']:>6} {eqprop_mem:>12.1f} {bp_mem:>12.1f} {ratio:>8.2f}x")
        
        results.append({
            "d_model": config["d_model"],
            "batch_size": config["batch_size"],
            "eqprop_mb": eqprop_mem,
            "bp_mb": bp_mem,
            "ratio": ratio
        })
    
    print("-" * 70)
    print("\nInterpretation:")
    print("- Ratio < 1.0: EqProp uses LESS memory than BP")
    print("- Ratio > 1.0: EqProp uses MORE memory than BP")
    print("- Target: EqProp should use <50% of BP memory for large models")
    
    return results


if __name__ == "__main__":
    run_profiling()
