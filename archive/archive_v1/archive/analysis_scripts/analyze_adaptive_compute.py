"""Analyze adaptive compute dynamics in TorEqProp models.

This script loads a trained model and analyzes:
- Per-sample iteration counts to convergence
- Correlation between iterations and prediction confidence/margin
- Iterations per digit class
- Early exit potential for compute savings
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")
import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver


def load_model_and_data(checkpoint_path: str, batch_size: int = 128):
    """Load trained model and test dataset.
    
    Returns:
        embedding, model, output_head, test_loader, config
    """
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    embedding = nn.Linear(784, config['d_model']).to(device)
    model = LoopedTransformerBlock(
        config['d_model'],
        config['n_heads'],
        config['d_ff'],
        dropout=config.get('dropout', 0.0),
        attention_type=config.get('attention_type', 'linear'),
        symmetric=config.get('symmetric', False)
    ).to(device)
    output_head = nn.Linear(config['d_model'], 10).to(device)
    
    # Load weights (handle torch.compile wrapped models)
    def strip_compile_prefix(state_dict):
        """Remove _orig_mod. prefix from compiled model state dicts."""
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    embedding.load_state_dict(strip_compile_prefix(checkpoint['embedding']))
    model.load_state_dict(strip_compile_prefix(checkpoint['model']))
    output_head.load_state_dict(strip_compile_prefix(checkpoint['output_head']))
    
    # Set to eval mode
    embedding.eval()
    model.eval()
    output_head.eval()
    
    return embedding, model, output_head, test_loader, config, device


def analyze_sample_dynamics(embedding, model, output_head, solver, data, target, device):
    """Analyze equilibrium dynamics for a single batch.
    
    Returns:
        Dictionary with per-sample metrics
    """
    data, target = data.to(device), target.to(device)
    batch_size = data.size(0)
    
    # Embed input
    x_emb = embedding(data).unsqueeze(0)  # [1, batch, d_model]
    
    # Track iterations per sample
    h = torch.zeros_like(x_emb)
    iters_per_sample = []
    residuals_over_time = []
    
    with torch.no_grad():
        # Run equilibrium with detailed tracking
        for t in range(solver.max_iters):
            h_new = (1 - solver.damping) * h + solver.damping * model(h, x_emb)
            residual = (h_new - h).norm(dim=-1).squeeze(0)  # [batch]
            residuals_over_time.append(residual.cpu().numpy())
            
            # Check convergence per sample
            converged = residual < solver.tol
            h = h_new
            
            if converged.all():
                iters_per_sample = [t + 1] * batch_size
                break
        else:
            # Didn't converge, assign max_iters
            iters_per_sample = [solver.max_iters] * batch_size
        
        # Final predictions
        h_final = h
        y_pred = output_head(h_final.mean(dim=0))  # [batch, 10]
        probs = torch.softmax(y_pred, dim=-1)
        
        pred_classes = y_pred.argmax(dim=-1)
        correct = (pred_classes == target)
        
        # Confidence metrics
        max_probs, _ = probs.max(dim=-1)
        
        # Margin: difference between top-2 probabilities
        top2_probs, _ = torch.topk(probs, k=2, dim=-1)
        margins = (top2_probs[:, 0] - top2_probs[:, 1])
    
    results = {
        'iterations': iters_per_sample if isinstance(iters_per_sample, list) else [iters_per_sample],
        'predicted': pred_classes.cpu().numpy(),
        'target': target.cpu().numpy(),
        'correct': correct.cpu().numpy(),
        'confidence': max_probs.cpu().numpy(),
        'margin': margins.cpu().numpy(),
        'residuals_over_time': np.array(residuals_over_time)  # [max_iters, batch]
    }
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze adaptive compute dynamics")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='adaptive_compute_results.json',
                        help='Output JSON file')
    parser.add_argument('--plot-dir', type=str, default='plots',
                        help='Directory for plots')
    parser.add_argument('--max-iters', type=int, default=50,
                        help='Max iterations for equilibrium')
    parser.add_argument('--tol', type=float, default=1e-5,
                        help='Convergence tolerance')
    parser.add_argument('--damping', type=float, default=0.8,
                        help='Damping factor')
    
    args = parser.parse_args()
    
    # Load model and data
    print("Loading model and data...")
    embedding, model, output_head, test_loader, config, device = load_model_and_data(args.checkpoint)
    
    # Create solver
    solver = EquilibriumSolver(
        max_iters=args.max_iters,
        tol=args.tol,
        damping=args.damping
    )
    
    # Collect results
    print("Analyzing equilibrium dynamics...")
    all_results = defaultdict(list)
    
    for batch_idx, (data, target) in enumerate(test_loader):
        results = analyze_sample_dynamics(embedding, model, output_head, solver, data, target, device)
        
        for key in results:
            if key == 'residuals_over_time':
                continue  # Skip for now, too large
            all_results[key].extend(results[key])
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Convert to numpy
    for key in all_results:
        all_results[key] = np.array(all_results[key])
    
    # Compute statistics
    print("\n" + "="*70)
    print("Adaptive Compute Analysis Results")
    print("="*70)
    
    # Overall stats
    iterations = all_results['iterations']
    print(f"\nIteration Statistics:")
    print(f"  Mean: {iterations.mean():.2f}")
    print(f"  Std: {iterations.std():.2f}")
    print(f"  Min: {iterations.min()}, Max: {iterations.max()}")
    
    # Correlation with confidence
    confidence = all_results['confidence']
    margin = all_results['margin']
    
    corr_conf, p_conf = pearsonr(iterations, confidence)
    corr_margin, p_margin = pearsonr(iterations, margin)
    
    print(f"\nCorrelations:")
    print(f"  Iterations vs Confidence: r={corr_conf:.4f}, p={p_conf:.4e}")
    print(f"  Iterations vs Margin: r={corr_margin:.4f}, p={p_margin:.4e}")
    
    # Per-class analysis
    print(f"\nIterations per Digit Class:")
    for digit in range(10):
        mask = all_results['target'] == digit
        if mask.sum() > 0:
            iters_digit = iterations[mask]
            print(f"  Digit {digit}: {iters_digit.mean():.2f} ± {iters_digit.std():.2f}")
    
    # Correct vs incorrect
    correct_iters = iterations[all_results['correct']]
    incorrect_iters = iterations[~all_results['correct']]
    print(f"\nCorrect vs Incorrect Predictions:")
    print(f"  Correct: {correct_iters.mean():.2f} ± {correct_iters.std():.2f}")
    print(f"  Incorrect: {incorrect_iters.mean():.2f} ± {incorrect_iters.std():.2f}")
    
    # Early exit analysis
    print(f"\nEarly Exit Potential:")
    accuracy = all_results['correct'].mean()
    print(f"  Baseline accuracy: {accuracy:.4f}")
    
    for budget in [10, 20, 30, 40]:
        early_exit_mask = iterations <= budget
        if early_exit_mask.sum() > 0:
            acc_early = all_results['correct'][early_exit_mask].mean()
            coverage = early_exit_mask.mean()
            print(f"  Budget {budget} iters: {acc_early:.4f} acc, {coverage:.2%} coverage")
    
    # Save results
    output_path = Path(args.output)
    output_data = {
        'statistics': {
            'mean_iterations': float(iterations.mean()),
            'std_iterations': float(iterations.std()),
            'min_iterations': int(iterations.min()),
            'max_iterations': int(iterations.max()),
            'correlation_confidence': float(corr_conf),
            'correlation_margin': float(corr_margin),
            'p_value_confidence': float(p_conf),
            'p_value_margin': float(p_margin)
        },
        'per_class': {
            int(digit): {
                'mean': float(iterations[all_results['target'] == digit].mean()),
                'std': float(iterations[all_results['target'] == digit].std())
            }
            for digit in range(10)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Create plots
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        print("="*70)
        return
    
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(exist_ok=True)
    
    # Plot 1: Iteration distribution
    plt.figure(figsize=(10, 6))
    plt.hist(iterations, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Iterations to Convergence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Iterations to Equilibrium')
    plt.savefig(plot_dir / 'iteration_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Iterations vs Confidence
    plt.figure(figsize=(10, 6))
    plt.scatter(confidence, iterations, alpha=0.3, s=10)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Iterations to Convergence')
    plt.title(f'Iterations vs Confidence (r={corr_conf:.3f})')
    plt.savefig(plot_dir / 'iterations_vs_confidence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Per-class iterations
    plt.figure(figsize=(10, 6))
    class_iters = [iterations[all_results['target'] == d] for d in range(10)]
    plt.boxplot(class_iters, labels=range(10))
    plt.xlabel('Digit Class')
    plt.ylabel('Iterations to Convergence')
    plt.title('Iterations per Digit Class')
    plt.savefig(plot_dir / 'iterations_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {plot_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
