"""Generate paper-ready figures from experimental results."""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


def plot_memory_scaling():
    """Plot memory usage: EqProp vs BP across model sizes."""
    # Data from memory_profile_results.txt
    d_models = [64, 128, 256, 512]
    eqprop_mem = [79.6, 194.7, 202.6, 349.6]
    bp_mem = [76.2, 187.7, 191.8, 312.2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute memory
    ax1.plot(d_models, eqprop_mem, 'o-', label='EqProp', linewidth=2, markersize=8)
    ax1.plot(d_models, bp_mem, 's-', label='BP', linewidth=2, markersize=8)
    ax1.set_xlabel('Model Dimension (d_model)', fontsize=12)
    ax1.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax1.set_title('Memory Usage vs Model Size', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Ratio
    ratios = [eq/bp for eq, bp in zip(eqprop_mem, bp_mem)]
    ax2.plot(d_models, ratios, 'o-', color='purple', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', label='Equal (1.0×)')
    ax2.axhline(y=0.5, color='green', linestyle='--', label='Target (0.5×)')
    ax2.set_xlabel('Model Dimension (d_model)', fontsize=12)
    ax2.set_ylabel('Memory Ratio (EqProp / BP)', fontsize=12)
    ax2.set_title('Relative Memory Efficiency', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig('figures/memory_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/memory_scaling.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/memory_scaling.{pdf,png}")


def plot_hyperparameter_heatmap():
    """Plot heatmap of test accuracy across hyperparameter sweep."""
    # From sweep results
    betas = [0.05, 0.10, 0.20]
    dampings = [0.80, 0.90, 0.95]
    lrs = [0.0005, 0.001, 0.002]
    
    # Results grid (beta × damping, average over lr)
    results = {
        (0.05, 0.80): [0.7944, 0.9017, 0.9206],
        (0.05, 0.90): [0.8880, 0.9051, 0.8819],
        (0.05, 0.95): [0.8690, 0.9211, 0.9181],
        (0.10, 0.80): [0.8688, 0.9149, 0.9039],
        (0.10, 0.90): [0.9055, 0.9146, 0.9259],
        (0.10, 0.95): [0.8870, 0.9190, 0.9151],
        (0.20, 0.80): [0.9144, 0.7835, 0.9390],
        (0.20, 0.90): [0.9128, 0.9281, 0.9256],
        (0.20, 0.95): [0.9051, 0.9040, 0.8686],
    }
    
    # Create grid
    grid = np.zeros((len(betas), len(dampings)))
    for i, beta in enumerate(betas):
        for j, damp in enumerate(dampings):
            grid[i, j] = max(results[(beta, damp)])  # Best lr for each combo
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap='RdYlGn', aspect='auto', vmin=0.75, vmax=0.95)
    
    # Labels
    ax.set_xticks(range(len(dampings)))
    ax.set_yticks(range(len(betas)))
    ax.set_xticklabels([f"{d:.2f}" for d in dampings])
    ax.set_yticklabels([f"{b:.2f}" for b in betas])
    ax.set_xlabel('Damping', fontsize=12)
    ax.set_ylabel('Beta (β)', fontsize=12)
    ax.set_title('Test Accuracy: Hyperparameter Sweep', fontsize=14)
    
    # Annotate cells
    for i in range(len(betas)):
        for j in range(len(dampings)):
            text = ax.text(j, i, f"{grid[i, j]:.3f}",
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Test Accuracy')
    plt.tight_layout()
    plt.savefig('figures/hyperparam_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/hyperparam_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/hyperparam_heatmap.{pdf,png}")


def plot_training_curves():
    """Plot training curves (placeholder - needs actual training logs)."""
    print("⚠ Training curves plot: needs wandb export or training logs")
    print("  Use: wandb.Api().run('project/run_id').history()")


if __name__ == "__main__":
    print("Generating publication figures...\n")
    plot_memory_scaling()
    plot_hyperparameter_heatmap()
    print("\n✓ All figures generated!")
