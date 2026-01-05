"""Analyze β stability sweep results and generate publication-ready figures."""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(results_file: Path) -> dict:
    """Load results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def plot_stability_boundary(results: list, output_dir: Path):
    """Plot β vs final accuracy with stability markers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stable = [r for r in results if r.get('stable', False)]
    collapsed = [r for r in results if r.get('collapsed', False)]
    
    if stable:
        stable_betas = [r['beta'] for r in stable]
        stable_accs = [r['final_test_acc'] for r in stable]
        ax.scatter(stable_betas, stable_accs, color='green', s=100, marker='o', 
                  label='Stable', zorder=3)
    
    if collapsed:
        collapsed_betas = [r['beta'] for r in collapsed]
        collapsed_accs = [r['final_test_acc'] for r in collapsed]
        ax.scatter(collapsed_betas, collapsed_accs, color='red', s=100, marker='x',
                  label='Collapsed', zorder=3)
    
    # Connect points with line
    all_betas = [r['beta'] for r in results]
    all_accs = [r['final_test_acc'] for r in results]
    ax.plot(all_betas, all_accs, 'k--', alpha=0.3, zorder=1)
    
    # Mark stability threshold
    if stable and collapsed:
        threshold = (min([r['beta'] for r in stable]) + max([r['beta'] for r in collapsed])) / 2
        ax.axvline(threshold, color='orange', linestyle=':', linewidth=2, 
                  label=f'Threshold β≈{threshold:.3f}', zorder=2)
    
    ax.set_xlabel('β (Nudge Strength)', fontsize=12)
    ax.set_ylabel('Final Test Accuracy', fontsize=12)
    ax.set_title('β Stability Boundary Characterization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'beta_stability_boundary.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'beta_stability_boundary.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'beta_stability_boundary.png'}")
    plt.close()


def plot_training_curves(results: list, output_dir: Path):
    """Plot training curves for each β."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color map from red (low β) to green (high β)
    betas = [r['beta'] for r in results]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(results)))
    
    for r, color in zip(results, colors):
        beta = r['beta']
        test_accs = r.get('test_accs', [])
        train_accs = r.get('train_accs', [])
        epochs = list(range(len(test_accs)))
        
        if test_accs:
            linestyle = '-' if r.get('stable', False) else ':'
            alpha = 1.0 if r.get('stable', False) else 0.4
            
            ax1.plot(epochs, test_accs, label=f'β={beta:.2f}', 
                    color=color, linestyle=linestyle, alpha=alpha)
            ax2.plot(epochs, train_accs, label=f'β={beta:.2f}',
                    color=color, linestyle=linestyle, alpha=alpha)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Test Accuracy', fontsize=11)
    ax1.set_title('Test Accuracy Progression', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Train Accuracy', fontsize=11)
    ax2.set_title('Train Accuracy Progression', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'beta_training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'beta_training_curves.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'beta_training_curves.png'}")
    plt.close()


def plot_peak_vs_final(results: list, output_dir: Path):
    """Plot peak accuracy vs final accuracy for each β."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stable = [r for r in results if r.get('stable', False)]
    collapsed = [r for r in results if r.get('collapsed', False)]
    
    if stable:
        betas = [r['beta'] for r in stable]
        peaks = [r['peak_test_acc'] for r in stable]
        finals = [r['final_test_acc'] for r in stable]
        
        ax.plot(betas, peaks, 'o-', color='green', label='Peak Accuracy', marker='o', markersize=8)
        ax.plot(betas, finals, 's-', color='blue', label='Final Accuracy', marker='s', markersize=8)
    
    if collapsed:
        betas = [r['beta'] for r in collapsed]
        peaks = [r['peak_test_acc'] for r in collapsed]
        finals = [r['final_test_acc'] for r in collapsed]
        
        ax.plot(betas, peaks, 'o:', color='red', alpha=0.5, label='Peak (Collapsed)', marker='o', markersize=8)
        ax.plot(betas, finals, 's:', color='orange', alpha=0.5, label='Final (Collapsed)', marker='s', markersize=8)
    
    ax.set_xlabel('β (Nudge Strength)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Peak vs Final Accuracy by β', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'beta_peak_vs_final.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'beta_peak_vs_final.pdf', bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'beta_peak_vs_final.png'}")
    plt.close()


def generate_summary_table(results: list, output_dir: Path):
    """Generate markdown summary table."""
    table_file = output_dir / 'summary_table.md'
    
    with open(table_file, 'w') as f:
        f.write("# β Stability Sweep Summary\n\n")
        f.write("| β | Status | Epochs | Final Train | Final Test | Peak Test | Collapse Epoch |\n")
        f.write("|---|--------|--------|-------------|------------|-----------|----------------|\n")
        
        for r in sorted(results, key=lambda x: x['beta']):
            status = "✅ Stable" if r.get('stable') else ("❌ Collapse" if r.get('collapsed') else "⚠️ Error")
            epochs = r.get('num_epochs', 0)
            final_train = r.get('final_train_acc', 0.0)
            final_test = r.get('final_test_acc', 0.0)
            peak_test = r.get('peak_test_acc', 0.0)
            collapse = r.get('collapse_epoch', '-')
            
            f.write(f"| {r['beta']:.2f} | {status} | {epochs} | {final_train:.4f} | "
                   f"{final_test:.4f} | {peak_test:.4f} | {collapse} |\n")
        
        # Add analysis
        f.write("\n## Analysis\n\n")
        
        stable_betas = [r['beta'] for r in results if r.get('stable', False)]
        collapsed_betas = [r['beta'] for r in results if r.get('collapsed', False)]
        
        if stable_betas and collapsed_betas:
            min_stable = min(stable_betas)
            max_collapsed = max(collapsed_betas)
            threshold = (min_stable + max_collapsed) / 2
            
            f.write(f"**Stability Threshold**: β ≈ {threshold:.3f}\n\n")
            f.write(f"- Maximum collapsed β: {max_collapsed:.2f}\n")
            f.write(f"- Minimum stable β: {min_stable:.2f}\n\n")
        
        if stable_betas:
            optimal = max([r for r in results if r.get('stable', False)], 
                         key=lambda x: x.get('peak_test_acc', 0.0))
            f.write(f"**Optimal β**: {optimal['beta']:.2f}\n\n")
            f.write(f"- Peak accuracy: {optimal['peak_test_acc']:.4f}\n")
            f.write(f"- Final accuracy: {optimal['final_test_acc']:.4f}\n\n")
        
        f.write("## Theory-Practice Gap\n\n")
        f.write("**Theory**: EqProp gradient equivalence theorem suggests β→0 for best gradient approximation.\n\n")
        f.write(f"**Practice**: β ≥ {min_stable:.2f} required for stable training on transformers.\n\n")
        f.write("This discrepancy suggests:\n")
        f.write("1. Finite learning rate effects dominate at small β\n")
        f.write("2. Loss landscape curvature requires minimum nudge strength\n")
        f.write("3. Discrete optimization dynamics differ from continuous theory\n")
    
    print(f"✅ Saved: {table_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze β sweep results")
    parser.add_argument("--results", type=str, required=True, help="Path to results.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same as results)")
    args = parser.parse_args()
    
    results_file = Path(args.results)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    output_dir = Path(args.output_dir) if args.output_dir else results_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ANALYZING β STABILITY SWEEP")
    print(f"{'='*70}")
    print(f"Results: {results_file}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load results
    data = load_results(results_file)
    results = data['results']
    
    print(f"Loaded {len(results)} experimental runs\n")
    
    # Generate plots
    print("📊 Generating plots...")
    plot_stability_boundary(results, output_dir)
    plot_training_curves(results, output_dir)
    plot_peak_vs_final(results, output_dir)
    
    # Generate summary
    print("\n📝 Generating summary table...")
    generate_summary_table(results, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Outputs in: {output_dir}")
    print(f"  - beta_stability_boundary.png/pdf")
    print(f"  - beta_training_curves.png/pdf")
    print(f"  - beta_peak_vs_final.png/pdf")
    print(f"  - summary_table.md")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
