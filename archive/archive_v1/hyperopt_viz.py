"""Visualization module for hyperopt experiment analysis.

Creates dimension-reduced plots of experiment configurations to reveal
trends and niches in the hyperparameter space.

Usage:
    from hyperopt_viz import visualize_trials
    visualize_trials(trials, output_path="hyperopt_viz.png")
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Try to import visualization libraries (optional dependencies)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def prepare_features(trials: List[Dict]) -> np.ndarray:
    """Convert trial configs to feature matrix for dimension reduction.
    
    Features:
    - beta (float, EqProp only)
    - damping (float)
    - max_iters (int)
    - d_model (int)
    - lr (float)
    - symmetric (bool)
    - algorithm (categorical: eqprop=0, bp=1)
    """
    features = []
    
    for trial in trials:
        cfg = trial.get("config", trial)
        
        # Extract features with defaults
        feat = [
            cfg.get("beta", 0.1),
            cfg.get("damping", 0.9),
            cfg.get("max_iters", 50) / 100,  # Normalize
            cfg.get("d_model", 128) / 256,  # Normalize
            np.log10(cfg.get("lr", 1e-3) + 1e-6),  # Log scale
            1.0 if cfg.get("symmetric", False) else 0.0,
            0.0 if cfg.get("algorithm", "eqprop") == "eqprop" else 1.0,
        ]
        features.append(feat)
    
    return np.array(features)


def get_performance_and_time(trials: List[Dict]) -> tuple:
    """Extract performance and time from trials."""
    performances = []
    times = []
    
    for trial in trials:
        perf = trial.get("performance", 0.0)
        time = trial.get("cost", {}).get("wall_time_seconds", 0.0)
        if isinstance(trial.get("cost"), dict):
            time = trial["cost"].get("wall_time_seconds", 0.0)
        elif hasattr(trial, "performance"):
            perf = trial.performance
            time = trial.cost.wall_time_seconds if hasattr(trial, "cost") else 0
        
        performances.append(perf)
        times.append(time)
    
    return np.array(performances), np.array(times)


def visualize_trials(
    trials: List[Dict],
    task_name: str = "Task",
    output_path: Optional[str] = None,
    show_legend: bool = True
) -> Optional[str]:
    """Create visualization of hyperopt trials using PCA.
    
    Args:
        trials: List of trial dictionaries with config and performance
        task_name: Name for the plot title
        output_path: Path to save figure (creates if None)
        show_legend: Whether to show legend
        
    Returns:
        Path to saved figure, or None if visualization failed
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        print("Warning: matplotlib and sklearn required for visualization")
        return None
    
    if len(trials) < 3:
        print("Warning: Need at least 3 trials for visualization")
        return None
    
    # Prepare features
    features = prepare_features(trials)
    performances, times = get_performance_and_time(trials)
    
    # Determine algorithm for each trial
    algorithms = [
        trial.get("config", trial).get("algorithm", "eqprop")
        for trial in trials
    ]
    
    # Standardize and reduce dimensions
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    n_components = min(2, len(trials) - 1, features.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(features_scaled)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Performance colored
    ax1 = axes[0]
    for algo, marker, label in [("eqprop", "o", "EqProp"), ("bp", "^", "BP")]:
        mask = np.array([a == algo for a in algorithms])
        if mask.any():
            scatter = ax1.scatter(
                coords[mask, 0], coords[mask, 1] if n_components > 1 else np.zeros(mask.sum()),
                c=performances[mask],
                cmap='viridis',
                marker=marker,
                s=100,
                alpha=0.7,
                label=label,
                edgecolors='black',
                linewidths=0.5
            )
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    if n_components > 1:
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    else:
        ax1.set_ylabel('(single component)')
    ax1.set_title(f'{task_name}: Performance (darker = worse)')
    if show_legend:
        ax1.legend(loc='upper right')
    plt.colorbar(scatter, ax=ax1, label='Performance')
    
    # Plot 2: Time colored
    ax2 = axes[1]
    for algo, marker, label in [("eqprop", "o", "EqProp"), ("bp", "^", "BP")]:
        mask = np.array([a == algo for a in algorithms])
        if mask.any():
            scatter = ax2.scatter(
                coords[mask, 0], coords[mask, 1] if n_components > 1 else np.zeros(mask.sum()),
                c=times[mask],
                cmap='plasma',
                marker=marker,
                s=100,
                alpha=0.7,
                label=label,
                edgecolors='black',
                linewidths=0.5
            )
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    if n_components > 1:
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    else:
        ax2.set_ylabel('(single component)')
    ax2.set_title(f'{task_name}: Time (lighter = faster)')
    if show_legend:
        ax2.legend(loc='upper right')
    plt.colorbar(scatter, ax=ax2, label='Time (s)')
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = f"hyperopt_viz_{task_name.lower().replace(' ', '_')}.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_path}")
    return output_path


def visualize_pareto_frontier(
    trials: List[Dict],
    task_name: str = "Task",
    output_path: Optional[str] = None
) -> Optional[str]:
    """Create Pareto frontier visualization (performance vs time).
    
    Args:
        trials: List of trial dictionaries
        task_name: Name for plot title
        output_path: Path to save figure
        
    Returns:
        Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib required for visualization")
        return None
    
    if len(trials) < 2:
        return None
    
    performances, times = get_performance_and_time(trials)
    algorithms = [
        trial.get("config", trial).get("algorithm", "eqprop")
        for trial in trials
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot by algorithm
    for algo, marker, color, label in [
        ("eqprop", "o", "#2ecc71", "EqProp"),
        ("bp", "^", "#e74c3c", "BP")
    ]:
        mask = np.array([a == algo for a in algorithms])
        if mask.any():
            ax.scatter(
                times[mask],
                performances[mask],
                marker=marker,
                c=color,
                s=100,
                alpha=0.7,
                label=label,
                edgecolors='black',
                linewidths=0.5
            )
    
    # Find and highlight Pareto frontier
    pareto_mask = np.zeros(len(trials), dtype=bool)
    for i in range(len(trials)):
        is_pareto = True
        for j in range(len(trials)):
            if i != j:
                # j dominates i if j is better in one dimension and not worse in other
                if performances[j] >= performances[i] and times[j] <= times[i]:
                    if performances[j] > performances[i] or times[j] < times[i]:
                        is_pareto = False
                        break
        pareto_mask[i] = is_pareto
    
    # Sort Pareto points by time for line plotting
    pareto_times = times[pareto_mask]
    pareto_perfs = performances[pareto_mask]
    sort_idx = np.argsort(pareto_times)
    
    if len(sort_idx) > 1:
        ax.plot(pareto_times[sort_idx], pareto_perfs[sort_idx], 
               'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Performance (accuracy/reward)', fontsize=12)
    ax.set_title(f'{task_name}: Performance vs Time Tradeoff', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = f"pareto_{task_name.lower().replace(' ', '_')}.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Pareto visualization to: {output_path}")
    return output_path


def generate_report_plots(db_path: str, output_dir: str = "plots") -> List[str]:
    """Generate all visualization plots from hyperopt database.
    
    Args:
        db_path: Path to hyperopt_results.json
        output_dir: Directory to save plots
        
    Returns:
        List of generated plot paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load database
    with open(db_path) as f:
        data = json.load(f)
    
    trials = data.get("trials", [])
    if not trials:
        print("No trials found in database")
        return []
    
    # Group by task
    tasks = {}
    for trial in trials:
        task = trial.get("task", "unknown")
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(trial)
    
    # Generate plots per task
    plot_paths = []
    
    for task_name, task_trials in tasks.items():
        print(f"\nGenerating plots for: {task_name} ({len(task_trials)} trials)")
        
        # PCA visualization
        pca_path = output_dir / f"pca_{task_name}.png"
        result = visualize_trials(task_trials, task_name, str(pca_path))
        if result:
            plot_paths.append(result)
        
        # Pareto frontier
        pareto_path = output_dir / f"pareto_{task_name}.png"
        result = visualize_pareto_frontier(task_trials, task_name, str(pareto_path))
        if result:
            plot_paths.append(result)
    
    return plot_paths


if __name__ == "__main__":
    import sys
    
    # Test with dummy data
    print("Testing visualization module...")
    
    dummy_trials = [
        {"config": {"algorithm": "eqprop", "beta": 0.1, "d_model": 64, "lr": 0.001}, 
         "performance": 0.85, "cost": {"wall_time_seconds": 10}},
        {"config": {"algorithm": "eqprop", "beta": 0.22, "d_model": 128, "lr": 0.002}, 
         "performance": 0.92, "cost": {"wall_time_seconds": 25}},
        {"config": {"algorithm": "bp", "d_model": 128, "lr": 0.001}, 
         "performance": 0.95, "cost": {"wall_time_seconds": 100}},
        {"config": {"algorithm": "bp", "d_model": 256, "lr": 0.002}, 
         "performance": 0.97, "cost": {"wall_time_seconds": 200}},
    ]
    
    result = visualize_trials(dummy_trials, "Test", "/tmp/test_viz.png")
    if result:
        print(f"✓ PCA visualization: {result}")
    
    result = visualize_pareto_frontier(dummy_trials, "Test", "/tmp/test_pareto.png")
    if result:
        print(f"✓ Pareto visualization: {result}")
    
    print("\nVisualization module ready!")
