#!/usr/bin/env python3
"""
TorEq Dynamic Observatory - Main Demo Script

Run the real-time visualization of network dynamics during EqProp training.

Usage:
    python scripts/run_observatory.py --dataset moons --layers 3
    python scripts/run_observatory.py --dataset mnist --layers 5 --headless
    
Controls:
    - ESC: Exit
    - UP/DOWN: Adjust Lipschitz σ constraint
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons, make_circles

from src.observatory import (
    SynapseHeatmap, 
    DynamicsCapture, 
    ObservatoryMetrics,
    ObservatoryRenderer,
    HeadlessRenderer,
    RendererConfig,
)
from src.observatory.heatmap import LayerState


class ObservableEqProp(nn.Module):
    """EqProp model with observatory hooks for visualization.
    
    A multi-layer version that exposes internal states for the heatmap.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, alpha: float = 0.5,
                 use_spectral_norm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.use_spectral_norm = use_spectral_norm
        
        # Layers
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            if use_spectral_norm:
                from torch.nn.utils.parametrizations import spectral_norm
                layer = spectral_norm(layer)
            self.layers.append(layer)
        
        self.head = nn.Linear(hidden_dim, output_dim)
        
        # Initialize for stability
        for layer in self.layers:
            # Access weight through parametrization if spectral norm applied
            weight = layer.weight if not use_spectral_norm else layer.parametrizations.weight.original
            nn.init.orthogonal_(weight)
            with torch.no_grad():
                weight.mul_(0.8)
        
        # Observatory hooks
        self.dynamics_capture = DynamicsCapture()
        self._prev_states: dict = {}
    
    def forward_step(self, h_states: dict, x: torch.Tensor) -> dict:
        """Single equilibrium step for all layers.
        
        Args:
            h_states: Dict of layer states {layer_idx: tensor}
            x: Input tensor
            
        Returns:
            Updated states dict
        """
        x_emb = self.embed(x)
        new_states = {}
        
        for i, layer in enumerate(self.layers):
            # Get input (previous layer output or embedding)
            if i == 0:
                h_in = x_emb
            else:
                h_in = h_states.get(i - 1, x_emb)
            
            h_current = h_states.get(i, torch.zeros_like(h_in))
            
            # Recurrent dynamics: h = (1-α)h + α*tanh(W*h_in + b)
            pre_act = layer(h_in)
            h_new = (1 - self.alpha) * h_current + self.alpha * torch.tanh(pre_act)
            new_states[i] = h_new
            
            # Record for visualization
            self.dynamics_capture.record_step(
                f"layer_{i}", 
                h_new, 
                h_current
            )
        
        return new_states
    
    def forward(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        """Forward pass to equilibrium."""
        batch_size = x.size(0)
        
        # Initialize states
        h_states = {
            i: torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for i in range(self.num_layers)
        }
        
        # Run to equilibrium
        for step in range(steps):
            h_states = self.forward_step(h_states, x)
        
        # Store free equilibrium for nudge computation
        self.dynamics_capture.record_free_equilibrium(
            {f"layer_{i}": h for i, h in h_states.items()}
        )
        
        # Output from last layer
        return self.head(h_states[self.num_layers - 1])
    
    def get_layer_states(self) -> dict:
        """Get current internal states for visualization."""
        if self.dynamics_capture.history:
            return self.dynamics_capture.history[-1]
        return {}


def create_dataset(name: str, n_samples: int = 500):
    """Create a toy dataset."""
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif name == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return torch.FloatTensor(X), torch.LongTensor(y)


def run_observatory(args):
    """Main observatory loop."""
    print(f"TorEq Dynamic Observatory")
    print(f"Dataset: {args.dataset}, Layers: {args.layers}")
    print("-" * 40)
    
    # Create dataset
    X, y = create_dataset(args.dataset, n_samples=args.samples)
    
    # Create model
    model = ObservableEqProp(
        input_dim=2,
        hidden_dim=args.hidden,
        output_dim=2,
        num_layers=args.layers,
        use_spectral_norm=True,
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Observatory components
    heatmap = SynapseHeatmap(
        lipschitz_threshold=1.0,
        velocity_scale=args.velocity_scale,
        nudge_scale=args.nudge_scale,
    )
    metrics = ObservatoryMetrics()
    
    # Renderer
    config = RendererConfig(
        window_width=max(400, 150 * args.layers + 300),
        window_height=400,
        layer_size=args.layer_size,
        record_frames=args.record,
    )
    
    if args.headless:
        renderer = HeadlessRenderer(config)
    else:
        renderer = ObservatoryRenderer(config)
    
    renderer.init()
    
    print("Starting training loop...")
    print("Press ESC to exit, UP/DOWN to adjust σ")
    
    try:
        for epoch in range(args.epochs):
            # Shuffle data
            perm = torch.randperm(len(X))
            X_shuffled, y_shuffled = X[perm], y[perm]
            
            for batch_start in range(0, len(X), args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(X))
                x_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Clear dynamics history
                model.dynamics_capture.clear()
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(x_batch, steps=args.steps)
                loss = criterion(logits, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Get accuracy
                acc = (logits.argmax(dim=1) == y_batch).float().mean().item()
                
                # Generate heatmaps from captured states
                layer_states = model.get_layer_states()
                layer_heatmaps = {}
                
                for name, state in layer_states.items():
                    rgb = heatmap.generate_rgb(state)
                    layer_heatmaps[name] = rgb
                
                # Update metrics
                step = batch_start // args.batch_size
                current_metrics = {
                    'Loss': loss.item(),
                    'Accuracy': acc * 100,
                    'Epoch': epoch,
                    'Step': step,
                }
                
                # Render frame
                renderer.render_frame(
                    layer_heatmaps=layer_heatmaps,
                    metrics=current_metrics,
                    epoch=epoch,
                    step=step,
                )
                
                if not renderer.running:
                    break
            
            if not renderer.running:
                break
            
            # Print epoch summary
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc*100:.1f}%")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        renderer.close()
        
        if args.headless and hasattr(renderer, 'save_gif'):
            renderer.save_gif(f"results/observatory_{args.dataset}.gif")
    
    print("Observatory closed.")


def main():
    parser = argparse.ArgumentParser(description="TorEq Dynamic Observatory")
    parser.add_argument('--dataset', type=str, default='moons',
                       choices=['moons', 'circles'],
                       help='Dataset to use')
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of hidden layers')
    parser.add_argument('--hidden', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of samples in dataset')
    parser.add_argument('--steps', type=int, default=20,
                       help='Equilibrium steps per forward pass')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--layer_size', type=int, default=96,
                       help='Pixel size for each layer heatmap')
    parser.add_argument('--velocity_scale', type=float, default=10.0,
                       help='Multiplier for velocity channel visibility')
    parser.add_argument('--nudge_scale', type=float, default=5.0,
                       help='Multiplier for nudge channel visibility')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display (record to GIF)')
    parser.add_argument('--record', action='store_true',
                       help='Record frames to PNG files')
    
    args = parser.parse_args()
    run_observatory(args)


if __name__ == '__main__':
    main()
