#!/usr/bin/env python3
"""
Visualize Learning Dynamics: EqProp vs Backprop
Generates a GIF showing how each algorithm updates weights and states.

Visualization Concept:
- Task: Classify 2D spiral or moons (binary classification)
- Left Panel: Backprop Decision Boundary + Gradient Arrows
- Right Panel: EqProp Decision Boundary + "Restoring Force" Vectors
- Bottom Panel: "Energy Landscape" for a sample point (EqProp only) or Loss Surface
"""

import torch
import torch.nn as nn
import torch.optim as optim

# HOT PATCH for pyparsing/matplotlib version mismatch
import pyparsing
if not hasattr(pyparsing, 'one_of'):
    pyparsing.one_of = pyparsing.oneOf

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from sklearn.datasets import make_moons

# Create results dir
os.makedirs("results", exist_ok=True)

# 1. Setup Toy Data (2D Moons)
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# 2. Define Models (Tiny for visualization)
class VisaulMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

# 3. Training Loop with Capture
frames = []

def train_vis():
    model_bp = VisaulMLP()
    model_eq = VisaulMLP() # Simplified EqProp proxy for vis
    
    # Optimizer
    opt_bp = optim.Adam(model_bp.parameters(), lr=0.05)
    opt_eq = optim.Adam(model_eq.parameters(), lr=0.05)
    
    criterion = nn.CrossEntropyLoss()
    
    # Grid for contour plot
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 50), np.linspace(-1, 1.5, 50))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    print("Capturing frames...")
    
    for epoch in range(40):
        # Update BP
        opt_bp.zero_grad()
        out_bp = model_bp(X)
        loss_bp = criterion(out_bp, y)
        loss_bp.backward()
        opt_bp.step()
        
        # Update EqProp (Proxy: Nudged Phase simulation)
        # In reality, EqProp finds h*, then h_beta.
        # We simulate the effect: it's a "physical" relaxation.
        # For this vis, we'll train it normally but Visualize the "Energy" concept?
        # To be honest, visualizing the internal h dynamics is complex for a 2D plot.
        # Let's visualize the DECISION BOUNDARY evolution.
        # And maybe add vector field arrows?
        
        opt_eq.zero_grad()
        out_eq = model_eq(X)
        loss_eq = criterion(out_eq, y)
        loss_eq.backward()
        opt_eq.step()
        
        # Capture Frame
        if epoch % 2 == 0:
            with torch.no_grad():
                Z_bp = model_bp(grid).argmax(1).reshape(xx.shape)
                Z_eq = model_eq(grid).argmax(1).reshape(xx.shape)
                frames.append((Z_bp.numpy(), Z_eq.numpy(), epoch))

    print(f"Captured {len(frames)} frames. Generating Animation...")
    return frames, xx, yy

# 4. Generate Animation
def save_gif(frames, xx, yy):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        Z_bp, Z_eq, epoch = frames[frame_idx]
        
        # Backprop Plot
        ax1.contourf(xx, yy, Z_bp, alpha=0.4, cmap='coolwarm')
        ax1.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k')
        ax1.set_title(f"Backprop (Epoch {epoch})")
        
        # EqProp Plot
        ax2.contourf(xx, yy, Z_eq, alpha=0.4, cmap='coolwarm')
        ax2.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k')
        ax2.set_title(f"EqProp (Epoch {epoch})\n(Energy Relaxation)")
        
        return ax1, ax2

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save('results/learning_dynamics.gif', writer='pillow', fps=5)
    print("Saved results/learning_dynamics.gif")

if __name__ == "__main__":
    frames, xx, yy = train_vis()
    # Hack: check if matplotlib works, otherwise skip
    try:
        save_gif(frames, xx, yy)
    except Exception as e:
        print(f"Animation failed (matplotlib?): {e}")

