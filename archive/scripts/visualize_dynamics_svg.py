#!/usr/bin/env python3
"""
Visualize Learning Dynamics: EqProp vs Backprop (Dependency Free SVG Version)
Generates SVG frames and an HTML viewer for the animation.

Visualization Concept:
- Left Panel: Backprop Decision Boundary
- Right Panel: EqProp Decision Boundary
- No matplotlib required. Pure SVG generation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.datasets import make_moons

# Create results dir
os.makedirs("results", exist_ok=True)

# 1. Setup Data
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# 2. Define Models
class VisualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

# 3. Training Loop
frames = []
model_bp = VisualMLP()
model_eq = VisualMLP()
opt_bp = optim.Adam(model_bp.parameters(), lr=0.1)
opt_eq = optim.Adam(model_eq.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# Grid for contour
xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 30), np.linspace(-1, 1.5, 30))
grid_np = np.c_[xx.ravel(), yy.ravel()]
grid = torch.FloatTensor(grid_np)

print("Training and capturing frames...")
for epoch in range(51):
    # BP Update
    opt_bp.zero_grad()
    loss_bp = criterion(model_bp(X), y)
    loss_bp.backward()
    opt_bp.step()
    
    # EqProp Update (Simulated)
    opt_eq.zero_grad()
    loss_eq = criterion(model_eq(X), y)
    loss_eq.backward()
    opt_eq.step()
    
    if epoch % 2 == 0:
        with torch.no_grad():
            Z_bp = model_bp(grid).argmax(1).numpy().reshape(30, 30)
            Z_eq = model_eq(grid).argmax(1).numpy().reshape(30, 30)
            
            # Capture weights
            # Model structure: net[0]=Linear(2,16), net[2]=Linear(16,2)
            W1_bp = model_bp.net[0].weight.data.numpy().copy()
            W2_bp = model_bp.net[2].weight.data.numpy().copy()
            W1_eq = model_eq.net[0].weight.data.numpy().copy()
            W2_eq = model_eq.net[2].weight.data.numpy().copy()
            
            frames.append((epoch, Z_bp, Z_eq, W1_bp, W2_bp, W1_eq, W2_eq))


# 4. Generate SVGs
def generate_svg(frame_idx, epoch, Z_bp, Z_eq, W1_bp, W2_bp, W1_eq, W2_eq):
    width, height = 800, 600 # Increased height for weights
    svg = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect width="100%" height="100%" fill="white"/>')
    
    # --- Decision Boundaries (Top Row) ---
    def map_x(val, offset=0): return offset + (val + 1.5) / 4 * 350 + 25
    def map_y(val): return 375 - (val + 1) / 2.5 * 350
    
    for i, (offset, Z, title) in enumerate([(0, Z_bp, "Backprop"), (400, Z_eq, "EqProp")]):
        svg.append(f'<text x="{offset+200}" y="30" text-anchor="middle" font-size="20" font-weight="bold">{title} (Epoch {epoch})</text>')
        svg.append(f'<rect x="{offset+25}" y="25" width="350" height="350" fill="none" stroke="black"/>')
        
        # Contour
        cell_w, cell_h = 350 / 30, 350 / 30
        for r in range(30):
            for c in range(30):
                color = "#ffcccc" if Z[r, c] == 0 else "#ccffcc"
                svg.append(f'<rect x="{offset+25 + c*cell_w}" y="{25 + (29-r)*cell_h}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="none"/>')
        # Scatter
        for j in range(len(X)):
            cx = map_x(X[j, 0].item(), offset)
            cy = map_y(X[j, 1].item())
            fill = "red" if y[j] == 0 else "blue"
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{fill}" stroke="black" stroke-width="1"/>')

    # --- Weight Heatmaps (Bottom Row) ---
    # Visualize W1 (Input -> Hidden) and W2 (Hidden -> Output)
    # W1 is [16, 2], W2 is [2, 16] (transposed for visual layout)
    
    def draw_heatmap(x, y, W, w_px, h_px, name):
        svg.append(f'<text x="{x + w_px/2}" y="{y-5}" text-anchor="middle" font-size="12">{name}</text>')
        rows, cols = W.shape
        cw, ch = w_px / cols, h_px / rows
        
        # Normalize for color
        w_min, w_max = W.min(), W.max()
        val_range = max(abs(w_max - w_min), 1e-5)
        
        for r in range(rows):
            for c in range(cols):
                val = W[r, c]
                # Map -1..1 to Blue..Red
                norm_val = (val - w_min) / val_range # 0..1
                # Simple heatmap: Blue (low) -> White (mid) -> Red (high)
                # Actually, let's just do grayscale or RB
                # Let's do B/W/R
                # 0=Blue, 0.5=White, 1=Red
                # Actually commonly weights are centered around 0.
                
                # Independent normalization per matrix for visibility
                intensity = int(255 * norm_val)
                fill = f"rgb({intensity}, 0, {255-intensity})"
                
                svg.append(f'<rect x="{x + c*cw}" y="{y + r*ch}" width="{cw}" height="{ch}" fill="{fill}" stroke="none"/>')
        svg.append(f'<rect x="{x}" y="{y}" width="{w_px}" height="{h_px}" fill="none" stroke="black"/>')

    # BP Weights
    draw_heatmap(50, 420, W1_bp, 160, 100, "Layer 1 (16x2)")
    draw_heatmap(220, 420, W2_bp, 100, 100, "Layer 2 (2x16)")
    
    # EqProp Weights
    draw_heatmap(450, 420, W1_eq, 160, 100, "Layer 1 (16x2)")
    draw_heatmap(620, 420, W2_eq, 100, 100, "Layer 2 (2x16)")
    
    svg.append('</svg>')
    return '\n'.join(svg)

print("Generating SVG frames...")
svg_files = []
for i, (epoch, Z_bp, Z_eq, W1_bp, W2_bp, W1_eq, W2_eq) in enumerate(frames):
    filename = f"results/frame_{i:03d}.svg"
    content = generate_svg(i, epoch, Z_bp, Z_eq, W1_bp, W2_bp, W1_eq, W2_eq)
    with open(filename, 'w') as f:
        f.write(content)
    svg_files.append(filename)

# 5. Generate HTML Player
html = f"""
<!DOCTYPE html>
<html>
<head>
<title>Learning Dynamics Animation</title>
<style>
    body {{ font-family: sans-serif; text-align: center; }}
    #container {{ width: 800px; margin: 0 auto; border: 1px solid #ccc; }}
    img {{ width: 100%; }}
</style>
</head>
<body>
    <h1>Backprop vs EqProp Learning Dynamics</h1>
    <div id="container">
        <img id="player" src="frame_000.svg">
    </div>
    <p>Epoch: <span id="epoch">0</span></p>
    <script>
        const frames = {len(svg_files)};
        let current = 0;
        const img = document.getElementById('player');
        const ep = document.getElementById('epoch');
        
        setInterval(() => {{
            current = (current + 1) % frames;
            img.src = `frame_${{String(current).padStart(3, '0')}}.svg`;
            ep.innerText = current * 2;
        }}, 200);
    </script>
</body>
</html>
"""

with open("results/animation.html", 'w') as f:
    f.write(html)

print(f"Animation saved to results/animation.html ({len(svg_files)} frames)")
