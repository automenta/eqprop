#!/usr/bin/env python3
"""
Generate SVG Plot for Beta Stability Analysis.
"""

import json
import os

def create_stability_plot_svg(data, output_path):
    results = data.get("results", [])
    betas = [r['beta'] for r in results]
    accs = [r['final_test_acc'] * 100 for r in results]
    
    # SVG Config
    width = 800
    height = 600
    margin = 60
    
    min_b, max_b = min(betas), max(betas)
    min_acc, max_acc = 85.0, 95.0 # Zoom in on the high accuracy region
    
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect width="100%" height="100%" fill="white"/>')
    
    origin_x = margin
    origin_y = height - margin
    plot_w = width - 2*margin
    plot_h = height - 2*margin
    
    # Scales
    def scale_x(b):
        return origin_x + (b - min_b) / (max_b - min_b) * plot_w
        
    def scale_y(a):
        return origin_y - (a - min_acc) / (max_acc - min_acc) * plot_h

    # Grid
    svg.append(f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x+plot_w}" y2="{origin_y}" stroke="black" stroke-width="2"/>')
    svg.append(f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x}" y2="{margin}" stroke="black" stroke-width="2"/>')
    
    # Points and Line
    points = []
    for b, a in zip(betas, accs):
        px = scale_x(b)
        py = scale_y(a)
        points.append(f"{px},{py}")
        
    svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#2c3e50" stroke-width="3"/>')
    
    for b, a in zip(betas, accs):
        px = scale_x(b)
        py = scale_y(a)
        svg.append(f'<circle cx="{px}" cy="{py}" r="5" fill="#e74c3c"/>')
        svg.append(f'<text x="{px}" y="{py-10}" text-anchor="middle" font-size="12">{a:.2f}%</text>')
        svg.append(f'<text x="{px}" y="{origin_y+20}" text-anchor="middle">{b}</text>')

    # Labels
    svg.append(f'<text x="{width/2}" y="{height-15}" text-anchor="middle">Beta Value</text>')
    svg.append(f'<text x="{20}" y="{height/2}" text-anchor="middle" transform="rotate(-90 20,{height/2})">Accuracy (%)</text>')
    svg.append(f'<text x="{width/2}" y="{30}" text-anchor="middle" font-size="20" font-weight="bold">Beta Stability Sweep</text>')

    svg.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg))
    print(f"SVG saved to {output_path}")

def main():
    json_path = "archive_v1/archive/experiments/beta_sweep/results.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path) as f:
        data = json.load(f)
        
    os.makedirs("results", exist_ok=True)
    create_stability_plot_svg(data, "results/beta_stability.svg")

if __name__ == "__main__":
    main()
