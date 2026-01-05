#!/usr/bin/env python3
"""
Generate SVG Comparison for Lipschitz constants.
Standalone, no heavy dependencies.
"""

import json
import os

def create_bar_chart_svg(data, output_path):
    # Setup data
    models = [d['model'] for d in data]
    l_no_sn = [d['L_without_sn'] for d in data]
    l_sn = [d['L_with_sn'] for d in data]
    
    # SVG Config
    width = 800
    height = 600
    margin = 50
    bar_width = 60
    group_gap = 100
    
    max_val = max(max(l_no_sn), max(l_sn))
    # Cap meaningful max to avoid outlier distortion (EqProp ~20.7)
    # Let's use log scale or cut off? Or just scale linearly but let 20 go off chart?
    # Better: Break the axis or just linear scale. 20 is huge.
    # If we scale to 21, the 0.5 bars will be tiny.
    # Let's cap at 2.0 for visualization and label the overflow.
    display_max = 2.0
    scale = (height - 2 * margin) / display_max
    
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect width="100%" height="100%" fill="white"/>')
    
    # Axes
    origin_y = height - margin
    origin_x = margin
    svg.append(f'<line x1="{origin_x}" y1="{origin_y}" x2="{width-margin}" y2="{origin_y}" stroke="black" stroke-width="2"/>') # X
    svg.append(f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x}" y2="{margin}" stroke="black" stroke-width="2"/>') # Y
    
    # Threshold line at 1.0
    thresh_y = origin_y - (1.0 * scale)
    svg.append(f'<line x1="{origin_x}" y1="{thresh_y}" x2="{width-margin}" y2="{thresh_y}" stroke="red" stroke-dasharray="5,5" stroke-width="2"/>')
    svg.append(f'<text x="{width-margin-10}" y="{thresh_y-5}" fill="red" text-anchor="end">Stability Limit (L=1)</text>')
    
    # Bars
    colors = ["#ff9999", "#66b3ff"] # No SN, With SN
    
    for i, model in enumerate(models):
        center_x = origin_x + margin + i * (2*bar_width + group_gap)
        
        # No SN Bar
        val1 = l_no_sn[i]
        bar_h1 = min(val1, display_max) * scale
        y1 = origin_y - bar_h1
        svg.append(f'<rect x="{center_x}" y="{y1}" width="{bar_width}" height="{bar_h1}" fill="{colors[0]}"/>')
        label1 = f"{val1:.2f}"
        svg.append(f'<text x="{center_x + bar_width/2}" y="{y1-5}" text-anchor="middle" font-size="12">{label1}</text>')
        
        # With SN Bar
        val2 = l_sn[i]
        bar_h2 = min(val2, display_max) * scale
        y2 = origin_y - bar_h2
        svg.append(f'<rect x="{center_x + bar_width}" y="{y2}" width="{bar_width}" height="{bar_h2}" fill="{colors[1]}"/>')
        label2 = f"{val2:.2f}"
        svg.append(f'<text x="{center_x + 1.5*bar_width}" y="{y2-5}" text-anchor="middle" font-size="12">{label2}</text>')
        
        # Group Label
        svg.append(f'<text x="{center_x + bar_width}" y="{origin_y + 20}" text-anchor="middle" font-weight="bold">{model}</text>')

    # Legend
    svg.append(f'<rect x="{width-200}" y="{margin}" width="20" height="20" fill="{colors[0]}"/>')
    svg.append(f'<text x="{width-170}" y="{margin+15}" alignment-baseline="middle">Without SN</text>')
    svg.append(f'<rect x="{width-200}" y="{margin+30}" width="20" height="20" fill="{colors[1]}"/>')
    svg.append(f'<text x="{width-170}" y="{margin+45}" alignment-baseline="middle">With SN</text>')
    
    # Title
    svg.append(f'<text x="{width/2}" y="{margin/2}" text-anchor="middle" font-size="20" font-weight="bold">Spectral Normalization & Stability</text>')
    
    svg.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg))
    print(f"SVG saved to {output_path}")

def main():
    json_path = "/tmp/lipschitz_analysis.json"
    if not os.path.exists(json_path):
        json_path = "results/lipschitz_analysis.json"
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path) as f:
        data = json.load(f)
        
    os.makedirs("results", exist_ok=True)
    create_bar_chart_svg(data, "results/spectral_norm_stability.svg")

if __name__ == "__main__":
    main()
