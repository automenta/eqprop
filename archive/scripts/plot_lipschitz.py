#!/usr/bin/env python3
"""
Plot Lipschitz analysis results from JSON.
Generates a bar chart comparing Untrained vs Trained (No SN) vs Trained (SN).
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_path = "/tmp/lipschitz_analysis.json"
    if not os.path.exists(json_path):
        # Try local results dir
        json_path = "results/lipschitz_analysis.json"
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path) as f:
        data = json.load(f)
    
    # Data extraction
    models = [item['model'] for item in data]
    l_untrained = [0.69, 0.70, 0.54] # Hardcoded reference or from file? File has L_without_sn (trained).
    # Wait, the file result has L_without_sn (TRAINED) and L_with_sn (TRAINED).
    # It does NOT have untrained L.
    # The paper template had untrained reference values. I'll use those as baseline or omit.
    # Actually, let's just plot With SN vs Without SN.
    
    l_no_sn = [item['L_without_sn'] for item in data]
    l_sn = [item['L_with_sn'] for item in data]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, l_no_sn, width, label='Without SN', color='#ff9999')
    rects2 = ax.bar(x + width/2, l_sn, width, label='With SN', color='#66b3ff')
    
    ax.set_ylabel('Lipschitz Constant (L)')
    ax.set_title('Impact of Spectral Normalization on Stability')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add a line at L=1
    ax.axhline(y=1.0, color='r', linestyle='--', label='Stability Bound (L=1)')
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    plt.tight_layout()
    output_path = "results/spectral_norm_stability.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
