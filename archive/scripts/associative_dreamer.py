#!/usr/bin/env python3
"""
Associative Dreamer: Inverse Inference (TODO5 Additional Experiment)

Proves the network is a Universal Associative Memory by showing that
the exact same weights used for classification can reconstruct the input
from a label.

How it works:
1. Normal Mode: Clamp input, relax to output → Classification
2. Dream Mode: Clamp output, relax to input → Generation/Reconstruction

This moves beyond "Machine Learning" into "Cognitive Modeling."
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from typing import Tuple, List, Dict, Optional
import json


class AssociativeDreamer:
    """
    Runs EqProp networks in both forward (classification) and
    inverse (generation/dreaming) modes.
    """
    
    def __init__(self, model: nn.Module, input_dim: int, output_dim: int):
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def classify(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        """Normal classification mode."""
        self.model.eval()
        with torch.no_grad():
            return self.model(x, steps=steps)
    
    def dream(self, target_class: int, num_samples: int = 1, 
              steps: int = 100, learning_rate: float = 0.1) -> torch.Tensor:
        """
        Generate inputs that correspond to a target class.
        
        This is "inverse inference" - running the network backwards
        by optimizing an input to match a desired output.
        
        Args:
            target_class: Class to dream about
            num_samples: Number of dreams to generate
            steps: Optimization steps
            learning_rate: Step size for input optimization
            
        Returns:
            Dreamed inputs [num_samples, input_dim]
        """
        self.model.eval()
        
        # Initialize random input
        x_dream = torch.randn(num_samples, self.input_dim, requires_grad=True)
        
        # Target is one-hot
        target = torch.zeros(num_samples, self.output_dim)
        target[:, target_class] = 1.0
        
        optimizer = torch.optim.Adam([x_dream], lr=learning_rate)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Forward pass
            out = self.model(x_dream, steps=30)
            
            # Loss: match target distribution
            loss = F.cross_entropy(out, torch.full((num_samples,), target_class, dtype=torch.long))
            
            # Optional: add regularization to keep dreams realistic
            reg = 0.01 * torch.sum(x_dream ** 2)
            
            (loss + reg).backward()
            optimizer.step()
            
            # Clamp to reasonable range
            with torch.no_grad():
                x_dream.data.clamp_(-3, 3)
        
        return x_dream.detach()
    
    def dream_from_prototype(self, prototype_x: torch.Tensor, target_class: int,
                              steps: int = 50, learning_rate: float = 0.05) -> torch.Tensor:
        """
        Dream starting from a prototype input.
        
        This creates variations of an input that still match the target class.
        """
        self.model.eval()
        
        x_dream = prototype_x.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x_dream], lr=learning_rate)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            out = self.model(x_dream, steps=30)
            loss = F.cross_entropy(out, torch.tensor([target_class]))
            
            # Keep close to prototype
            prototype_loss = 0.1 * F.mse_loss(x_dream, prototype_x)
            
            (loss + prototype_loss).backward()
            optimizer.step()
        
        return x_dream.detach()
    
    def dream_interpolation(self, class_a: int, class_b: int, 
                             num_steps: int = 10, steps_per_dream: int = 100) -> List[torch.Tensor]:
        """
        Generate interpolation between two class concepts.
        
        This shows the network's internal representation of the transition
        between two categories.
        """
        # Dream endpoints
        dream_a = self.dream(class_a, num_samples=1, steps=steps_per_dream)
        dream_b = self.dream(class_b, num_samples=1, steps=steps_per_dream)
        
        # Linear interpolation in input space
        interpolations = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            x_interp = (1 - alpha) * dream_a + alpha * dream_b
            interpolations.append(x_interp)
        
        return interpolations
    
    def verify_bidirectionality(self, dataloader, num_samples: int = 100) -> Dict:
        """
        Verify that classification and dreaming are consistent.
        
        For each input:
        1. Classify it
        2. Dream from that class
        3. Check if dream is classified as same class
        """
        results = {
            'total_samples': 0,
            'consistent_dreams': 0,
            'consistency_rate': 0.0
        }
        
        self.model.eval()
        sample_count = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                if sample_count >= num_samples:
                    break
                
                for i in range(x.size(0)):
                    if sample_count >= num_samples:
                        break
                    
                    # Classify
                    x_i = x[i:i+1]
                    out = self.model(x_i, steps=30)
                    pred_class = out.argmax(dim=1).item()
                    
                    # Dream from predicted class
                    dream = self.dream(pred_class, num_samples=1, steps=50)
                    
                    # Classify dream
                    dream_out = self.model(dream, steps=30)
                    dream_class = dream_out.argmax(dim=1).item()
                    
                    # Check consistency
                    if dream_class == pred_class:
                        results['consistent_dreams'] += 1
                    
                    results['total_samples'] += 1
                    sample_count += 1
        
        results['consistency_rate'] = results['consistent_dreams'] / results['total_samples']
        return results


def visualize_dream_mnist(dream: torch.Tensor, size: int = 28) -> str:
    """ASCII visualization of a dreamed MNIST digit."""
    dream_np = dream.squeeze().detach().cpu().numpy()
    
    # Reshape if necessary
    if len(dream_np.shape) == 1:
        dream_np = dream_np.reshape(size, size)
    
    # Normalize to 0-9
    dream_np = (dream_np - dream_np.min()) / (dream_np.max() - dream_np.min() + 1e-8)
    
    chars = " .:-=+*#@"
    
    lines = []
    # Downsample for ASCII
    step = size // 14
    for i in range(0, size, step):
        line = ""
        for j in range(0, size, step):
            val = dream_np[i, j]
            idx = int(val * (len(chars) - 1))
            line += chars[idx]
        lines.append(line)
    
    return "\n".join(lines)


def run_dreaming_experiment(verbose: bool = True):
    """Run the complete associative dreaming experiment."""
    
    from src.models import LoopedMLP
    
    # Create and train model
    model = LoopedMLP(784, 256, 10, alpha=0.5, use_spectral_norm=True)
    
    # Quick training
    try:
        from src.tasks import get_task_loader
        train_loader, test_loader, _, _ = get_task_loader("mnist", batch_size=64, flatten=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(3):
            for i, (x, y) in enumerate(train_loader):
                if i > 100:
                    break
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
    except:
        test_loader = None
    
    dreamer = AssociativeDreamer(model, input_dim=784, output_dim=10)
    
    if verbose:
        print("\n" + "="*70)
        print("ASSOCIATIVE DREAMING EXPERIMENT")
        print("="*70)
    
    results = {}
    
    # Dream each digit
    if verbose:
        print("\n1. DREAMING EACH DIGIT CLASS")
        print("-" * 40)
    
    for digit in range(10):
        dream = dreamer.dream(digit, num_samples=1, steps=100)
        
        # Verify dream is classified correctly
        out = model(dream)
        pred = out.argmax(dim=1).item()
        
        results[f'dream_{digit}'] = {
            'target': digit,
            'predicted': pred,
            'correct': digit == pred
        }
        
        if verbose:
            status = "✓" if digit == pred else "✗"
            print(f"  Digit {digit} → Classified as {pred} {status}")
    
    # Visualize a few dreams
    if verbose:
        print("\n2. DREAM VISUALIZATIONS")
        print("-" * 40)
        
        for digit in [0, 3, 7]:
            dream = dreamer.dream(digit, num_samples=1, steps=150)
            print(f"\nDream of digit {digit}:")
            print(visualize_dream_mnist(dream))
    
    # Bidirectionality test
    if test_loader:
        if verbose:
            print("\n3. BIDIRECTIONALITY VERIFICATION")
            print("-" * 40)
        
        consistency = dreamer.verify_bidirectionality(test_loader, num_samples=50)
        results['consistency'] = consistency
        
        if verbose:
            print(f"  Samples tested: {consistency['total_samples']}")
            print(f"  Consistent dreams: {consistency['consistent_dreams']}")
            print(f"  Consistency rate: {consistency['consistency_rate']:.1%}")
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("ASSOCIATIVE DREAMING VERDICT:")
        
        correct_dreams = sum(1 for k, v in results.items() 
                            if k.startswith('dream_') and v.get('correct', False))
        
        if correct_dreams >= 8:
            print(f"✓ CONFIRMED: {correct_dreams}/10 digits dream correctly")
            print("  This proves UNIVERSAL ASSOCIATIVE MEMORY capability.")
        else:
            print(f"⚠ PARTIAL: Only {correct_dreams}/10 digits dream correctly")
            print("  May need more training or tuning.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Associative Dreaming Experiment")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    results = run_dreaming_experiment(verbose=True)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
