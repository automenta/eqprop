#!/usr/bin/env python3
"""
Research Track Evaluation: Systematic comparison of TODO5 features.

Runs all experiments and produces a ranked list of most viable research tracks.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import time
import json


@dataclass
class TrackResult:
    """Result from evaluating a research track."""
    track_name: str
    viability_score: float  # 0-100
    novelty_score: float    # 0-100 (how unique vs prior art)
    evidence_strength: float  # 0-100 (how clear the results are)
    publication_potential: str  # "high", "medium", "low"
    key_finding: str
    metrics: Dict


def evaluate_gradient_alignment():
    """Track 1.2: Gradient Cosine Similarity"""
    print("\n" + "="*60)
    print("TRACK 1.2: GRADIENT ALIGNMENT")
    print("="*60)
    
    from scripts.gradient_alignment import run_gradient_alignment_test
    
    results = {}
    for depth in [5, 10, 20]:
        result = run_gradient_alignment_test(num_layers=depth, verbose=False)
        results[depth] = {
            'avg_sim': result.avg_similarity,
            'min_sim': result.min_similarity,
            'passed': result.passed
        }
        print(f"  Depth {depth}: avg={result.avg_similarity:.3f}, min={result.min_similarity:.3f}")
    
    # Calculate viability
    avg_all = np.mean([r['avg_sim'] for r in results.values()])
    
    if avg_all > 0.9:
        viability = 90
        finding = "Strong gradient alignment proven across depths"
    elif avg_all > 0.5:
        viability = 60
        finding = "Moderate gradient alignment, needs investigation"
    else:
        viability = 30
        finding = "Weak alignment, implementation may need refinement"
    
    return TrackResult(
        track_name="Gradient Alignment",
        viability_score=viability,
        novelty_score=70,  # Extends prior work on gradient equivalence
        evidence_strength=min(100, avg_all * 100),
        publication_potential="medium" if viability > 50 else "low",
        key_finding=finding,
        metrics=results
    )


def evaluate_ternary_weights():
    """Track 2.2: Ternary/1-Bit Learning"""
    print("\n" + "="*60)
    print("TRACK 2.2: TERNARY WEIGHTS")
    print("="*60)
    
    from src.models.ternary_eqprop import TernaryEqProp
    
    model = TernaryEqProp(input_dim=784, hidden_dim=256, output_dim=10)
    stats = model.get_model_stats()
    
    sparsity = stats['overall_sparsity']
    bit_ops = model.count_bit_operations()
    
    print(f"  Sparsity: {sparsity:.1%}")
    print(f"  Bit operations: {bit_ops:,}")
    
    # Quick training test
    x = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_loss = F.cross_entropy(model(x), y).item()
    
    for _ in range(20):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, steps=15), y)
        loss.backward()
        optimizer.step()
    
    final_loss = F.cross_entropy(model(x), y).item()
    learning = 1 - (final_loss / initial_loss)
    
    print(f"  Learning: {initial_loss:.3f} → {final_loss:.3f} ({learning:.0%} reduction)")
    
    viability = 80 if learning > 0.5 else 50
    
    return TrackResult(
        track_name="Ternary Weights",
        viability_score=viability,
        novelty_score=85,  # Novel: EqProp + quantization
        evidence_strength=learning * 100,
        publication_potential="high" if learning > 0.5 else "medium",
        key_finding=f"{sparsity:.0%} sparsity with {learning:.0%} learning capacity",
        metrics={'sparsity': sparsity, 'bit_ops': bit_ops, 'learning': learning}
    )


def evaluate_homeostatic_stability():
    """Track 3.1: Autonomic Homeostasis"""
    print("\n" + "="*60)
    print("TRACK 3.1: HOMEOSTATIC STABILITY")
    print("="*60)
    
    from src.models.homeostatic_eqprop import HomeostaticEqProp
    
    model = HomeostaticEqProp(
        input_dim=64, hidden_dim=128, output_dim=10, num_layers=5,
        adaptation_rate=0.05
    )
    
    x = torch.randn(16, 64)
    
    # Stress test: inflate weights
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.mul_(1.5)
    
    lipschitz_before = max(model._estimate_layer_lipschitz(i) for i in range(model.num_layers))
    print(f"  Pre-homeostasis Lipschitz: {lipschitz_before:.3f}")
    
    # Run homeostasis
    for _ in range(50):
        model(x, steps=15, apply_homeostasis=True)
    
    lipschitz_after = max(model._estimate_layer_lipschitz(i) for i in range(model.num_layers))
    print(f"  Post-homeostasis Lipschitz: {lipschitz_after:.3f}")
    
    recovery = (lipschitz_before - lipschitz_after) / lipschitz_before if lipschitz_before > 0 else 0
    stable = lipschitz_after < 1.0
    
    print(f"  Recovery: {recovery:.1%}, Stable: {stable}")
    
    viability = 85 if stable else 50
    
    return TrackResult(
        track_name="Homeostatic Stability",
        viability_score=viability,
        novelty_score=90,  # Very novel: self-tuning neural networks
        evidence_strength=80 if stable else 40,
        publication_potential="high" if stable else "medium",
        key_finding=f"Auto-stabilization to L={lipschitz_after:.2f}" if stable else "Needs tuning",
        metrics={'lipschitz_before': lipschitz_before, 'lipschitz_after': lipschitz_after, 'stable': stable}
    )


def evaluate_adversarial_healing():
    """Track 3.2: Adversarial Self-Healing"""
    print("\n" + "="*60)
    print("TRACK 3.2: ADVERSARIAL HEALING")
    print("="*60)
    
    from src.models import LoopedMLP
    from scripts.adversarial_healing import SelfHealingAnalyzer
    
    model = LoopedMLP(784, 256, 10, alpha=0.5, use_spectral_norm=True)
    analyzer = SelfHealingAnalyzer(model)
    
    x = torch.randn(16, 784)
    damping = analyzer.run_relaxation_damping_test(x, noise_level=1.0)
    
    damping_ratio = damping['damping_ratio']
    print(f"  Damping ratio: {damping_ratio:.3f}")
    print(f"  Initial diff: {damping['initial_difference']:.3f}")
    print(f"  Final diff: {damping['final_difference']:.3f}")
    
    strong_damping = damping_ratio < 0.3
    viability = 85 if strong_damping else 60
    
    return TrackResult(
        track_name="Adversarial Healing",
        viability_score=viability,
        novelty_score=80,
        evidence_strength=max(0, (1 - damping_ratio) * 100),
        publication_potential="high" if strong_damping else "medium",
        key_finding=f"Noise damped by {(1-damping_ratio):.0%} via contraction",
        metrics={'damping_ratio': damping_ratio}
    )


def evaluate_neural_cube():
    """Track 4.1: Neural Cube (3D Topology)"""
    print("\n" + "="*60)
    print("TRACK 4.1: NEURAL CUBE (3D TOPOLOGY)")
    print("="*60)
    
    from src.models.neural_cube import NeuralCube
    
    cube = NeuralCube(cube_size=6, input_dim=64, output_dim=10)
    
    x = torch.randn(16, 64)
    y = torch.randint(0, 10, (16,))
    
    # Training test
    optimizer = torch.optim.Adam(cube.parameters(), lr=0.01)
    initial_loss = F.cross_entropy(cube(x, steps=15), y).item()
    
    for _ in range(20):
        optimizer.zero_grad()
        loss = F.cross_entropy(cube(x, steps=15), y)
        loss.backward()
        optimizer.step()
    
    final_loss = F.cross_entropy(cube(x, steps=15), y).item()
    learning = 1 - (final_loss / initial_loss)
    
    print(f"  Learning: {initial_loss:.3f} → {final_loss:.3f} ({learning:.0%})")
    print(f"  Neurons: {cube.num_neurons} in 3D lattice")
    
    viability = 70 if learning > 0.3 else 40
    
    return TrackResult(
        track_name="Neural Cube (3D)",
        viability_score=viability,
        novelty_score=95,  # Highly novel
        evidence_strength=learning * 100,
        publication_potential="high" if learning > 0.3 else "low",
        key_finding=f"3D topology learns ({learning:.0%}), {cube.num_neurons} neurons",
        metrics={'learning': learning, 'neurons': cube.num_neurons}
    )


def evaluate_temporal_resonance():
    """Track 4.2: Temporal Resonance"""
    print("\n" + "="*60)
    print("TRACK 4.2: TEMPORAL RESONANCE")
    print("="*60)
    
    from src.models.temporal_resonance import TemporalResonanceEqProp
    
    model = TemporalResonanceEqProp(
        input_dim=32, hidden_dim=64, output_dim=10,
        oscillation_strength=0.3
    )
    
    x = torch.randn(8, 32)
    
    # Detect limit cycle
    cycle_info = model.detect_limit_cycle(x, max_steps=100)
    
    print(f"  Cycle detected: {cycle_info['cycle_detected']}")
    print(f"  Cycle length: {cycle_info['cycle_length']}")
    print(f"  Max correlation: {cycle_info['max_correlation']:.3f}")
    
    # Sequence test
    x_seq = torch.randn(4, 50, 32)
    outputs, trajectories = model.forward_sequence(x_seq, steps_per_frame=3)
    
    # Check context retention
    start_traj = trajectories[5]
    end_traj = trajectories[-1]
    context_retention = F.cosine_similarity(start_traj.flatten(1), end_traj.flatten(1)).mean().item()
    
    print(f"  Context retention: {context_retention:.3f}")
    
    viability = 75 if cycle_info['cycle_detected'] or context_retention > 0.1 else 50
    
    return TrackResult(
        track_name="Temporal Resonance",
        viability_score=viability,
        novelty_score=90,
        evidence_strength=context_retention * 100,
        publication_potential="medium",
        key_finding=f"Limit cycles: {cycle_info['cycle_detected']}, retention: {context_retention:.2f}",
        metrics={'cycle_detected': cycle_info['cycle_detected'], 'context_retention': context_retention}
    )


def evaluate_feedback_alignment():
    """Track 1.3: Feedback Alignment"""
    print("\n" + "="*60)
    print("TRACK 1.3: FEEDBACK ALIGNMENT")
    print("="*60)
    
    from src.models.feedback_alignment import FeedbackAlignmentEqProp
    
    model = FeedbackAlignmentEqProp(
        input_dim=64, hidden_dim=128, output_dim=10,
        feedback_mode='random'
    )
    
    x = torch.randn(16, 64)
    y = torch.randint(0, 10, (16,))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_loss = F.cross_entropy(model(x, steps=15), y).item()
    
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, steps=15), y)
        loss.backward()
        optimizer.step()
    
    final_loss = F.cross_entropy(model(x, steps=15), y).item()
    learning = 1 - (final_loss / initial_loss)
    
    # Alignment angles
    angles = model.get_alignment_angle()
    avg_angle = np.mean(list(angles.values()))
    
    print(f"  Learning: {initial_loss:.3f} → {final_loss:.3f} ({learning:.0%})")
    print(f"  Avg alignment angle: {avg_angle:.3f}")
    
    viability = 85 if learning > 0.5 else 60
    
    return TrackResult(
        track_name="Feedback Alignment",
        viability_score=viability,
        novelty_score=75,  # Extends prior work
        evidence_strength=learning * 100,
        publication_potential="medium",
        key_finding=f"Random feedback enables {learning:.0%} learning",
        metrics={'learning': learning, 'avg_angle': avg_angle}
    )


def run_all_evaluations():
    """Run all track evaluations and produce ranking."""
    print("\n" + "="*70)
    print("RESEARCH TRACK EVALUATION - SYSTEMATIC ANALYSIS")
    print("="*70)
    
    results = []
    
    evaluators = [
        evaluate_gradient_alignment,
        evaluate_ternary_weights,
        evaluate_homeostatic_stability,
        evaluate_adversarial_healing,
        evaluate_neural_cube,
        evaluate_temporal_resonance,
        evaluate_feedback_alignment,
    ]
    
    for evaluator in evaluators:
        try:
            result = evaluator()
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Rank by composite score
    for r in results:
        r.composite_score = (r.viability_score * 0.4 + 
                            r.novelty_score * 0.3 + 
                            r.evidence_strength * 0.3)
    
    results.sort(key=lambda x: x.composite_score, reverse=True)
    
    # Print ranking
    print("\n" + "="*70)
    print("RESEARCH TRACK RANKING")
    print("="*70)
    print(f"\n{'Rank':<6} {'Track':<25} {'Composite':<12} {'Pub':<8} {'Finding'}")
    print("-" * 90)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<6} {r.track_name:<25} {r.composite_score:<12.1f} {r.publication_potential:<8} {r.key_finding[:40]}")
    
    print("\n" + "="*70)
    print("TOP 3 RECOMMENDED RESEARCH TRACKS")
    print("="*70)
    
    for i, r in enumerate(results[:3], 1):
        print(f"\n{i}. {r.track_name}")
        print(f"   Viability: {r.viability_score}/100, Novelty: {r.novelty_score}/100")
        print(f"   Evidence: {r.evidence_strength:.0f}/100")
        print(f"   Publication Potential: {r.publication_potential.upper()}")
        print(f"   Key Finding: {r.key_finding}")
    
    return results


if __name__ == "__main__":
    results = run_all_evaluations()
    
    # Save results
    output = {
        'ranking': [
            {
                'rank': i+1,
                'track': r.track_name,
                'composite_score': r.composite_score,
                'viability': r.viability_score,
                'novelty': r.novelty_score,
                'evidence': r.evidence_strength,
                'publication': r.publication_potential,
                'finding': r.key_finding,
                'metrics': r.metrics
            }
            for i, r in enumerate(results)
        ]
    }
    
    with open('results/research_track_evaluation.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/research_track_evaluation.json")
