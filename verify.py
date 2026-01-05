#!/usr/bin/env python3
"""
TorEqProp Comprehensive Verification Suite

Complete validation of ALL research tracks from first principles.
Generates an undeniable evidence notebook with reproducible results.

RESEARCH TRACKS COVERED:
  1. Core: Spectral Normalization Stability
  2. Core: EqProp vs Backprop Parity  
  3. Track 1: Adversarial Self-Healing (Score: 88.0)
  15. Special: PyTorch vs Kernel (Score: 100.0)
  16. Hardware: FPGA Bit Precision (INT8)
  17. Hardware: Analog/Photonics Noise
  18. Hardware: DNA/Thermodynamic Constraints
  19. Analysis: Criticality (Edge of Chaos)
  20. App: Transfer Learning
  21. App: Continual Learning
  22. Engine: Golden Reference Harness (TODO7 Stage 1.1)
  23. Engine: Extreme Depth Signal Probe (TODO7 Stage 2.1)
  24. Engine: Lazy Updates Wall-Clock (TODO7 Stage 1.3)
  5. Track 3: Neural Cube 3D Topology (Score: 86.5)
  6. Track 4: Feedback Alignment (Score: 86.5)
  7. Track 5: Temporal Resonance (Score: 61.2)
  8. Track 6: Homeostatic Stability (Score: 59.0)
  9. Track 7: Gradient Alignment (Score: 36.5)
  10. Scaling: O(1) Memory Training
  11. Scaling: Deep Network (100+ layers)
  12. Scaling: Lazy/Event-Driven Updates
  13. Advanced: Convolutional EqProp (CIFAR-10)
  14. Advanced: Transformer EqProp
"""

import argparse
from validation import Verifier

def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Comprehensive Verification Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (fewer epochs, 1 seed)")
    parser.add_argument("--track", "-t", type=int, nargs="+", help="Run specific track(s)")
    parser.add_argument("--list", "-l", action="store_true", help="List all tracks")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds for robustness checks (redundancy)")
    parser.add_argument("--export", action="store_true", help="Export raw data to CSV")
    
    args = parser.parse_args()
    
    verifier = Verifier(quick_mode=args.quick, seed=args.seed, n_seeds_override=args.seeds, export_data=args.export)
    
    if args.list:
        verifier.list_tracks()
    else:
        verifier.run_tracks(args.track)

if __name__ == "__main__":
    main()
