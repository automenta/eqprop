#!/usr/bin/env python3
"""
RL Benchmark Script

Test IDEA models on RL tasks (CartPole, Acrobot).
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_tasks import train_rl, compare_rl_models


def main():
    parser = argparse.ArgumentParser(description='RL benchmarks')
    parser.add_argument('--env', type=str, default='CartPole-v1', 
                       help='Environment (CartPole-v1, Acrobot-v1)')
    parser.add_argument('--model', type=str, default=None,
                       help='Single model to test')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--output', type=str, default='results/rl_benchmark.json')
    args = parser.parse_args()
    
    if args.compare:
        results = compare_rl_models(args.env, episodes=args.episodes, seeds=args.seeds)
    elif args.model:
        from src.models import MODEL_REGISTRY
        model_class = MODEL_REGISTRY[args.model]
        result = train_rl(args.env, args.episodes, actor_model=model_class, 
                         critic_model=model_class)
        results = {args.model: result}
    else:
        # Default: train with TPEqProp
        print("Training with TPEqProp (use --compare for model comparison)")
        result = train_rl(args.env, args.episodes)
        results = {'TPEqProp': result}
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved to {args.output}")


if __name__ == '__main__':
    main()
