#!/usr/bin/env python3
"""
Multi-Environment RL Test Suite

Test ModernEqProp across multiple RL environments systematically.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError:
    import gym

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Environment configurations
ENV_CONFIGS = {
    'CartPole-v1': {
        'state_dim': 4,
        'action_dim': 2,
        'max_steps': 500,
        'solve_threshold': 195,
        'episodes': 200,
        'lr': 1e-3,
        'eqprop_steps': 10,
    },
    'Acrobot-v1': {
        'state_dim': 6,
        'action_dim': 3,
        'max_steps': 500,
        'solve_threshold': -100,  # Lower is better
        'episodes': 300,
        'lr': 1e-3,
        'eqprop_steps': 10,
    },
    'MountainCar-v0': {
        'state_dim': 2,
        'action_dim': 3,
        'max_steps': 200,
        'solve_threshold': -110,
        'episodes': 300,
        'lr': 3e-3,  # Higher LR for sparse rewards
        'eqprop_steps': 5,
    },
}


class SimplePolicy(nn.Module):
    """Simple MLP policy."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x, **kwargs):
        return self.net(x)


class EqPropPolicy(nn.Module):
    """EqProp policy wrapper."""
    def __init__(self, state_dim, action_dim, hidden_dim=64, model_name='ModernEqProp', steps=10):
        super().__init__()
        model_class = MODEL_REGISTRY[model_name]
        self.backbone = model_class(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            use_spectral_norm=False
        )
        self.steps = steps
    
    def forward(self, x, **kwargs):
        return self.backbone(x, steps=self.steps)


def train_reinforce(policy, env_name, config, is_eqprop=False):
    """Train using REINFORCE."""
    env = gym.make(env_name)
    optimizer = optim.Adam(policy.parameters(), lr=config['lr'])
    gamma = 0.99
    
    rewards_history = []
    
    for episode in range(config['episodes']):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        log_probs = []
        rewards = []
        
        for step in range(config['max_steps']):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits = policy(state_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            
            result = env.step(action.item())
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
            
            rewards.append(reward)
            state = next_state
            
            if done:
                break
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(DEVICE)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient
        policy_loss = sum(-lp * G for lp, G in zip(log_probs, returns))
        
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        episode_reward = sum(rewards)
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            mean_recent = np.mean(rewards_history[-50:])
            print(f"  Episode {episode+1}: Mean50={mean_recent:.1f}")
    
    env.close()
    
    final_perf = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
    solved = (final_perf >= config['solve_threshold'] if 'CartPole' in env_name
              else final_perf >= config['solve_threshold'])
    
    return {
        'rewards': rewards_history,
        'final_performance': final_perf,
        'solved': solved
    }


def test_environment(env_name, seeds=2):
    """Test EqProp vs MLP on one environment."""
    
    config = ENV_CONFIGS[env_name]
    
    print(f"\n{'='*70}")
    print(f"Environment: {env_name}")
    print(f"  State dim: {config['state_dim']}, Actions: {config['action_dim']}")
    print(f"  Solve threshold: {config['solve_threshold']}, Episodes: {config['episodes']}")
    print(f"{'='*70}")
    
    results = {}
    
    for model_type in ['MLP', 'ModernEqProp']:
        print(f"\nTesting {model_type}:")
        seed_results = []
        
        for seed in range(42, 42 + seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create policy
            if model_type == 'MLP':
                policy = SimplePolicy(config['state_dim'], config['action_dim'], 64).to(DEVICE)
                is_eqprop = False
            else:
                policy = EqPropPolicy(
                    config['state_dim'], config['action_dim'], 64,
                    model_name=model_type, steps=config['eqprop_steps']
                ).to(DEVICE)
                is_eqprop = True
            
            result = train_reinforce(policy, env_name, config, is_eqprop)
            seed_results.append(result)
            
            print(f"    Seed {seed}: Final={result['final_performance']:.1f}, "
                  f"Solved={result['solved']}")
        
        # Aggregate
        final_perfs = [r['final_performance'] for r in seed_results]
        results[model_type] = {
            'mean': np.mean(final_perfs),
            'std': np.std(final_perfs),
            'solve_rate': sum(r['solved'] for r in seed_results) / seeds,
            'seeds': seed_results
        }
    
    # Summary
    print(f"\n{env_name} Summary:")
    print(f"  MLP: {results['MLP']['mean']:.1f} ± {results['MLP']['std']:.1f}, "
          f"Solve: {results['MLP']['solve_rate']:.0%}")
    print(f"  ModernEqProp: {results['ModernEqProp']['mean']:.1f} ± {results['ModernEqProp']['std']:.1f}, "
          f"Solve: {results['ModernEqProp']['solve_rate']:.0%}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=str, default='CartPole-v1',
                       help='Comma-separated list of environments')
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--output', type=str, default='results/rl_multi_env.json')
    args = parser.parse_args()
    
    envs_to_test = args.envs.split(',')
    
    all_results = {}
    
    for env_name in envs_to_test:
        if env_name.strip() in ENV_CONFIGS:
            results = test_environment(env_name.strip(), args.seeds)
            all_results[env_name.strip()] = {
                'MLP': {k: v for k, v in results['MLP'].items() if k != 'seeds'},
                'ModernEqProp': {k: v for k, v in results['ModernEqProp'].items() if k != 'seeds'}
            }
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Saved to {args.output}")
    
    # Overall summary
    print("\n" + "="*70)
    print("MULTI-ENVIRONMENT SUMMARY")
    print("="*70)
    print(f"{'Environment':<20} {'MLP':>15} {'ModernEqProp':>15} {'Winner':>10}")
    print("-"*70)
    
    for env_name, results in all_results.items():
        mlp_perf = results['MLP']['mean']
        eq_perf = results['ModernEqProp']['mean']
        winner = "EqProp" if eq_perf > mlp_perf else "MLP"
        
        print(f"{env_name:<20} {mlp_perf:>14.1f} {eq_perf:>14.1f} {winner:>10}")


if __name__ == '__main__':
    main()
