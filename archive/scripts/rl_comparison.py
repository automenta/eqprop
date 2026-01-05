#!/usr/bin/env python3
"""
RL Comparison: EqProp vs MLP

Compare EqProp models against standard MLP on CartPole.
CartPole has only 4 input dimensions - ideal for EqProp based on our findings.

Key metrics:
- Sample efficiency (episodes to solve)
- Final reward
- Training time
"""

import argparse
import json
from pathlib import Path
import time
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


class MLPPolicy(nn.Module):
    """Simple MLP policy for RL baseline."""
    
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
    """EqProp-based policy for RL."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, model_class=None):
        super().__init__()
        
        if model_class is None:
            model_class = MODEL_REGISTRY.get('TPEqProp', MODEL_REGISTRY['ModernEqProp'])
        
        self.backbone = model_class(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            use_spectral_norm=True
        )
    
    def forward(self, x, steps=10, **kwargs):
        return self.backbone(x, steps=steps)


def reinforce_train(policy, env_name='CartPole-v1', episodes=500, lr=3e-3, 
                    gamma=0.99, max_steps=500, is_eqprop=False, steps_if_eqprop=10):
    """Train policy using REINFORCE algorithm."""
    
    env = gym.make(env_name)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    episode_rewards = []
    solved_at = None
    
    for episode in range(episodes):
        # Collect trajectory
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        log_probs = []
        rewards = []
        
        for step in range(max_steps):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            
            # Get action
            if is_eqprop:
                logits = policy(state_t, steps=steps_if_eqprop)
            else:
                logits = policy(state_t)
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            log_probs.append(log_prob)
            
            # Step environment
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
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        policy_loss = 0
        for log_prob, G in zip(log_probs, returns):
            policy_loss -= log_prob * G
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Track
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        # Check if solved
        if len(episode_rewards) >= 100:
            mean_100 = np.mean(episode_rewards[-100:])
            if mean_100 >= 195 and solved_at is None:
                solved_at = episode + 1
        
        # Print progress
        if (episode + 1) % 50 == 0:
            mean_reward = np.mean(episode_rewards[-50:])
            print(f"  Episode {episode+1}/{episodes}: Mean50={mean_reward:.1f}")
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'solved_at': solved_at,
        'final_mean100': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    }


def compare_rl(env_name='CartPole-v1', episodes=300, hidden_dim=64, 
               eqprop_steps=10, seeds=3):
    """Compare EqProp vs MLP on RL task."""
    
    print(f"\n🎮 RL Comparison: EqProp vs MLP")
    print(f"   Env: {env_name}, Episodes: {episodes}, Seeds: {seeds}")
    print(f"   Device: {DEVICE}")
    print("="*70)
    
    # Get env info
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    print(f"   State dim: {state_dim}, Action dim: {action_dim}")
    
    results = {}
    
    # Test configurations
    configs = [
        ('MLP', False, None),
        ('TPEqProp', True, MODEL_REGISTRY.get('TPEqProp')),
        ('SpectralTorEqProp', True, MODEL_REGISTRY.get('SpectralTorEqProp')),
    ]
    
    for name, is_eqprop, model_class in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        seed_results = []
        
        for seed in range(42, 42 + seeds):
            print(f"\n  Seed {seed}:")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create policy
            if is_eqprop:
                policy = EqPropPolicy(state_dim, action_dim, hidden_dim, model_class).to(DEVICE)
            else:
                policy = MLPPolicy(state_dim, action_dim, hidden_dim).to(DEVICE)
            
            param_count = sum(p.numel() for p in policy.parameters())
            
            # Train
            start_time = time.time()
            result = reinforce_train(
                policy, env_name, episodes,
                lr=3e-3, gamma=0.99, max_steps=500,
                is_eqprop=is_eqprop, steps_if_eqprop=eqprop_steps
            )
            train_time = time.time() - start_time
            
            result['train_time_s'] = train_time
            result['params'] = param_count
            seed_results.append(result)
            
            print(f"    Final Mean100: {result['final_mean100']:.1f}, "
                  f"Solved: {result['solved_at'] or 'No'}, Time: {train_time:.1f}s")
        
        # Aggregate
        final_rewards = [r['final_mean100'] for r in seed_results]
        solved_counts = sum(1 for r in seed_results if r['solved_at'] is not None)
        
        results[name] = {
            'mean_reward': np.mean(final_rewards),
            'std_reward': np.std(final_rewards),
            'solve_rate': solved_counts / seeds,
            'mean_time_s': np.mean([r['train_time_s'] for r in seed_results]),
            'params': seed_results[0]['params'],
            'seeds': seed_results
        }
    
    # Summary
    print("\n" + "="*80)
    print("RL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Mean Reward':>12} {'Solve Rate':>12} {'Time (s)':>10} {'Params':>10}")
    print("-"*80)
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True):
        print(f"{name:<20} {data['mean_reward']:>11.1f} {data['solve_rate']:>11.0%} "
              f"{data['mean_time_s']:>9.1f} {data['params']:>9,}")
    
    # Winner
    best = max(results.items(), key=lambda x: x[1]['mean_reward'])
    print(f"\n🏆 Best: {best[0]} (reward: {best[1]['mean_reward']:.1f})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='RL comparison')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--eqprop-steps', type=int, default=10)
    parser.add_argument('--output', type=str, default='results/rl_comparison.json')
    args = parser.parse_args()
    
    results = compare_rl(
        args.env, args.episodes, args.hidden_dim,
        args.eqprop_steps, args.seeds
    )
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert for JSON
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            'mean_reward': data['mean_reward'],
            'std_reward': data['std_reward'],
            'solve_rate': data['solve_rate'],
            'mean_time_s': data['mean_time_s'],
            'params': data['params']
        }
    
    with open(args.output, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Saved to {args.output}")


if __name__ == '__main__':
    main()
