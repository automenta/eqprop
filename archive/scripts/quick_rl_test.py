#!/usr/bin/env python3
"""
Quick RL Test - Find What Works

Test different EqProp configurations on CartPole to find a working setup.
"""

import argparse
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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MODEL_REGISTRY


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleMLPPolicy(nn.Module):
    """Baseline MLP."""
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x, **kwargs):
        return self.net(x)


def quick_rl_test(model_name, episodes=150, lr=1e-3, hidden_dim=32, eqprop_steps=5):
    """Quick RL test."""
    
    env = gym.make('CartPole-v1')
    state_dim = 4
    action_dim = 2
    
    # Create policy
    if model_name == 'MLP':
        policy = SimpleMLPPolicy(state_dim, action_dim, hidden_dim).to(DEVICE)
        is_eqprop = False
    else:
        try:
            model_class = MODEL_REGISTRY[model_name]
            policy_backbone = model_class(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                output_dim=action_dim,
                use_spectral_norm=False  # Try without spectral norm
            ).to(DEVICE)
            
            class EqPropPolicyWrapper(nn.Module):
                def __init__(self, backbone, steps):
                    super().__init__()
                    self.backbone = backbone
                    self.steps = steps
                
                def forward(self, x, **kwargs):
                    return self.backbone(x, steps=self.steps)
            
            policy = EqPropPolicyWrapper(policy_backbone, eqprop_steps)
            is_eqprop = True
        except Exception as e:
            print(f"Failed to create {model_name}: {e}")
            return None
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    gamma = 0.99
    
    rewards_history = []
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"LR: {lr}, Hidden: {hidden_dim}, Steps: {eqprop_steps if is_eqprop else 'N/A'}")
    print(f"{'='*50}")
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        log_probs = []
        rewards = []
        
        for step in range(500):
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
        
        # Returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(DEVICE)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update
        policy_loss = sum(-lp * G for lp, G in zip(log_probs, returns))
        
        optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        
        optimizer.step()
        
        episode_reward = sum(rewards)
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 25 == 0:
            mean_25 = np.mean(rewards_history[-25:])
            print(f"Episode {episode+1}: Mean25={mean_25:.1f}, Last={episode_reward:.0f}")
    
    env.close()
    
    final_mean = np.mean(rewards_history[-50:])
    solved = final_mean >= 195
    
    print(f"\nFinal Mean50: {final_mean:.1f}, Solved: {solved}")
    
    return {
        'rewards': rewards_history,
        'final_mean': final_mean,
        'solved': solved
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=150)
    args = parser.parse_args()
    
    # Test configurations
    configs = [
        # Baseline
        ('MLP', 1e-3, 32, 0),
        
        # ModernEqProp
        ('ModernEqProp', 1e-3, 32, 5),
        ('ModernEqProp', 3e-3, 32, 5),
        ('ModernEqProp', 1e-3, 32, 10),
        
        # TPEqProp
        ('TPEqProp', 1e-3, 32, 5),
        
        # SpectralTorEqProp
        ('SpectralTorEqProp', 1e-3, 32, 5),
    ]
    
    results = {}
    
    for model_name, lr, hidden, steps in configs:
        config_name = f"{model_name}_lr{lr:.0e}_h{hidden}_s{steps}"
        result = quick_rl_test(model_name, args.episodes, lr, hidden, steps)
        if result:
            results[config_name] = result
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Config':<40} {'Mean50':>10} {'Solved':>10}")
    print("-"*60)
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['final_mean'], reverse=True):
        solved_str = "✓" if data['solved'] else "✗"
        print(f"{name:<40} {data['final_mean']:>9.1f} {solved_str:>10}")


if __name__ == '__main__':
    main()
