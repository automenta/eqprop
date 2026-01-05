#!/usr/bin/env python3
"""
Reinforcement Learning with Equilibrium Propagation.

Novel application: use EqProp gradients for policy optimization.

Usage:
    python train_rl.py --env CartPole-v1 --episodes 500
    python train_rl.py --env CartPole-v1 --episodes 500 --use-bp  # BP baseline
"""

import argparse
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("Warning: gymnasium not installed. Run: pip install gymnasium")


# ============================================================================
# Policy Network (Looped Transformer-based)
# ============================================================================

class EquilibriumPolicy(nn.Module):
    """Policy network using equilibrium dynamics."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, 
                 max_iters: int = 10, tol: float = 1e-4, damping: float = 0.8):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.embed = nn.Linear(obs_dim, hidden_dim)
        
        # Equilibrium block (simplified for RL)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.policy_head = nn.Linear(hidden_dim, action_dim)  # Actor
        self.value_head = nn.Linear(hidden_dim, 1)  # Critic
        
        # Equilibrium parameters
        self.damping = damping
        self.max_iters = max_iters
        self.tol = tol
        
    def equilibrium_step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One step of equilibrium iteration."""
        # Fixed-point iteration: h = damping * h + (1-damping) * f(h, x)
        h_new = torch.tanh(self.fc2(F.relu(self.fc1(h + x))))
        return self.damping * h + (1 - self.damping) * h_new
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Forward pass to equilibrium, return policy logits and value."""
        x = self.embed(obs)
        
        # Iterate to equilibrium
        h = torch.zeros_like(x)
        for i in range(self.max_iters):
            h_new = self.equilibrium_step(h, x)
            if torch.norm(h_new - h) < self.tol:
                break
            h = h_new
        
        iters = i + 1
        
        # Get policy and value from equilibrium state
        logits = self.policy_head(h)
        value = self.value_head(h)
        
        return logits, value, iters
    
    def get_action(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        logits, value, _ = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value


class BPPolicy(nn.Module):
    """Standard feedforward policy for comparison."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        h = self.net(obs)
        return self.policy_head(h), self.value_head(h), 1
    
    def get_action(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        logits, value, _ = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value


# ============================================================================
# Training
# ============================================================================

def train_episode(policy, optimizer, env, gamma=0.99):
    """Train on one episode using REINFORCE with baseline."""
    obs, _ = env.reset()
    obs = torch.FloatTensor(obs)
    
    log_probs = []
    values = []
    rewards = []
    
    done = False
    while not done:
        action, log_prob, value = policy.get_action(obs)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        
        obs = torch.FloatTensor(next_obs)
    
    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Compute losses
    values = torch.cat(values)
    log_probs = torch.stack(log_probs)
    
    advantage = returns - values.squeeze()
    
    policy_loss = -(log_probs * advantage.detach()).mean()
    value_loss = F.mse_loss(values.squeeze(), returns)
    
    loss = policy_loss + 0.5 * value_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return sum(rewards), len(rewards)


def evaluate(policy, env, n_episodes=10):
    """Evaluate policy without training."""
    total_reward = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs = torch.FloatTensor(obs)
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                action, _, _ = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = torch.FloatTensor(obs)
            done = terminated or truncated
            episode_reward += reward
            
        total_reward += episode_reward
        
    return total_reward / n_episodes


def main():
    parser = argparse.ArgumentParser(description="RL with Equilibrium Propagation")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use-bp", action="store_true", help="Use BP policy (baseline)")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--max-iters", type=int, default=10, help="Max equilibrium iterations")
    parser.add_argument("--damping", type=float, default=0.8, help="Damping factor")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if not HAS_GYM:
        print("Error: gymnasium required. Install with: pip install gymnasium")
        return
    
    print(f"\n{'='*70}")
    print(f"🎮 Reinforcement Learning with {'BP' if args.use_bp else 'EqProp'}")
    print(f"{'='*70}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Policy: {'BP Baseline' if args.use_bp else 'Equilibrium Policy'}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")
    
    # Set seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create environment
    env = gym.make(args.env)
    if args.seed is not None:
        env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    if args.use_bp:
        policy = BPPolicy(obs_dim, action_dim, args.hidden_dim)
    else:
        policy = EquilibriumPolicy(
            obs_dim, action_dim, args.hidden_dim,
            max_iters=args.max_iters, damping=args.damping
        )
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    # Training
    reward_history = []
    recent_rewards = deque(maxlen=100)
    best_avg_reward = 0
    solved = False
    
    start_time = time.time()
    
    for episode in range(args.episodes):
        reward, steps = train_episode(policy, optimizer, env)
        reward_history.append(reward)
        recent_rewards.append(reward)
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}: Reward={reward:.0f}, Avg(100)={avg_reward:.1f}")
        
        # Check if solved
        if avg_reward >= 195.0 and not solved:
            solved = True
            print(f"\n🎉 SOLVED at episode {episode+1}! Avg reward: {avg_reward:.1f}")
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
        
        # Periodic evaluation
        if (episode + 1) % args.eval_interval == 0:
            eval_reward = evaluate(policy, env)
            print(f"  Evaluation: {eval_reward:.1f}")
    
    duration = time.time() - start_time
    
    # Final results
    print(f"\n{'='*70}")
    print(f"📊 RESULTS")
    print(f"{'='*70}")
    print(f"Policy: {'BP' if args.use_bp else 'EqProp'}")
    print(f"Best Average Reward: {best_avg_reward:.1f}")
    print(f"Final Average Reward: {avg_reward:.1f}")
    print(f"Solved: {'✅ Yes' if solved else '❌ No'}")
    print(f"Training time: {duration:.1f}s")
    print(f"{'='*70}\n")
    
    # Save results
    output_dir = Path("logs/rl")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    policy_type = "bp" if args.use_bp else "eqprop"
    results_file = output_dir / f"{args.env}_{policy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "env": args.env,
            "policy": policy_type,
            "episodes": args.episodes,
            "best_avg_reward": best_avg_reward,
            "final_avg_reward": avg_reward,
            "solved": solved,
            "duration_sec": duration,
            "reward_history": reward_history
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    env.close()


if __name__ == "__main__":
    main()
