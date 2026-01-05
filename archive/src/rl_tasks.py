"""
Reinforcement Learning with EqProp

Adapts IDEA models for RL tasks using actor-critic architecture.
Key advantages:
- O(1) memory for deep policies
- Energy-based value functions
- Local updates for distributed RL

Supports: CartPole, Acrobot
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class EqPropActor(nn.Module):
    """Policy network using EqProp dynamics."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, model_class=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Use any EqProp model as backbone
        if model_class is None:
            from src.models import TPEqProp
            model_class = TPEqProp
        
        self.backbone = model_class(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            use_spectral_norm=True
        )
    
    def forward(self, state, steps=10):
        """Get action logits from state."""
        return self.backbone(state, steps=steps)
    
    def act(self, state, steps=10):
        """Sample action from policy."""
        with torch.no_grad():
            logits = self.forward(state.unsqueeze(0), steps=steps)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action, probs[0]


class EqPropCritic(nn.Module):
    """Value function using EqProp."""
    
    def __init__(self, state_dim, hidden_dim=64, model_class=None):
        super().__init__()
        
        if model_class is None:
            from src.models import TPEqProp
            model_class = TPEqProp
        
        self.backbone = model_class(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=1,  # Single value output
            use_spectral_norm=True
        )
    
    def forward(self, state, steps=10):
        """Get state value."""
        return self.backbone(state, steps=steps).squeeze(-1)


class EqPropActorCritic:
    """Actor-Critic RL agent using EqProp models."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, 
                 actor_model=None, critic_model=None,
                 actor_lr=3e-4, critic_lr=1e-3, gamma=0.99):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.actor = EqPropActor(state_dim, action_dim, hidden_dim, actor_model).to(self.device)
        self.critic = EqPropCritic(state_dim, hidden_dim, critic_model).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """Single training step."""
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Critic update
        values = self.critic(states)
        next_values = self.critic(next_states).detach()
        targets = rewards + self.gamma * next_values * (1 - dones)
        
        critic_loss = F.mse_loss(values, targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        logits = self.actor(states)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        advantages = (targets - values).detach()
        actor_loss = -(action_log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_value': values.mean().item()
        }


def train_rl(env_name='CartPole-v1', episodes=500, max_steps=500,
             hidden_dim=64, actor_model=None, critic_model=None,
             verbose=True):
    """Train RL agent."""
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = EqPropActorCritic(
        state_dim, action_dim, hidden_dim,
        actor_model, critic_model
    )
    
    episode_rewards = []
    recent_rewards = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # New gym API
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Act
            action, _ = agent.actor.act(torch.FloatTensor(state).to(agent.device))
            
            # Step environment
            result = env.step(action)
            if len(result) == 5:  # New gym API
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
            
            # Train
            agent.train_step(
                [state], [action], [reward], [next_state], [float(done)]
            )
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        if verbose and (episode + 1) % 50 == 0:
            mean_reward = np.mean(recent_rewards)
            print(f"Episode {episode+1}/{episodes}: Reward={episode_reward:.1f}, "
                  f"Mean(100)={mean_reward:.1f}")
            
            # Check if solved
            if env_name == 'CartPole-v1' and mean_reward >= 195:
                print(f"✅ Solved in {episode+1} episodes!")
                break
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'mean_reward': np.mean(recent_rewards),
        'episodes': len(episode_rewards)
    }


def compare_rl_models(env_name='CartPole-v1', models=None, episodes=300, seeds=3):
    """Compare different EqProp models on RL tasks."""
    
    if models is None:
        from src.models import TPEqProp, SpectralTorEqProp, ModernEqProp
        models = {
            'TPEqProp': TPEqProp,
            'SpectralTorEqProp': SpectralTorEqProp,
            'ModernEqProp': ModernEqProp
        }
    
    results = {}
    
    for model_name, model_class in models.items():
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        seed_results = []
        
        for seed in range(42, 42 + seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            result = train_rl(
                env_name, episodes, 
                actor_model=model_class,
                critic_model=model_class,
                verbose=False
            )
            
            seed_results.append(result)
            print(f"  Seed {seed}: Episodes={result['episodes']}, "
                  f"Mean Reward={result['mean_reward']:.1f}")
        
        # Aggregate
        results[model_name] = {
            'mean_reward': np.mean([r['mean_reward'] for r in seed_results]),
            'mean_episodes': np.mean([r['episodes'] for r in seed_results]),
            'seeds': seed_results
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("RL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Mean Reward':>12} {'Episodes':>10}")
    print("-"*60)
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True):
        print(f"{name:<20} {data['mean_reward']:>11.1f} {data['mean_episodes']:>9.0f}")
    
    return results
