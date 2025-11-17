"""
DreamerV3: Modular Implementation for Plug-and-Play Usage
Based on "Mastering Diverse Domains through World Models" (Hafner et al., 2024)

This implementation provides a clean, modular DreamerV3 agent that can be
used with any world model for various applications (autonomous driving, robotics, etc.)
"""

import numpy as np
import torch
from DreamerConfig import DreamerConfig
import Actor
import Critic
import ReturnNormalizer
from torch.distributions import Normal, Categorical, Independent
from typing import Dict, Tuple, Optional, List

class DreamerV3Agent:
    """
    DreamerV3 reinforcement learning agent for use with world models.
    
    This agent learns behaviors purely from imagined trajectories predicted
    by a world model, making it applicable to any domain where a world model
    can be trained.
    """
    
    def __init__(self, 
                 feature_dim: int,
                 action_dim: int, 
                 action_type: str = 'continuous',
                 config: Optional[DreamerConfig] = None,
                 device: str = 'cuda'):
        """
        Args:
            feature_dim: Dimension of world model features (concat of h_t and z_t)
            action_dim: Action space dimension
            action_type: 'continuous' or 'discrete'
            config: DreamerV3 configuration
            device: Torch device
        """
        self.config = config or DreamerConfig()
        self.device = device
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.action_type = action_type
        
        # Actor and Critic networks
        self.actor = Actor(feature_dim, action_dim, self.config, action_type).to(device)
        self.critic = Critic(feature_dim, self.config).to(device)
        
        # Slow/target critic (EMA of main critic)
        self.slow_critic = Critic(feature_dim, self.config).to(device)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        
        # Return normalizer
        self.return_normalizer = ReturnNormalizer(self.config)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate
        )
    
    def compute_lambda_returns(self,
                               rewards: torch.Tensor,
                               values: torch.Tensor,
                               continues: torch.Tensor,
                               bootstrap: torch.Tensor) -> torch.Tensor:
        """
        Compute λ-returns for value targets.
        
        Args:
            rewards: [B, T] reward predictions
            values: [B, T+1] value predictions  
            continues: [B, T] continuation probabilities (1 - done)
            bootstrap: [B] bootstrap value for last timestep
            
        Returns:
            returns: [B, T] λ-returns
        """
        B, T = rewards.shape
        discount = self.config.discount
        lambda_ = self.config.lambda_
        
        # Initialize returns from the end
        returns = []
        last_return = bootstrap
        
        # Backward iteration for λ-return computation
        for t in reversed(range(T)):
            # λ-return formula: r_t + γ*c_t*((1-λ)*V_t+1 + λ*R_t+1)
            value_next = values[:, t + 1] if t < T - 1 else bootstrap
            lambda_return = (
                rewards[:, t] + 
                discount * continues[:, t] * (
                    (1 - lambda_) * value_next + lambda_ * last_return
                )
            )
            returns.append(lambda_return)
            last_return = lambda_return
        
        returns = torch.stack(list(reversed(returns)), dim=1)
        return returns
    
    def imagine_trajectories(self,
                            world_model,
                            initial_features: torch.Tensor,
                            horizon: Optional[int] = None) -> Tuple:
        """
        Imagine trajectories using the world model and current policy.
        
        Args:
            world_model: World model with methods:
                - imagine_step(features, actions) -> next_features
                - predict_reward(features) -> rewards
                - predict_continue(features) -> continues
            initial_features: [B, D] starting model states
            horizon: Length of imagination (defaults to config.horizon)
            
        Returns:
            imagined_features: [B, H+1, D]
            imagined_actions: [B, H, A]
            imagined_rewards: [B, H]
            imagined_continues: [B, H]
        """
        horizon = horizon or self.config.horizon
        B = initial_features.shape[0]
        
        features_list = [initial_features]
        actions_list = []
        rewards_list = []
        continues_list = []
        
        current_features = initial_features
        
        for _ in range(horizon):
            # Sample action from policy
            with torch.no_grad():
                action, _ = self.actor(current_features)
            
            # Predict next state using world model
            next_features = world_model.imagine_step(current_features, action)
            reward = world_model.predict_reward(next_features)
            cont = world_model.predict_continue(next_features)
            
            features_list.append(next_features)
            actions_list.append(action)
            rewards_list.append(reward)
            continues_list.append(cont)
            
            current_features = next_features
        
        return (
            torch.stack(features_list, dim=1),  # [B, H+1, D]
            torch.stack(actions_list, dim=1),   # [B, H, A]
            torch.stack(rewards_list, dim=1),   # [B, H]
            torch.stack(continues_list, dim=1)  # [B, H]
        )
    
    def update_actor_critic(self,
                           world_model,
                           features: torch.Tensor) -> Dict[str, float]:
        """
        Update actor and critic using imagined trajectories.
        
        Args:
            world_model: Trained world model
            features: [B, D] model states from real data
            
        Returns:
            metrics: Dictionary of training metrics
        """
        B = features.shape[0]
        
        # Imagine trajectories
        img_features, img_actions, img_rewards, img_continues = \
            self.imagine_trajectories(world_model, features)
        
        # Compute values
        img_values, img_value_logits = self.critic(img_features)
        with torch.no_grad():
            img_slow_values, _ = self.slow_critic(img_features)
        
        # Use slow/target values for stability
        target_values = img_slow_values
        
        # Compute λ-returns
        returns = self.compute_lambda_returns(
            img_rewards,
            target_values,
            img_continues,
            target_values[:, -1]  # Bootstrap from last value
        )
        
        # Update return normalizer
        self.return_normalizer.update(returns)
        
        # === Critic Update ===
        self.critic_optimizer.zero_grad()
        
        # Value loss: predict λ-returns
        value_targets = returns.detach()
        value_loss = self.critic.loss(
            img_features[:, :-1].detach(),  # Stop gradients to world model
            value_targets
        ).mean()
        
        # Slow critic regularization
        slow_value_targets = img_slow_values[:, :-1].detach()
        slow_reg_loss = self.critic.loss(
            img_features[:, :-1].detach(),
            slow_value_targets
        ).mean()
        
        total_value_loss = (
            self.config.value_loss_scale * value_loss +
            self.config.value_loss_scale * slow_reg_loss
        )
        
        total_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 
            self.config.grad_clip
        )
        self.critic_optimizer.step()
        
        # === Actor Update ===
        self.actor_optimizer.zero_grad()
        
        # Advantages with return normalization
        normalized_returns = self.return_normalizer.normalize(returns)
        advantages = normalized_returns - target_values[:, :-1].detach()
        
        # Policy gradient with entropy regularization
        _, img_policy_dist = self.actor(img_features[:, :-1])
        
        if self.action_type == 'continuous':
            log_probs = img_policy_dist.log_prob(img_actions)
            entropy = img_policy_dist.entropy()
        else:
            log_probs = img_policy_dist.log_prob(img_actions)
            entropy = img_policy_dist.entropy()
        
        policy_loss = -(
            log_probs * advantages.detach() + 
            self.config.entropy_scale * entropy
        ).mean()
        
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.grad_clip
        )
        self.actor_optimizer.step()
        
        # Update slow critic (EMA)
        self._update_slow_critic()
        
        # Metrics
        metrics = {
            'actor_loss': policy_loss.item(),
            'critic_loss': value_loss.item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'return_mean': returns.mean().item(),
            'value_mean': img_values[:, :-1].mean().item(),
            'entropy': entropy.mean().item(),
            'return_scale': self.return_normalizer.scale,
        }
        
        return metrics
    
    def _update_slow_critic(self):
        """Update slow/target critic using exponential moving average"""
        decay = self.config.value_ema_decay
        for slow_param, param in zip(
            self.slow_critic.parameters(),
            self.critic.parameters()
        ):
            slow_param.data.copy_(
                decay * slow_param.data + (1 - decay) * param.data
            )
    
    def act(self, features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Select action given current world model features.
        
        Args:
            features: [B, D] or [D] model state features
            deterministic: If True, return deterministic action
            
        Returns:
            action: Selected action(s)
        """
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            action, _ = self.actor(features, deterministic)
            
            if features.shape[0] == 1:
                action = action.squeeze(0)
            
            return action
    
    def save(self, path: str):
        """Save agent networks"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'slow_critic': self.slow_critic.state_dict(),
            'return_scale': self.return_normalizer.scale,
        }, path)
    
    def load(self, path: str):
        """Load agent networks"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.slow_critic.load_state_dict(checkpoint['slow_critic'])
        self.return_normalizer.scale = checkpoint['return_scale']

