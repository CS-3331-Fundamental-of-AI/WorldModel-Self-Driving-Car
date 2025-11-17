import torch.nn as nn
import torch
import DreamerConfig
import MLPNetwork
import numpy as np
from torch.distributions import Independent, Categorical, Normal
import torch.functional as F

class Actor(nn.Module):
    # Actor : a_t ~ pi_theta(a_t | s_t)

    """Policy network that outputs action distributions"""
    
    def __init__(self, feature_dim: int, action_dim: int, config: DreamerConfig,
                 action_type: str = 'continuous'):
        super().__init__()
        self.action_dim = action_dim
        self.action_type = action_type
        
        if action_type == 'continuous':
            # Output mean and log_std for continuous actions
            self.network = MLPNetwork(
                feature_dim, 2 * action_dim, 
                config.hidden_size, config.num_layers, config.activation
            )
            self.raw_init_std = np.log(np.exp(5.0) - 1)  # Softplus inverse of 5.0
        else:
            # Output logits for discrete actions
            self.network = MLPNetwork(
                feature_dim, action_dim,
                config.hidden_size, config.num_layers, config.activation
            )
    
    def forward(self, features: torch.Tensor, deterministic: bool = False):
        """
        Args:
            features: Model state features [B, T, D] or [B, D]
            deterministic: If True, return mean/mode instead of sampling
        
        Returns:
            actions: Sampled actions
            dist: Action distribution
        """
        out = self.network(features)
        
        if self.action_type == 'continuous':
            mean, std = torch.chunk(out, 2, dim=-1)
            std = F.softplus(std + self.raw_init_std) + 0.01
            dist = Independent(Normal(mean, std), 1)
            
            if deterministic:
                action = mean
            else:
                action = dist.rsample()
            action = torch.tanh(action)  # Bound actions to [-1, 1]
            
        else:  # discrete
            dist = Categorical(logits=out)
            if deterministic:
                action = torch.argmax(out, dim=-1)
            else:
                action = dist.sample()
        
        return action, dist

