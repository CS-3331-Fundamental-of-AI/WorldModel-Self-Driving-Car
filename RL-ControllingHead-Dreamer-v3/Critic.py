
import DreamerConfig
import TwoHotEncoder
import MLPNetwork
import torch.nn as nn
import torch
import torch.functional as F

class Critic(nn.Module):
    """Value network with distributional output"""
    
    def __init__(self, feature_dim: int, config: DreamerConfig):
        super().__init__()
        self.encoder = TwoHotEncoder()
        
        self.network = MLPNetwork(
            feature_dim, self.encoder.num_bins,
            config.hidden_size, config.num_layers, config.activation
        )
        
        # Initialize output layer to zero for stability
        nn.init.zeros_(self.network.network[-1].weight)
        nn.init.zeros_(self.network.network[-1].bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict value distribution and return expected value"""
        logits = self.network(features)
        value = self.encoder.decode(logits)
        return value, logits
    
    def loss(self, features: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """Compute two-hot cross-entropy loss"""
        logits = self.network(features)
        target_encoding = self.encoder.encode(target_values)
        loss = -torch.sum(target_encoding * F.log_softmax(logits, dim=-1), dim=-1)
        return loss

