import torch.nn as nn
import torch

class MLPNetwork(nn.Module):
    """Multi-layer perceptron with configurable activation"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 num_layers: int, activation: str = 'silu'):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU() if activation == 'silu' else nn.ReLU()
            ])
            in_dim = hidden_size
        
        layers.append(nn.Linear(hidden_size, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

