import torch
import torch.nn as nn

class ActionEncoder(nn.Module):
    """
    Encodes continuous control vector (steer, accel, brake).
    """
    def __init__(self, action_dim=3, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, a):
        return self.net(a)
