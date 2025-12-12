import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPredictor(nn.Module):
    def __init__(self, dim_s, dim_latent, hidden_dim, out_dim):
        """
        :param dim_s       : dimension of the first input (s_c)
        :param dim_latent  : dimension of the second input (z_latent)
        :param hidden_dim  : hidden layer size
        :param out_dim     : output dimension (e.g., embedding dim you predict)
        """
        super().__init__()
        self.input_dim = dim_s + dim_latent
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, s_c, z_latent):
        """
        :param s_c       : tensor of shape (B, dim_s)
        :param z_latent  : tensor of shape (B, dim_latent)
        """
        # concatenate along feature dimension
        x = torch.cat((s_c, z_latent), dim=-1)  # shape (B, dim_s + dim_latent)
        return self.net(x)