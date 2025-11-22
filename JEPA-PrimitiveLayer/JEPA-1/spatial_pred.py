import torch
import torch.nn as nn

class SpatialPredictorCNN(nn.Module):
    """Predict token embeddings from token grid"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )

    def forward(self, z_tokens, h, w):
        B, N, D = z_tokens.shape
        x = z_tokens.transpose(1, 2).reshape(B, D, h, w)
        x = self.conv(x)
        x = x.reshape(B, D, h * w).transpose(1, 2)
        return x
