import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """
    CNN backbone + CLIP-like projection.
    This supports future CLIP distillation by replacing fc_clip with a trained head.
    """
    def __init__(self, latent_dim=256, clip_dim=512):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_latent = nn.Linear(128, latent_dim)
        self.fc_clip = nn.Linear(128, clip_dim)

    def forward(self, img):
        feat = self.cnn(img).squeeze(-1).squeeze(-1)
        latent = self.fc_latent(feat)
        clip_vec = F.normalize(self.fc_clip(feat), dim=-1)
        return latent, clip_vec
