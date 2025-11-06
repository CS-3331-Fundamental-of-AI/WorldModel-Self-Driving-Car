import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryEncoder(nn.Module):
    """
    Encodes trajectory: (x, y, heading) over time.
    CNN captures short-term motion.
    Transformer captures long-term temporal dependencies.
    """
    def __init__(self, input_dim=3, cnn_dim=64, embed_dim=128, latent_dim=256):
        super().__init__()

        # 1D CNN over time axis
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(embed_dim, latent_dim)

    def forward(self, traj):
        # traj: [B, T, 3]
        x = traj.transpose(1, 2)  # → [B, 3, T]
        h = self.cnn(x).transpose(1, 2)  # → [B, T, embed_dim]
        h_t = h.transpose(0, 1)          # → [T, B, embed_dim]
        h_t = self.transformer(h_t)
        h_final = h_t.mean(dim=0)        # [B, embed_dim]
        return self.fc(h_final)
