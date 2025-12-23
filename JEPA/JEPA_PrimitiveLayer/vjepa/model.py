import copy
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
class TokenProjector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, z):
        return self.proj(z)

class SpatialPredictorCNN(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

    def forward(self, x):
        return self.conv(x)
class PrimitiveLayerJEPA(nn.Module):
    def __init__(
        self,
        encoder,        # frozen V-JEPA-2 encoder
        grid_h=16,
        grid_w=16,
        enc_dim=1024,
        prim_dim=128
    ):
        super().__init__()
        self.encoder = encoder
        self.grid_h = grid_h
        self.grid_w = grid_w

        # 🔑 Projection: 1024 → 128
        self.project = nn.Linear(enc_dim, prim_dim)

        # Spatial predictor
        self.predictor = SpatialPredictorCNN(embed_dim=prim_dim)

    def forward(self, pixel_values):
        # -----------------------------
        # 1) Encoder (frozen)
        # -----------------------------
        with torch.no_grad():
            z = self.encoder(
                pixel_values_videos=pixel_values
            ).last_hidden_state  # [B, N, 1024]

        # -----------------------------
        # 2) Project to primitive dim
        # -----------------------------
        z = z.float()                    # FP32 safety
        z_proj = self.project(z)         # [B, N, 128]

        # -----------------------------
        # 3) Reshape to grid
        # -----------------------------
        B, N, D = z_proj.shape
        x = z_proj.transpose(1, 2).reshape(
            B, D, self.grid_h, self.grid_w
        )

        # -----------------------------
        # 4) Spatial predictor
        # -----------------------------
        delta = self.predictor(x)

        # -----------------------------
        # 5) Back to tokens
        # -----------------------------
        z_hat = delta.reshape(B, D, N).transpose(1, 2)

        return z_hat, z_proj