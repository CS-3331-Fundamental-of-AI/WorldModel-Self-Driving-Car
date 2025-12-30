import torch
import torch.nn as nn
        
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
         # pixel_values: [B, C, H, W] OR [B, T, C, H, W]
        is_sequence = False
        if pixel_values.dim() == 5:
            is_sequence = True
            B, T, C, H, W = pixel_values.shape
            x_flat = pixel_values.reshape(B*T, C, H, W)  # flatten for encoder
        elif pixel_values.dim() == 4:
            B, C, H, W = pixel_values.shape
            x_flat = pixel_values
        else:
            raise ValueError(f"pixel_values must be 4D or 5D, got {pixel_values.dim()}")

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
        if is_sequence:
            z_proj = z_proj.view(B, T, z_proj.shape[1], z_proj.shape[2])  # [B, T, N, D]
            B_flat, N, D = B*T, z_proj.shape[2], z_proj.shape[3]
        else:
            B_flat, N, D = z_proj.shape
            
        H_grid, W_grid = self.grid_h, self.grid_w
        N_target = H_grid * W_grid
        if N != N_target:
            z_proj_reshaped = z_proj.view(B_flat, N, D).transpose(1,2)  # [B_flat, D, N]
            z_proj_reshaped = torch.nn.functional.adaptive_avg_pool1d(z_proj_reshaped, N_target)
            z_proj_reshaped = z_proj_reshaped.transpose(1,2)  # [B_flat, N_target, D]
        else:
            z_proj_reshaped = z_proj.view(B_flat, N, D)
        
        # -----------------------------
        # 4) Spatial predictor
        # -----------------------------
        x_grid = z_proj_reshaped.transpose(1,2).reshape(B_flat, D, H_grid, W_grid)
        delta = self.predictor(x_grid)
        z_hat = delta.reshape(B_flat, D, N_target).transpose(1,2)  # [B_flat, N_target, D]

        # -----------------------------
        # 5) Back to tokens
        # -----------------------------
        if is_sequence:
            z_hat = z_hat.view(B, T, N_target, D)
            z_proj_reshaped = z_proj_reshaped.view(B, T, N_target, D)

        return z_hat, z_proj #z_hat = s_c