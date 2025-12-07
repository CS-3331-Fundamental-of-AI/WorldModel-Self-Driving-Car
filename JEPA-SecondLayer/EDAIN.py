import torch
import torch.nn as nn
import torch.nn.functional as F

class EDAINLayer(nn.Module):
    """
    Extended Deep Adaptive Input Normalization (stable for trajectory signals).
    Behaves like BatchNorm1d for temporal inputs: [B, T, D].
    """
    def __init__(self, D, summary_dim=None):
        super().__init__()

        S = summary_dim or D

        # -----------------------------------------
        # 1. Learnable outlier suppression
        # -----------------------------------------
        # alpha: how much to suppress (0=no suppression, 1=full)
        self.outlier_alpha = nn.Parameter(torch.zeros(D))  
        # beta: slope of shrinkage
        self.outlier_beta  = nn.Parameter(torch.zeros(D))  

        # -----------------------------------------
        # 2. Summary feature extractor
        # -----------------------------------------
        self.summary_mlp = nn.Sequential(
            nn.Linear(D, S),
            nn.ReLU(),
            nn.Linear(S, D)
        )

        # -----------------------------------------
        # 3. Adaptive shift (initial = 0 → identity)
        # -----------------------------------------
        self.shift_layer = nn.Linear(D, D, bias=True)
        nn.init.zeros_(self.shift_layer.weight)
        nn.init.zeros_(self.shift_layer.bias)

        # -----------------------------------------
        # 4. Adaptive scale (initial = 1 → identity)
        # -----------------------------------------
        self.scale_layer = nn.Linear(D, D, bias=True)
        nn.init.zeros_(self.scale_layer.weight)
        nn.init.zeros_(self.scale_layer.bias)

        # -----------------------------------------
        # 5. Power transform (initial = no transform)
        # -----------------------------------------
        self.power_alpha = nn.Parameter(torch.zeros(D))

        self.eps = 1e-5

    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape

        # ================================================================
        # (1) OUTLIER MITIGATION
        # ================================================================
        batch_mean = x.mean(dim=1, keepdim=True)       # [B, 1, D]
        centered = x - batch_mean

        # Learnable soft parameters
        alpha = torch.sigmoid(self.outlier_alpha)       # (0..1)
        beta  = torch.tanh(self.outlier_beta) * 3.0     # (-3..3)

        # Nonlinear shrinkage for outlier-resistant center
        shrink = torch.tanh(centered * beta)

        # Blend original + shrunk
        x = alpha * shrink + (1 - alpha) * x

        # ================================================================
        # (2) SUMMARY ADAPTATION (like LayerNorm learned stats)
        # ================================================================
        summary = x.mean(dim=1)                         # [B, D]
        summary_feat = self.summary_mlp(summary)        # [B, D]

        # ================================================================
        # (3) ADAPTIVE SHIFT
        # ================================================================
        # Bound to avoid exploding offset
        shift = torch.tanh(self.shift_layer(summary_feat)) * 3.0
        x = x - shift.unsqueeze(1)                      # broadcast to [B,T,D]

        # ================================================================
        # (4) ADAPTIVE SCALE (BatchNorm-like)
        # ================================================================
        batch_std = x.std(dim=1) + self.eps             # [B, D]

        raw_scale = self.scale_layer(batch_std)         # [B, D]
        scale = torch.exp(torch.clamp(raw_scale, -3, 3))  # scale in [0.05 .. 20]

        x = x / (scale.unsqueeze(1) + self.eps)

        # ================================================================
        # (5) POWER TRANSFORM (robust nonlinear compression)
        # ================================================================
        pa = torch.tanh(self.power_alpha) * 0.25        # |alpha| ≤ 0.25
        x = torch.sign(x) * (torch.abs(x) ** (1 + pa))

        return x