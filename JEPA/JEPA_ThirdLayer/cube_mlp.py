import torch
import torch.nn as nn
from typing import Optional

class CubeMLP(nn.Module):
    """
    CubeMLP: axis-wise MLP mixing over (L x M x D)
      - x_tokens: [B, L, D]   (temporal tokens)
      - y_modal : [B, M, D]   (modalities / context)
      - z_chans : [B, M, D]   (channel info per modality) optional
    Returns:
      - out: [B, out_dim]  (fused target embedding)
    """

    def __init__(self, L: int, M: int, D: int, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.L, self.M, self.D = L, M, D

        # Axis mixers: they mix *across* the axis dimension.
        # For mixing across L (token axis) we treat each channel vector of length L.
        self.mix_L = nn.Sequential(
            nn.Linear(L, hidden),
            nn.GELU(),
            nn.Linear(hidden, L),
        )

        # For mixing across M (modalities) we treat each channel vector of length M.
        self.mix_M = nn.Sequential(
            nn.Linear(M, hidden),
            nn.GELU(),
            nn.Linear(hidden, M),
        )

        # For mixing across D (channels) we treat each token/modality vector of length D.
        self.mix_D = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Linear(hidden, D),
        )

        # Final projection from fused channel-dim to out_dim
        self.out = nn.Linear(D, out_dim)

    def forward(
        self,
        x_tokens: torch.Tensor,            # [B, L, D]
        y_modal: torch.Tensor,             # [B, M, D]
        z_channels: Optional[torch.Tensor] = None,  # [B, M, D] (optional)
    ):
        B = x_tokens.shape[0]
        device = x_tokens.device

        if z_channels is None:
            z_channels = torch.zeros_like(y_modal, device=device)

        # ---------- 1) L-axis mixing ----------
        # We want to mix across L for each (B, D) pair, so transpose to [B, D, L],
        # apply mixer (linear on last dim), then transpose back.
        x_L = self.mix_L(x_tokens.transpose(1, 2)).transpose(1, 2)  # [B, L, D]

        # ---------- 2) M-axis mixing ----------
        y_M = self.mix_M(y_modal.transpose(1, 2)).transpose(1, 2)   # [B, M, D]
        z_M = self.mix_M(z_channels.transpose(1, 2)).transpose(1, 2) # [B, M, D]

        # ---------- 3) D-axis mixing ----------
        x_D = self.mix_D(x_L)  # [B, L, D]
        y_D = self.mix_D(y_M)  # [B, M, D]
        z_D = self.mix_D(z_M)  # [B, M, D]

        # ---------- 4) Pooling -> get per-axis summaries ----------
        # mean-pool tokens and modal slots to obtain channel-level vectors
        x_repr = x_D.mean(dim=1)  # [B, D]
        y_repr = y_D.mean(dim=1)  # [B, D]
        z_repr = z_D.mean(dim=1)  # [B, D]

        # ---------- 5) Attention-like fusion ----------
        # combine additive + multiplicative terms (lightweight attention-like)
        fused = x_repr + y_repr + z_repr + x_repr * y_repr + x_repr * z_repr + y_repr * z_repr

        # ---------- 6) Output projection ----------
        return self.out(fused)  # [B, out_dim]
