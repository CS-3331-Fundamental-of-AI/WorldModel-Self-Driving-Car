import torch
import torch.nn as nn

class LatentExpander(nn.Module):
    """
    Ensures world_latent has expected semantic dimension.
    Identity if already correct.
    """

    def __init__(self, expected_dim: int):
        super().__init__()
        self.expected_dim = expected_dim
        self.proj = None  # lazy init

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, T, D] or [B, D]
        returns: [B, T, expected_dim]
        """

        # Normalize shape
        if z.ndim == 2:
            z = z.unsqueeze(1)  # [B, 1, D]

        assert z.ndim == 3, f"Expected [B,T,D], got {z.shape}"

        B, T, D = z.shape

        # Identity path
        if D == self.expected_dim:
            return z

        # Projection path (lazy, explicit)
        if self.proj is None or self.proj.in_features != D:
            self.proj = nn.Linear(D, self.expected_dim).to(z.device)

        return self.proj(z)