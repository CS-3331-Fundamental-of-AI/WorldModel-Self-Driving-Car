import torch
import torch.nn as nn

from JEPA.jepa_encoder import JEPA_Encoder


class FrozenEncoder(nn.Module):
    """
    Encoder wrapper that replaces ResNet with your full JEPA encoder.
    Only outputs a fixed-size embedding for RSSM (default 128 dim).
    """

    def __init__(self, out_dim=128, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")

        # -----------------------
        # Use full JEPA instead of ResNet
        # -----------------------
        self.encoder = JEPA_Encoder().to(self.device)

        # JEPA world_latent is 128, but let's keep general
        self.proj = nn.Linear(128, out_dim).to(self.device)

        # No freezing (unless you want)
        self.out_dim = out_dim
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        x: (B, C, H, W) — standard env frame.
        """
        if x.dim() == 4 and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2)

        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        B = x.size(0)

        # -------------------------------------------------
        # JEPA expects masked/unmasked images + masks
        # For RSSM, we treat input frame as "unmasked"
        # and the masked_img = unmasked_img (identity)
        # -------------------------------------------------
        masked_img = x
        unmasked_img = x
        mask_empty = torch.zeros(B, 1, 1, 1, device=x.device)
        mask_non = torch.zeros(B, 1, 1, 1, device=x.device)
        mask_any = torch.ones(B, 1, 1, 1, device=x.device)

        # Minimal dummy inputs for tier2 + tier3
        traj = torch.zeros(B, 6, 256, device=x.device)
        adj = torch.zeros(B, 13, 13, device=x.device)
        x_graph = torch.zeros(B, 13, 13, device=x.device)
        action = torch.zeros(B, 2, device=x.device)

        outputs = self.encoder(
            masked_img,
            unmasked_img,
            mask_empty,
            mask_non,
            mask_any,
            traj,
            adj,
            x_graph,
            action,
        )

        # Final world latent from JEPA-3
        world_latent = outputs["world_latent"]  # (B, 128)

        # Map to RSSM embedding dim
        emb = self.proj(world_latent)
        return emb
