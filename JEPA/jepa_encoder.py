import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel

from JEPA_PrimitiveLayer.vjepa.model import PrimitiveLayerJEPA
from JEPA.latent_expander import LatentExpander   # adjust import if needed


class FrozenJEPA1VisionEncoder(nn.Module):
    """
    Unified Frozen JEPA-1 Vision Encoder

    ✔ Loads V-JEPA2 ViT-L backbone internally
    ✔ Uses PrimitiveLayerJEPA only
    ✔ Fully frozen (eval-only)
    ✔ Handles single-frame & temporal inputs
    ✔ Normalizes and reshapes inputs safely
    ✔ RSSM-safe latent output

    Output:
        - [B, D]        single frame
        - [B, T, D]     temporal
    """

    def __init__(
        self,
        out_dim: int = 128,
        device=None,
        ckpt_root=None,
    ):
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")

        # --------------------------------------------------
        # Vision backbone (must match JEPA-1 training)
        # --------------------------------------------------
        backbone = AutoModel.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256",
            torch_dtype=torch.float16,
        )
        backbone.to(self.device)
        backbone.eval()

        

        for p in backbone.parameters():
            p.requires_grad = False

        vision_encoder = backbone.encoder

        # --------------------------------------------------
        # JEPA-1 Primitive Layer
        # --------------------------------------------------
        self.jepa1 = PrimitiveLayerJEPA(
            encoder=vision_encoder,
            grid_h=16,
            grid_w=16,
            enc_dim=1024,
            prim_dim=out_dim,
        ).to(self.device)

        # --------------------------------------------------
        # Load JEPA-1 checkpoint (optional but recommended)
        # --------------------------------------------------
        if ckpt_root is not None:
            self._load_checkpoint_jepa1(ckpt_root)

        # --------------------------------------------------
        # RSSM safety
        # --------------------------------------------------
        self.latent_expander = LatentExpander(expected_dim=out_dim)

        # --------------------------------------------------
        # Freeze everything
        # --------------------------------------------------
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    # ======================================================
    # Checkpoint loader
    # ======================================================
    def _load_checkpoint_jepa1(self, root):
        root = Path(root)
        assert root.exists(), f"JEPA ckpt root not found: {root}"

        ckpt = torch.load(root / "jepa1_final.pt", map_location="cpu")
        self.jepa1.load_state_dict(ckpt["state"], strict=False)

        print("Loaded JEPA-1 checkpoint.")

    # ======================================================
    # Forward
    # ======================================================
    @torch.no_grad()
    def forward(self, pixel_values):
        """
        Args:
            pixel_values:
                [B,H,W,3]
                [B,T,H,W,3]
                [B,3,T,H,W]
                [B,C,H,W]
                [B,T,C,H,W]

        Returns:
            embed:
                [B, D]      single-frame
                [B, T, D]   temporal
        """

        x = pixel_values

        # ----------------------------------
        # Normalize uint8
        # ----------------------------------
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # ----------------------------------
        # Canonicalize shape
        # ----------------------------------
        if x.ndim == 4:
            # [B,H,W,3] or [B,C,H,W]
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)   # → [B,3,H,W]
            x = x.unsqueeze(1)              # → [B,1,3,H,W]

        elif x.ndim == 5:
            # [B,T,H,W,3]
            if x.shape[-1] == 3:
                x = x.permute(0, 1, 4, 2, 3)  # → [B,T,3,H,W]
            # [B,3,T,H,W]
            elif x.shape[1] == 3:
                x = x.permute(0, 2, 1, 3, 4)  # → [B,T,3,H,W]

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        B, T, C, H, W = x.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        # ----------------------------------
        # JEPA-1 forward
        # ----------------------------------
        # x = x.reshape(B * T, C, H, W)
        # print(f"THE SHAPE of x - debug in FrozenJEPA1VisionEncoder {x.shape}")
        tokens, _ = self.jepa1(x)           # [B*T, N, D] # it's need 5
        embed = tokens.mean(dim=1)          # [B*T, D]
        embed = embed.view(B, T, -1)        # [B,T,D]
        # print(f"THE SHAPE of embed- debug in FrozenJEPA1VisionEncoder before reshape {embed.shape}")


        # ----------------------------------
        # RSSM safety
        # ----------------------------------
        embed = self.latent_expander(embed)
        # print(f"THE SHAPE of embed 2- debug in FrozenJEPA1VisionEncoder before reshape {embed.shape}")

        # ----------------------------------
        # Return shape
        # ----------------------------------
        # if T == 1:
        #     return embed[:, 0]              # [B,D]
        # print(f"THE SHAPE of embed 3- debug in FrozenJEPA1VisionEncoder before reshape {embed.shape}")

        return embed                         # [B,T,D]