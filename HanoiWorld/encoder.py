from click import Path
import torch
import torch.nn as nn
from transformers import AutoModel
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent  # HanoiWorld's parent
sys.path.append(str(ROOT_DIR))
JEPA_DIR = Path(__file__).parent.parent / "JEPA"
sys.path.append(str(JEPA_DIR))
from JEPA.jepa_encoder import JEPA_Encoder

CKPT_ROOT = "kaggle/input/5k"

class FrozenEncoder(nn.Module):
    """
    Frozen JEPA encoder (JEPA-1 + JEPA-2 + JEPA-3).
    Outputs world_latent from JEPA-3.
    """

    def __init__(
        self,
        out_dim=128,
        device=None,
        ckpt_root=CKPT_ROOT,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")

        # --------------------------------------------------
        # Vision backbone (MUST match JEPA-1 training)
        # --------------------------------------------------
        backbone = AutoModel.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256",
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        vision_encoder = backbone.encoder
        
        # --------------------------------------------------
        # JEPA Encoder (full stack)
        # --------------------------------------------------
        self.encoder = JEPA_Encoder(
            vision_encoder=vision_encoder
        ).to(self.device)

        # --------------------------------------------------
        # Load pretrained checkpoints
        # --------------------------------------------------
        self._load_checkpoints(ckpt_root)

        # --------------------------------------------------
        # Optional projection (RSSM compatibility)
        # --------------------------------------------------
        self.proj = nn.Linear(128, out_dim)

        # --------------------------------------------------
        # Freeze everything
        # --------------------------------------------------
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    # ======================================================
    # Load checkpoints
    # ======================================================
    def _load_checkpoints(self, root):
        ckpt1 = torch.load(f"{root}/jepa1_final.pt", map_location="cpu")
        ckpt2 = torch.load(f"{root}/jepa2_final.pt", map_location="cpu")
        ckpt3 = torch.load(f"{root}/jepa3_final.pt", map_location="cpu")

        self.encoder.jepa1.load_state_dict(ckpt1["state"], strict=False)

        self.encoder.jepa2_phys.load_state_dict(
            ckpt2["state"]["pa"], strict=False
        )
        self.encoder.jepa2_inv.load_state_dict(
            ckpt2["state"]["ia"], strict=False
        )

        self.encoder.jepa3.load_state_dict(ckpt3["state"], strict=False)

        print("✅ Loaded JEPA-1, JEPA-2 (PA+IA), JEPA-3 checkpoints")

    # ======================================================
    # Forward
    # ======================================================
    def forward(self, x):
        """
        x: (B, C, H, W) or (B, H, W, C)
        """
        if x.dim() == 4 and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2)

        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        B = x.size(0)
        device = x.device

        # --------------------------------------------------
        # Minimal dummy inputs (RSSM inference)
        # --------------------------------------------------
        traj    = torch.zeros(B, 256, 6, device=device)
        adj     = torch.zeros(B, 13, 13, device=device)
        x_graph = torch.zeros(B, 13, 13, device=device)
        action  = torch.zeros(B, 2, device=device)

        # --------------------------------------------------
        # JEPA forward
        # --------------------------------------------------
        with torch.no_grad():
            out = self.encoder(
                pixel_values=x,
                traj=traj,
                adj=adj,
                x_graph=x_graph,
                action=action,
                global_nodes=None,
                global_edges=None,
            )

        world_latent = out["world_latent"]  # [B, 128]
        return self.proj(world_latent)
