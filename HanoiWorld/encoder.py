from click import Path
import torch
import torch.nn as nn
from transformers import AutoModel
import sys
import os
import torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent  # HanoiWorld's parent
sys.path.append(str(ROOT_DIR))
JEPA_DIR = Path(__file__).parent.parent / "JEPA"
sys.path.append(str(JEPA_DIR))
from JEPA.jepa_encoder import JEPA_Encoder

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()  # loads .env from project root if present

CKPT_ROOT = os.getenv("JEPA_CKPT_ROOT")

if CKPT_ROOT is None:
    raise RuntimeError(
        "JEPA_CKPT_ROOT is not set. "
        "Please define it in .env or environment variables."
    )


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
        root = Path(root)

        assert root.exists(), f"JEPA ckpt root not found: {root}"

        ckpt1_path = root / "jepa1_final.pt"
        ckpt2_path = root / "jepa2_final.pt"
        ckpt3_path = root / "jepa3_final.pt"

        for p in [ckpt1_path, ckpt2_path, ckpt3_path]:
            assert p.exists(), f"Missing checkpoint: {p}"

        ckpt1 = torch.load(ckpt1_path, map_location="cpu")
        ckpt2 = torch.load(ckpt2_path, map_location="cpu")
        ckpt3 = torch.load(ckpt3_path, map_location="cpu")

        self.encoder.jepa1.load_state_dict(ckpt1["state"], strict=False)

        self.encoder.jepa2_phys.load_state_dict(
            ckpt2["state"]["pa"], strict=False
        )
        self.encoder.jepa2_inv.load_state_dict(
            ckpt2["state"]["ia"], strict=False
        )

        self.encoder.jepa3.load_state_dict(ckpt3["state"], strict=False)

        print("✅ Loaded JEPA-1, JEPA-2 (PA + IA), JEPA-3 checkpoints")
    # ======================================================
    # Forward
    # ======================================================
    def forward(self, x):
        """
        x: torch.Tensor
        Either:
        - 5D: [B, T, H, W, C] (sequqences)
        - 4D: [B, H, W, C] (single image)
        Returns:
        world_latent: [B, T, out_dim] or [B, 1, out_dim]
        """
        B = x.size(0)
        device = x.device
        print("Encoder input shape:", x.shape)

        # -------------------------------
        # Convert input to [B, T, C, H, W] for V-JEPA
        # -------------------------------
        if x.dim() == 5:  # [B, T, H, W, C]
            x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x_flat = x  # Keep 5D, pass directly
        elif x.dim() == 4:  # single images
            if x.shape[1] not in (1, 3):
                x = x.permute(0, 3, 1, 2)
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
            B, T, C, H, W = x.shape
            x_flat = x
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # Normalize if needed
        if x_flat.dtype == torch.uint8:
            x_flat = x_flat.float() / 255.0
        print("Encoder input pixel_values shape:", x_flat.shape)
        
        #-------------------------------
        # Downsample to match JEPA-1 grid size
        # -------------------------------
        # Flatten batch + time for pooling
        B, T, C, H, W = x_flat.shape
        x_flat_reshaped = x_flat.view(B*T, C, H, W)

        # Resize to something small the encoder expects (e.g., 16x16)
        x_flat_reshaped = F.adaptive_avg_pool2d(x_flat_reshaped, (16, 16))

        # Restore batch + time
        x_flat = x_flat_reshaped.view(B, T, C, x_flat_reshaped.shape[-2], x_flat_reshaped.shape[-1])
        print("Downsampled pixel_values shape:", x_flat.shape)
        
        # --------------------------------------------------
        # Minimal dummy inputs (RSSM inference)
        # --------------------------------------------------
        traj    = torch.zeros(B, T, 256, 6, device=device)
        adj     = torch.zeros(B, T, 13, 13, device=device)
        x_graph = torch.zeros(B, T, 13, 13, device=device)
        action  = torch.zeros(B, T, 2, device=device)


        # --------------------------------------------------
        # JEPA forward
        # --------------------------------------------------
        with torch.no_grad():
            out = self.encoder(
                pixel_values=x_flat,
                traj=traj,
                adj=adj,
                x_graph=x_graph,
                action=action,
                global_nodes=None,
                global_edges=None,
            )

        world_latent_flat = out["world_latent"]  # [B*T, 128]
        world_latent = world_latent_flat.view(B, T, -1)  # restore sequence: [B, T, 128]

        return self.proj(world_latent)  # [B, T, out_dim]

