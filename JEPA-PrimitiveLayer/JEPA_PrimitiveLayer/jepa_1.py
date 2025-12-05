import torch
import torch.nn as nn
import torch.nn.functional as F
from .bev_jepa import BEVJEPAEncoder2D
from .spatial_pred import SpatialPredictorCNN
from Utils.ema_buffer import init_target_from_online, LatentBuffer
from config import EMA_DECAY
import torch
import torch.nn as nn
from torchvision.models import resnet50
import os

def load_dino_resnet50():
    model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

class PrimitiveLayer(nn.Module):
    def __init__(self, embed_dim=128, ema_decay=EMA_DECAY, distilled_path: str = "bev_mobilenet_dino_init.pt"):
        super().__init__()

        self.context_encoder = BEVJEPAEncoder2D(width_mult=0.5)
        self.target_encoder  = BEVJEPAEncoder2D(width_mult=0.5)
        
        init_target_from_online(self.context_encoder, self.target_encoder)

        # 2) Load distilled MobileNet weights (if available)
        if distilled_path is not None and os.path.exists(distilled_path):
            state = torch.load(distilled_path, map_location="cpu")
            missing, unexpected = self.context_encoder.load_state_dict(state, strict=False)
            print(f"[Distill] Loaded distilled encoder from {distilled_path}")
            print(f"[Distill] Missing: {missing}, Unexpected: {unexpected}")
        else:
            print(f"[Distill] No distilled weights found at {distilled_path}, "
                  f"training JEPA from random init.")


        D = self.context_encoder.out_dim
        self.predictor = SpatialPredictorCNN(embed_dim=D)

        # ✅ Zhu-style tokens (learnable)
        self.mask_token  = nn.Parameter(torch.zeros(1, D))
        self.empty_token = nn.Parameter(torch.zeros(1, D))

        self.ema_decay = ema_decay
        self.buffer = LatentBuffer(embed_dim=D, ema_decay=ema_decay)

    def _inject_tokens_context(self, z_c_raw, mask_empty_lat, mask_any_lat):
      z = z_c_raw.clone()
      B, HW, D = z.shape

      # Flatten
      mask_any_lat   = mask_any_lat.reshape(B, HW).bool()
      mask_empty_lat = mask_empty_lat.reshape(B, HW).bool()

      # ===========================
      #  IF CUDA → AMP-safe version
      # ===========================
      if torch.cuda.is_available():
          tok_dtype = z.dtype
          mask_tok  = self.mask_token.to(tok_dtype)
          empty_tok = self.empty_token.to(tok_dtype)

          if mask_any_lat.any():
              num_any = int(mask_any_lat.sum().item())
              z[mask_any_lat] = mask_tok.expand(num_any, -1)

          if mask_empty_lat.any():
              num_empty = int(mask_empty_lat.sum().item())
              z[mask_empty_lat] = empty_tok.expand(num_empty, -1)

          return z

      # ===========================
      #  ELSE → ORIGINAL VERSION
      # ===========================
      if mask_any_lat.any():
          num_any = int(mask_any_lat.sum().item())
          z[mask_any_lat] = self.mask_token.expand(num_any, -1)

      if mask_empty_lat.any():
          num_empty = int(mask_empty_lat.sum().item())
          z[mask_empty_lat] = self.empty_token.expand(num_empty, -1)

      return z

    def _inject_tokens_target(self, z_t_raw, mask_empty_lat):
      z = z_t_raw.clone()
      B, HW, D = z.shape

      mask_empty_lat = mask_empty_lat.reshape(B, HW).bool()

      # ===========================
      #  IF CUDA → AMP-safe version
      # ===========================
      if torch.cuda.is_available():
          tok_dtype = z.dtype
          empty_tok = self.empty_token.to(tok_dtype)

          if mask_empty_lat.any():
              num_empty = int(mask_empty_lat.sum().item())
              z[mask_empty_lat] = empty_tok.expand(num_empty, -1)

          return z

      # ===========================
      #  ELSE → ORIGINAL VERSION
      # ===========================
      if mask_empty_lat.any():
          num_empty = int(mask_empty_lat.sum().item())
          z[mask_empty_lat] = self.empty_token.expand(num_empty, -1)

      return z

    def _distill_from_teacher(self):
        """
        Initialize context encoder weights using the DINO teacher encoder.
        Only layers with matching shapes are copied.
        """

        teacher_state = self.teacher.state_dict()
        student_state = self.context_encoder.state_dict()

        copied = 0
        for k_s, v_s in student_state.items():
            if k_s in teacher_state and teacher_state[k_s].shape == v_s.shape:
                student_state[k_s] = teacher_state[k_s]
                copied += 1

        self.context_encoder.load_state_dict(student_state, strict=False)
        print(f"[Variation A] Copied {copied} matching weights from teacher → student.")
      
    def forward(self, masked_img, unmasked_img,
            mask_empty_lat, mask_non_lat, mask_any_lat):
        """
        masked_img:   (B,3,H,W)
        unmasked_img: (B,3,H,W)
        mask_*_lat:   (B, Hc*Wc)
        """

        # ---- FORCE dtype/device consistency (BF16-safe) ----
        dtype = next(self.context_encoder.parameters()).dtype
        device = next(self.context_encoder.parameters()).device

        masked_img   = masked_img.to(device=device, dtype=dtype)
        unmasked_img = unmasked_img.to(device=device, dtype=dtype)

        mask_empty_lat = mask_empty_lat.to(device=device)
        mask_any_lat   = mask_any_lat.to(device=device)
        mask_non_lat   = mask_non_lat.to(device=device)

        # tokens must match dtype too
        self.mask_token  = self.mask_token.to(dtype=dtype)
        self.empty_token = self.empty_token.to(dtype=dtype)

        # ----------------------------------------------------
        # 1) Context encoder output normalization (Option B)
        # ----------------------------------------------------
        out_c = self.context_encoder(masked_img)

        # Encoder may return either:
        #   (z, (H,W))  OR  z only
        if isinstance(out_c, tuple):
            feat_c, (Hc, Wc) = out_c
        else:
            feat_c = out_c
            _, _, Hc, Wc = feat_c.shape

        # Flatten → (B, HW, C)
        B, Cc, Hc, Wc = feat_c.shape
        z_c_raw = feat_c.reshape(B, Cc, Hc * Wc).permute(0, 2, 1)

        # ----------------------------------------------------
        # 2) Inject tokens for context branch
        # ----------------------------------------------------
        z_c = self._inject_tokens_context(z_c_raw, mask_empty_lat, mask_any_lat)

        # ----------------------------------------------------
        # 3) Target encoder output normalization (Option B)
        # ----------------------------------------------------
        out_t = self.target_encoder(unmasked_img)

        if isinstance(out_t, tuple):
            feat_t, _ = out_t
        else:
            feat_t = out_t

        Bt, Ct, Ht, Wt = feat_t.shape
        z_t_raw = feat_t.reshape(Bt, Ct, Ht * Wt).permute(0, 2, 1)

        # ----------------------------------------------------
        # 4) Inject empty tokens for target
        # ----------------------------------------------------
        z_t = self._inject_tokens_target(z_t_raw, mask_empty_lat)

        # ----------------------------------------------------
        # 5) Predictor: expects (B, HW, C)
        # ----------------------------------------------------
        s_c = self.predictor(z_c, Hc, Wc)

        return z_c, s_c, z_t