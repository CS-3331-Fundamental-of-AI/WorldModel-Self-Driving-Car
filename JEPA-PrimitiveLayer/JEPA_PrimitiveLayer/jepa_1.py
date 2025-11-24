import torch
import torch.nn as nn
import torch.nn.functional as F
from .bev_jepa import BEVJEPAEncoder2D
from .spatial_pred import SpatialPredictorCNN
from Utils.ema_buffer import init_target_from_online, LatentBuffer
from config import EMA_DECAY

class PrimitiveLayer(nn.Module):
    def __init__(self, embed_dim=128, ema_decay=EMA_DECAY):
        super().__init__()

        self.context_encoder = BEVJEPAEncoder2D(in_ch=3, base_dim=embed_dim // 8)

        self.target_encoder = BEVJEPAEncoder2D(in_ch=3, base_dim=embed_dim // 8)
        init_target_from_online(self.context_encoder, self.target_encoder)

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

      # Flatten to (B, HW)
      mask_any_lat   = mask_any_lat.reshape(B, HW).bool()
      mask_empty_lat = mask_empty_lat.reshape(B, HW).bool()

      # --- Inject masked tokens ---
      if mask_any_lat.any():
          num_any = int(mask_any_lat.sum().item())    # correct count
          z[mask_any_lat] = self.mask_token.expand(num_any, -1)

      if mask_empty_lat.any():
          num_empty = int(mask_empty_lat.sum().item())
          z[mask_empty_lat] = self.empty_token.expand(num_empty, -1)

      return z


    def _inject_tokens_target(self, z_t_raw, mask_empty_lat):
      z = z_t_raw.clone()
      B, HW, D = z.shape

      # --- Flatten and convert to bool ---
      mask_empty_lat = mask_empty_lat.reshape(B, HW).bool()

      # --- Inject empty tokens ---
      if mask_empty_lat.any():
          num_empty = int(mask_empty_lat.sum().item())
          z[mask_empty_lat] = self.empty_token.expand(num_empty, -1)

      return z

    def forward(self, masked_img, unmasked_img,
            mask_empty_lat, mask_non_lat, mask_any_lat):
        """
        masked_img:   (B,3,H,W)
        unmasked_img: (B,3,H,W)
        mask_*_lat: (B, Hc*Wc)
        """

        # ---- FORCE dtype/device consistency (BF16-safe) ----
        dtype = next(self.context_encoder.parameters()).dtype  # bf16 under autocast
        device = next(self.context_encoder.parameters()).device

        masked_img   = masked_img.to(device=device, dtype=dtype)
        unmasked_img = unmasked_img.to(device=device, dtype=dtype)

        # masks should be on device; bool is fine
        mask_empty_lat = mask_empty_lat.to(device=device)
        mask_any_lat   = mask_any_lat.to(device=device)
        mask_non_lat   = mask_non_lat.to(device=device)  # even if unused now

        # tokens must match dtype too
        mask_token  = self.mask_token.to(dtype=dtype)
        empty_token = self.empty_token.to(dtype=dtype)

        # 1) Context encoder
        z_c_raw, (Hc, Wc) = self.context_encoder(masked_img)

        # 2) Insert tokens for context
        z_c = self._inject_tokens_context(z_c_raw, mask_empty_lat, mask_any_lat)

        # 3) Target encoder
        z_t_raw, _ = self.target_encoder(unmasked_img)

        # 4) Insert empty tokens for target
        z_t = self._inject_tokens_target(z_t_raw, mask_empty_lat)

        # 5) Predictor
        s_c = self.predictor(z_c, Hc, Wc)

        return z_c, s_c, z_t