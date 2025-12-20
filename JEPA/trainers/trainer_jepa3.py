# trainers/trainer_jepa3.py
import torch
from JEPA_ThirdLayer.losses import (
    inverse_affordance_losses,
    global_encoding_losses
)
from JEPA_ThirdLayer.utils import EMAHelper, freeze
from config.config import CLIP_NORM

class JEPA3Trainer:
    def __init__(
        self,
        inv,            # student inverse model
        glob,           # student global model
        inv_tgt,        # EMA target inverse
        glob_tgt,       # EMA target global
        optimizer,
        ema_decay=0.999
    ):
        self.inv = inv
        self.glob = glob
        self.inv_tgt = inv_tgt
        self.glob_tgt = glob_tgt
        self.opt = optimizer
        
        # -------------------------
        # Freeze EMA targets
        # -------------------------
        freeze(self.inv_tgt)
        freeze(self.glob_tgt)
        self.inv_tgt.eval()
        self.glob_tgt.eval()

        # -------------------------
        # EMA helpers
        # -------------------------
        self.ema_inv = EMAHelper(decay=ema_decay)
        self.ema_glob = EMAHelper(decay=ema_decay)

        self.ema_inv.register(self.inv)
        self.ema_glob.register(self.glob)

        # Initialize targets
        self.ema_inv.assign_to(self.inv_tgt)
        self.ema_glob.assign_to(self.glob_tgt)

    def step(self, action, spatial_x, s_c, s_tg, graph=None):
        # --------------------------------------------------
        # Forward (student only)
        # --------------------------------------------------
        inv_out = self.inv(action, spatial_x)

        glob_out = self.glob(
            inv_out["s_tg_hat"],
            s_c,
            *graph if graph else (None, None)
        )

        # --------------------------------------------------
        # Losses
        # --------------------------------------------------
        loss_inv = inverse_affordance_losses(inv_out, s_tg)
        loss_glob = global_encoding_losses(glob_out)

        loss_total = loss_inv["total"] + loss_glob["total"]

        # --------------------------------------------------
        # Optimization (student only)
        # --------------------------------------------------
        self.opt.zero_grad(set_to_none=True)
        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(self.inv.parameters(), CLIP_NORM)
        torch.nn.utils.clip_grad_norm_(self.glob.parameters(), CLIP_NORM)

        self.opt.step()
        
        # --------------------------------------------------
        # EMA update (NO gradients)
        # --------------------------------------------------
        self.ema_inv.update(self.inv)
        self.ema_glob.update(self.glob)

        self.ema_inv.assign_to(self.inv_tgt)
        self.ema_glob.assign_to(self.glob_tgt)

        return {
            "loss": loss_total.detach(),
            "loss_inv": loss_inv["total"].detach(),
            "loss_glob": loss_glob["total"].detach(),
        }

