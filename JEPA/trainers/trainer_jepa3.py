# trainers/trainer_jepa3.py
import torch
from JEPA_ThirdLayer.losses import inverse_affordance_losses, global_encoding_losses
from JEPA_ThirdLayer.utils import EMAHelper, freeze
from config.config import CLIP_NORM

class JEPA3Trainer:
    def __init__(
        self,
        inv,            # student inverse model (JEPA_Tier3_InverseAffordance)
        glob,           # student global model (JEPA_Tier3_GlobalEncoding)
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

        # Freeze EMA targets
        freeze(self.inv_tgt)
        freeze(self.glob_tgt)
        self.inv_tgt.eval()
        self.glob_tgt.eval()

        # EMA helpers
        self.ema_inv = EMAHelper(decay=ema_decay)
        self.ema_glob = EMAHelper(decay=ema_decay)
        self.ema_inv.register(self.inv)
        self.ema_glob.register(self.glob)
        self.ema_inv.assign_to(self.inv_tgt)
        self.ema_glob.assign_to(self.glob_tgt)

    def step(self, action, s_c, s_tg=None, global_nodes=None, global_adj=None):
        """
        JEPA-3 training step:
        - action : [B, 2]
        - s_c    : context representation from JEPA-1 [B, C]
        - s_tg   : target representation from JEPA-2 [B, C] (optional)
        - global_nodes : list of node tensors per sample
        - global_adj   : list of edge tensors per sample
        """
        # -----------------------------
        # Forward: inverse model
        # -----------------------------
        inv_out = self.inv(action, s_c)  # s_c used internally by IA

        # -----------------------------
        # Forward: global encoding
        # -----------------------------
        # Previously used s_c_mod, now just pass original s_c
        glob_out = self.glob(
            inv_out["s_tg_hat"],  # predicted target from IA
            s_c,                  # raw context from JEPA-1
            global_nodes,
            global_adj
        )

        # -----------------------------
        # Losses
        # -----------------------------
        loss_inv = inverse_affordance_losses(inv_out, s_tg)
        loss_glob = global_encoding_losses(glob_out)
        loss_total = loss_inv["total"] + loss_glob["total"]

        # -----------------------------
        # Optimization
        # -----------------------------
        self.opt.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.inv.parameters(), CLIP_NORM)
        torch.nn.utils.clip_grad_norm_(self.glob.parameters(), CLIP_NORM)
        self.opt.step()

        # -----------------------------
        # EMA update
        # -----------------------------
        self.ema_inv.update(self.inv)
        self.ema_glob.update(self.glob)
        self.ema_inv.assign_to(self.inv_tgt)
        self.ema_glob.assign_to(self.glob_tgt)

        return {
            "loss": loss_total.detach(),
            "loss_inv": loss_inv["total"].detach(),
            "loss_glob": loss_glob["total"].detach(),
        }
