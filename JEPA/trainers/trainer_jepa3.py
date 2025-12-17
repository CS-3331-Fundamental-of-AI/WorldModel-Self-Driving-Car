# trainers/trainer_jepa3.py
import torch
from JEPA_ThirdLayer.losses import (
    inverse_affordance_losses,
    global_encoding_losses
)
from config.config import CLIP_NORM

class JEPA3Trainer:
    def __init__(self, inv, glob, optimizer):
        self.inv = inv
        self.glob = glob
        self.opt = optimizer

    def step(self, action, spatial_x, s_c, s_tg, graph=None):
        inv_out = self.inv(action, spatial_x)
        glob_out = self.glob(
            inv_out["s_tg_hat"],
            s_c,
            *graph if graph else (None, None)
        )

        loss_inv = inverse_affordance_losses(inv_out, s_tg)
        loss_glob = global_encoding_losses(glob_out)

        loss_total = loss_inv["total"] + loss_glob["total"]

        self.opt.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.inv.parameters(), CLIP_NORM)
        torch.nn.utils.clip_grad_norm_(self.glob.parameters(), CLIP_NORM)
        self.opt.step()

        return {
            "loss": loss_total.detach(),
            "loss_inv": loss_inv["total"].detach(),
            "loss_glob": loss_glob["total"].detach(),
        }

