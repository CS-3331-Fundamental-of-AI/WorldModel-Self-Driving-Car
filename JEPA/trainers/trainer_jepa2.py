# trainers/trainer_jepa2.py

import torch
import torch.nn.functional as F

from JEPA_SecondLayer.inverse_affordance.utils import update_ema
from config.config import (
    CLIP_NORM,
    EMA_JEPA2,
    LAMBDA_JEPA2,
    LAMBDA_INV,
)


class JEPA2Trainer:
    """
    JEPA-2:
    - PA = student (backprop)
    - IA = online learner (aux loss)
    - IA_ema = teacher (EMA of IA)

    Losses:
    - PA loss:   D(s_tg, s_y_ema)
    - IA loss:   D(s_y, s_tg.detach())
    """

    def __init__(
        self,
        pa_model,
        ia_model,
        ia_ema_model,
        optimizer,
        dist_fn=None,
    ):
        self.pa = pa_model
        self.ia = ia_model
        self.ia_ema = ia_ema_model
        self.opt = optimizer

        # distance function between latents
        self.dist_fn = dist_fn or F.mse_loss

        # freeze EMA teacher
        for p in self.ia_ema.parameters():
            p.requires_grad = False
        self.ia_ema.eval()

    def step(
        self,
        traj,
        x_graph,
        adj,
        action,
        s_c,
        traj_mask=None,
        graph_mask=None,
    ):
        # --------------------------------------------------
        # 1. PA forward (student, grounded in data)
        # --------------------------------------------------
        pa_out = self.pa(
            traj=traj,
            adj=adj,
            x_graph=x_graph,
            traj_mask=traj_mask,
            graph_mask=graph_mask,
        )
        s_tg = pa_out["fusion"]          # [B, D]

        # --------------------------------------------------
        # 2. IA EMA forward (teacher, no gradients)
        # --------------------------------------------------
        with torch.no_grad():
            ia_ema_out = self.ia_ema(action, s_c)
            s_y_ema = ia_ema_out["s_y"]

        # --------------------------------------------------
        # 3. PA loss (PA ← IA_ema)
        # --------------------------------------------------
        loss_pa = self.dist_fn(s_tg, s_y_ema)

        # --------------------------------------------------
        # 4. IA online forward (IA ← PA.detach)
        # --------------------------------------------------
        ia_out = self.ia(action, s_c)
        s_y = ia_out["s_y"]

        loss_ia = self.dist_fn(s_y, s_tg.detach())

        # --------------------------------------------------
        # 5. Total loss (weighted)
        # --------------------------------------------------
        loss = (
            LAMBDA_JEPA2 * loss_pa
            + LAMBDA_INV * loss_ia
        )

        # --------------------------------------------------
        # 6. Optimize PA + IA
        # --------------------------------------------------
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.pa.parameters(), CLIP_NORM)
        torch.nn.utils.clip_grad_norm_(self.ia.parameters(), CLIP_NORM)

        self.opt.step()

        # --------------------------------------------------
        # 7. EMA update (IA → IA_ema)
        # --------------------------------------------------
        update_ema(self.ia_ema, self.ia, EMA_JEPA2)

        return {
            "loss": loss.detach(),
            "loss_pa": loss_pa.detach(),
            "loss_ia": loss_ia.detach(),
            "s_tg": s_tg.detach(),
            "s_y": s_y.detach(),
        }
