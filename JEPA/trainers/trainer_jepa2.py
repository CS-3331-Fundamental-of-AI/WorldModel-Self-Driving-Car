# trainers/trainer_jepa2.py

import torch
from JEPA_SecondLayer.losses import (
    vicreg_invariance,
    vicreg_variance,
    vicreg_covariance,
)
from JEPA_SecondLayer.utils import update_ema
from config.config import CLIP_NORM, EMA_JEPA2


class JEPA2Trainer:
    def __init__(self, model, ema_model, optimizer, aug_fn=None,
                 lambda_inv=1.0, lambda_var=1.0, lambda_cov=0.04):
        """
        JEPA-2 trainer:
        - trains fused trajectory–graph representation
        - FSQ tokenizer is frozen and NOT trained here
        """
        self.model = model
        self.ema_model = ema_model
        self.opt = optimizer
        self.aug_fn = aug_fn

        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def step(self, traj, x_graph, adj, traj_mask=None, graph_mask=None):
        # --------------------------------------------------
        # 1. Clean forward
        # --------------------------------------------------
        out_clean = self.model(
            traj=traj,
            adj=adj,
            x_graph=x_graph,
            traj_mask=traj_mask,
            graph_mask=graph_mask,
        )
        z_clean = out_clean["fusion"]  # [B, D]

        # --------------------------------------------------
        # 2. Augmented forward
        # --------------------------------------------------
        if self.aug_fn is not None:
            traj_aug = self.aug_fn(traj)
        else:
            traj_aug = traj + 0.01 * torch.randn_like(traj)

        out_aug = self.model(
            traj=traj_aug,
            adj=adj,
            x_graph=x_graph,
            traj_mask=traj_mask,
            graph_mask=graph_mask,
        )
        z_aug = out_aug["fusion"]

        # --------------------------------------------------
        # 3. VICReg losses
        # --------------------------------------------------
        loss_inv = vicreg_invariance(z_clean, z_aug)
        loss_var = vicreg_variance(z_clean) + vicreg_variance(z_aug)
        loss_cov = vicreg_covariance(z_clean) + vicreg_covariance(z_aug)

        loss = (
            self.lambda_inv * loss_inv
            + self.lambda_var * loss_var
            + self.lambda_cov * loss_cov
        )

        # --------------------------------------------------
        # 4. Optimization 
        # --------------------------------------------------
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_NORM)
        self.opt.step()

        # --------------------------------------------------
        # 5. EMA update
        # --------------------------------------------------
        update_ema(self.ema_model, self.model, EMA_JEPA2)
        
        # --------------------------------------------------
        # 6. EMA forward → slow target for JEPA-3
        # --------------------------------------------------
        with torch.no_grad():
            out_ema = self.ema_model(
                traj=traj,
                adj=adj,
                x_graph=x_graph,
                traj_mask=traj_mask,
                graph_mask=graph_mask,
            )
            s_tg = out_ema["fusion"]   # EMA target for JEPA-3

        return {
            "loss": loss.detach(),
            "loss_inv": loss_inv.detach(),
            "loss_var": loss_var.detach(),
            "loss_cov": loss_cov.detach(),
            "s_tg": s_tg.detach(),   # JEPA-3 input
        }
