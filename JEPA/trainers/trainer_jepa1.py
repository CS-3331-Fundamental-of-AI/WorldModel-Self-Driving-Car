# trainers/trainer_jepa1.py
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from JEPA_PrimitiveLayer.Utils import compute_jepa_loss
from JEPA_PrimitiveLayer.Utils import ema_update
from Utils.spatial import up2

from config.config import (
    ALPHA_0, ALPHA_1,
    BETA_1, BETA_2,
    LAMBDA_JEPA, LAMBDA_REG, GAMMA,
    CLIP_NORM, EMA_JEPA1
)

class JEPA1Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.opt = optimizer

    def step(self, batch_masks):
        mask_emp, mask_non, mask_union = batch_masks
        B = mask_emp.shape[0]

        mask_emp_up = up2(mask_emp.view(B,1,32,32))
        mask_non_up = up2(mask_non.view(B,1,32,32))
        mask_any_up = up2(mask_union.view(B,1,32,32))

        z_c, s_c, z_t = self.model(
            mask_emp.squeeze(1),
            mask_non.squeeze(1),
            mask_emp_up,
            mask_non_up,
            mask_any_up,
        )

        z_c = F.normalize(z_c, dim=-1)
        s_c = F.normalize(s_c, dim=-1)
        z_t = F.normalize(z_t, dim=-1)

        losses = compute_jepa_loss(
            s_c=s_c, s_t=z_t, z_c=z_c,
            mask_empty=mask_emp_up.view(B,-1),
            mask_nonempty=mask_non_up.view(B,-1),
            alpha0=ALPHA_0, alpha1=ALPHA_1,
            beta1=BETA_1, beta2=BETA_2,
            lambda_jepa=LAMBDA_JEPA,
            lambda_reg=LAMBDA_REG,
            gamma=GAMMA
        )

        loss = losses["loss_total"]

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_NORM)
        self.opt.step()

        ema_update(
            self.model.context_encoder,
            self.model.target_encoder,
            EMA_JEPA1
        )

        return {
            "loss": loss.detach(),
            "s_c": s_c.detach(),
            "z_t": z_t.detach(),
        }
