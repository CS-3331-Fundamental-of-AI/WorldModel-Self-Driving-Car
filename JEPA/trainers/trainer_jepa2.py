# trainers/trainer_jepa2.py

import torch

from JEPA_SecondLayer.losses import total_tokenizer_loss_fsq
from JEPA_SecondLayer.utils import update_ema
from config.config import CLIP_NORM, EMA_JEPA2


class JEPA2Trainer:
    def __init__(self, model, ema_model, optimizer):
        self.model = model
        self.ema_model = ema_model
        self.opt = optimizer

    def step(self, traj, graph=None):
        """
        traj  : trajectory tokens / deltas
        graph : optional graph batch
        """

        out = self.model(traj, graph)

        loss_dict = total_tokenizer_loss_fsq(out)
        loss = loss_dict["total"]

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_NORM)
        self.opt.step()

        # -------------------------
        # EMA update (teacher)
        # -------------------------
        update_ema(
            ema_model=self.ema_model,
            model=self.model,
            momentum=EMA_JEPA2
        )

        return {
            "loss": loss.detach(),
            "s_tg": out["s_tg"],   # student graph tokens
            "out": out
        }
