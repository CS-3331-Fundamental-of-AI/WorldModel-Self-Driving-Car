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

    def step(self, traj, graph, traj_mask=None, graph_mask=None):
        """
        traj       : [B, T, 6]
        graph      : (graph_feats, adj)
        graph_mask : [B, N]  (node validity mask)
        """

        graph_feats, adj = graph

        out = self.model(
            traj=traj,
            adj=adj,
            x_graph=graph_feats,
            traj_mask=traj_mask,
            graph_mask=graph_mask
        )

        loss_dict = total_tokenizer_loss_fsq(out)
        loss = loss_dict["total"]

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_NORM)
        self.opt.step()

        update_ema(self.ema_model, self.model, EMA_JEPA2)

        return {
            "loss": loss.detach(),
            "s_tg": out["fusion"],  # or correct token if different
            "out": out
        }
