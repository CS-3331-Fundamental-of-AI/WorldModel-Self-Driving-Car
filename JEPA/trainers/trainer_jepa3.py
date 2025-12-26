# trainers/trainer_jepa3.py

import torch
from JEPA_ThirdLayer.losses import global_encoding_losses
from config.config import CLIP_NORM


class JEPA3Trainer:
    def __init__(self, glob, optimizer):
        self.glob = glob
        self.opt = optimizer

    def step(
        self,
        s_y,
        s_c,
        s_tg,
        global_nodes,
        global_edges,
        tokens_final=None,
    ):
        # -----------------------------
        # Forward
        # -----------------------------
        out = self.glob(
            s_y,
            s_c,
            s_tg,
            global_nodes,
            global_edges,
            tokens_final=tokens_final,
        )

        s_tar = out["s_tar"]
        pred_tar = out["pred_tar"]
        # -----------------------------
        # Loss
        # -----------------------------
        if s_tar is not None:
            loss = global_encoding_losses(out)["total"]
        else:
            loss = 0.0 * pred_tar.sum()  # zero loss if no graph

        # -----------------------------
        # Optimize
        # -----------------------------
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.glob.parameters(), CLIP_NORM)
        self.opt.step()

        # -----------------------------
        # EMA Update
        # -----------------------------
        if s_tar is not None:
            self.glob.update_ema()

        return {
            "loss": loss.detach(),
        }
