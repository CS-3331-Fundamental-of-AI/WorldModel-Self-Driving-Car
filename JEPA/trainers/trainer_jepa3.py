# trainers/trainer_jepa3.py

import torch
from JEPA_ThirdLayer.losses import global_encoding_losses
from config.config import CLIP_NORM


class JEPA3Trainer:
    def __init__(self, glob, optimizer):
        self.glob = glob
        self.opt = optimizer
        self.global_step = 0
        
    def step(
        self,
        s_y,
        s_c,
        s_tg,
        global_nodes,
        global_edges,
        tokens_final=None,
    ):
        self.global_step += 1
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
        losses_dict = {}
        if s_tar is not None:
            losses_dict = global_encoding_losses(
                out,
                global_step=self.global_step,
            )
            loss_total = losses_dict["total"]
        else:
            loss_total = 0.0 * pred_tar.sum() if pred_tar is not None else torch.tensor(0.0)  # zero loss if no graph
            losses_dict = {"cos": torch.tensor(0.0), "nce": torch.tensor(0.0), "vicreg": torch.tensor(0.0), "tau": torch.tensor(float('nan'))}

        # -----------------------------
        # Optimize
        # -----------------------------
        self.opt.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.glob.parameters(), CLIP_NORM)
        self.opt.step()

        # -----------------------------
        # EMA Update
        # -----------------------------
        if s_tar is not None:
            self.glob.update_ema()

        # Return all individual losses + total
        return {**losses_dict, "loss": loss_total.detach()}