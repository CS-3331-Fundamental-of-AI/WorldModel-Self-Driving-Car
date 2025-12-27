import torch
from JEPA_ThirdLayer.losses import global_encoding_losses
from config.config import CLIP_NORM


class JEPA3Trainer:
    """
    Trainer for JEPA-3 Global Encoding (Student + EMA teacher)
    - Implements EMA warmup correctly
    - Keeps LayerNorm / grad clipping handled in model
    """
    def __init__(
        self,
        glob,
        optimizer,
        ema_start: int = 0,
        ema_warmup: int = 1000,
        ema_final: float = 0.999,
    ):
        self.glob = glob
        self.opt = optimizer

        # EMA scheduling
        self.global_step = 0
        self.ema_start = ema_start
        self.ema_warmup = ema_warmup
        self.ema_final = ema_final

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
        # Forward pass
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
        # Compute loss
        # -----------------------------
        if s_tar is not None:
            loss = global_encoding_losses(out)["total"]
        else:
            loss = 0.0 * pred_tar.sum()  # zero loss if no graph

        # -----------------------------
        # Backprop + gradient clipping
        # -----------------------------
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.glob.parameters(), CLIP_NORM)
        self.opt.step()

        # -----------------------------
        # EMA update (with warmup)
        # -----------------------------
        if s_tar is not None and self.global_step >= self.ema_start:
            # compute EMA decay
            if self.global_step < self.ema_warmup:
                progress = (self.global_step - self.ema_start) / max(1, self.ema_warmup - self.ema_start)
                decay = 0.99 + progress * (self.ema_final - 0.99)
            else:
                decay = self.ema_final

            # apply dynamic decay
            self.glob.ema_helper.decay = decay
            self.glob.update_ema()

        return {
            "loss": loss.detach(),
        }
