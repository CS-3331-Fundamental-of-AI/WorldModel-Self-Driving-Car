# trainers/trainer_jepa3.py
import torch
from JEPA_ThirdLayer.losses import inverse_affordance_losses, global_encoding_losses
from config.config import CLIP_NORM

class JEPA3Trainer:
    def __init__(
        self,
        inv,            # student inverse model (JEPA_Tier3_InverseAffordance)
        glob,           # student global model (JEPA_Tier3_GlobalEncoding)
        optimizer,
    ):
        self.inv = inv
        self.glob = glob
        self.opt = optimizer

    def step(self, action, s_c, s_tg=None, global_nodes=None, global_edges=None):
        """
        JEPA-3 training step:
        - action : [B, 2]
        - s_c    : context representation from JEPA-1 [B, C]
        - s_tg   : target representation from JEPA-2 [B, C] 
        - s_y    : predicted target from inverse model
        - global_nodes : list of node tensors per sample
        - global_edges   : list of edge tensors per sample
        """
        # -----------------------------
        # Forward: inverse model
        # -----------------------------
        if s_c is None or action is None:
            raise ValueError("s_c and action must not be None for JEPA-3 step")
        inv_out = self.inv(action, s_c)  # s_c used internally by IA
        s_y = inv_out["s_y"]

        # -----------------------------
        # Forward: global encoding
        # -----------------------------
       
        glob_out = self.glob(
            s_y,                # predicted target from IA
            s_c,                # raw context from JEPA-1
            s_tg,
            global_nodes,
            global_edges,
            tokens_final=inv_out.get("tokens", None),
        )

        # -----------------------------
        # Losses
        # -----------------------------
        loss_inv = inverse_affordance_losses(inv_out, s_tg)
        loss_glob = global_encoding_losses(glob_out, s_tg)
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
        # EMA update (GLOBAL ENCODER)
        # -----------------------------
        self.glob.update_ema()
        
        return {
            "loss": loss_total.detach(),
            "loss_inv": loss_inv["total"].detach(),
            "loss_glob": loss_glob["total"].detach(),
        }
