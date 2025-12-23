import torch
from typing import Dict
from JEPA_PrimitiveLayer.vjepa.losses import jepa_embedding_loss


class JEPA1VJEPATrainer:
    """
    JEPA-1 trainer backed by a frozen V-JEPA-2 encoder.

    - Encoder frozen
    - Predictor trainable
    - Produces s_c = z_hat
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_cfg: Dict = None,
    ):
        self.model = model
        self.opt = optimizer
        self.device = device

        self.loss_cfg = loss_cfg or {
            "alpha": 1.0,
            "beta":  1.0,
            "gamma": 0.1,
        }

        self.model.train()

    def step(self, batch: Dict):
        # --------------------------------------------------
        # 1) Inputs
        # --------------------------------------------------
        px = batch["pixel_values"].to(
            self.device,
            dtype=torch.float16,
            non_blocking=True,
        )

        # --------------------------------------------------
        # 2) Forward
        # --------------------------------------------------
        z_hat, z_proj = self.model(px)

        # --------------------------------------------------
        # 3) Loss (FP32)
        # --------------------------------------------------
        total_loss, loss_dict = jepa_embedding_loss(
            z_hat.float(),
            z_proj.float(),
            alpha=self.loss_cfg["alpha"],
            beta=self.loss_cfg["beta"],
            gamma=self.loss_cfg["gamma"],
        )

        # --------------------------------------------------
        # 4) Backprop
        # --------------------------------------------------
        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.opt.step()

        # --------------------------------------------------
        # 5) Outputs
        # --------------------------------------------------
        return {
            "loss": total_loss.detach(),
            "s_c": z_hat.detach(),
            "stats": {
                "loss_align": loss_dict["loss_align"].detach(),
                "loss_var":   loss_dict["loss_var"].detach(),
                "loss_cov":   loss_dict["loss_cov"].detach(),
                "z_hat_std":  z_hat.std().detach(),
                "z_hat_mean": z_hat.mean().detach(),
            }
        }
