import torch
import torch.nn.functional as F
from .utils import cosine_distance, vic_reg_loss

def inverse_affordance_losses(out, target_s_tg):
    losses = {}

    s_y = out["s_y"]
    z_ca = out["z_ca"]

    if s_y.dim() > 2:
        s_y = s_y.flatten(1)
    if target_s_tg.dim() > 2:
        target_s_tg = target_s_tg.flatten(1)

    losses["l1"] = F.l1_loss(s_y, target_s_tg)
    losses["cos"] = cosine_distance(s_y, target_s_tg).mean()

    losses["vic_s_y"]  = vic_reg_loss(s_y)
    losses["vic_z_ca"] = vic_reg_loss(z_ca)

    losses["total"] = (
        losses["l1"]
        + losses["cos"]
        + 0.1 * losses["vic_s_y"]
        + 0.1 * losses["vic_z_ca"]
    )

    return losses

# ---------------------------------------------------------
# Global encoding losses
# ---------------------------------------------------------
def global_encoding_losses(glob_out, s_tg):
    s_tar = glob_out["s_tar"]

    losses = {}
    losses["cos_tar_tg"] = cosine_distance(s_tar, s_tg).mean()
    losses["vic_tar"] = vic_reg_loss(s_tar)

    losses["total"] = (
        losses["cos_tar_tg"]
        + 0.1 * losses["vic_tar"]
    )
    return losses
