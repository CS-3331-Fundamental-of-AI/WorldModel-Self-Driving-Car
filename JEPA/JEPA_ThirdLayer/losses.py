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
def global_encoding_losses(glob_out, s_tar_target=None, z=None):
    losses = {}
    s_tar = glob_out["s_tar"]
    s_ctx = glob_out["s_ctx"]

    losses["cos_tar_ctx"] = cosine_distance(s_tar, s_ctx).mean()
    losses["l1_tar_ctx"] = F.l1_loss(s_tar, s_ctx)
    losses["vic_tar"] = vic_reg_loss(s_tar)

    if s_tar_target is not None:
        losses["cos_tar_true"] = cosine_distance(s_tar, s_tar_target).mean()
        losses["l1_tar_true"] = F.l1_loss(s_tar, s_tar_target)

    total = (
        losses["cos_tar_ctx"]
        + 0.5 * losses["l1_tar_ctx"]
        + 0.1 * losses["vic_tar"]
        + 0.5 * losses.get("cos_tar_true", 0)
    )
    losses["total"] = total
    return losses

def global_future_loss(s_tar_t, s_tar_future):
    """
    JEPA-style temporal prediction loss for global encoding.
    """

    losses = {}

    losses["cos_tar_future"] = cosine_distance(
        s_tar_t, s_tar_future
    ).mean()

    losses["vic_tar"] = vic_reg_loss(s_tar_t)

    total = (
        losses["cos_tar_future"]
        + 0.1 * losses["vic_tar"]
    )

    # ---- compatibility aliases ----
    losses["cos_tar_ctx"] = losses["cos_tar_future"]  # legacy name
    losses["total"] = total

    return losses
