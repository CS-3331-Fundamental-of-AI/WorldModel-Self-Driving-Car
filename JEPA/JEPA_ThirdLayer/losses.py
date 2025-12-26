import torch
import torch.nn.functional as F
from .utils import cosine_distance, vic_reg_loss

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
