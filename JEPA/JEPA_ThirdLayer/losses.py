import torch
import torch.nn.functional as F
from .utils import cosine_distance, vic_reg_loss

# ---------------------------------------------------------
# Inverse affordance losses
# ---------------------------------------------------------
def inverse_affordance_losses(out, target_s_tg=None):
    """
    Compute losses for JEPA Tier-3 Inverse Affordance (IA).

    Two main components:
      1. Consistency loss: s_tg_hat vs s_a_detached
      2. Supervised loss: s_tg_hat vs target_s_tg (ground truth)
    
    Also includes VIC regularization losses on s_y and z_ca.
    
    Returns:
        losses: dict with individual components
        loss_consistency: scalar tensor
        loss_supervised: scalar tensor
    """
    losses = {}

    s_tg_hat = out["s_tg_hat"]      # predicted latent
    s_a = out.get("s_a_detached")   # detached self-consistent latent

    # flatten if needed
    if s_tg_hat.dim() > 2:
        s_tg_hat = s_tg_hat.flatten(1)
    if s_a is not None and s_a.dim() > 2:
        s_a = s_a.flatten(1)

    # --------------------------
    # 1) Consistency loss: s_tg_hat vs s_a_detached
    # --------------------------
    if s_a is not None:
        losses["cos_s_tg_a"] = cosine_distance(s_tg_hat, s_a).mean()
        losses["l1_s_tg_a"] = F.l1_loss(s_tg_hat, s_a)
        loss_consistency = 1.0 * losses["cos_s_tg_a"] + 0.5 * losses["l1_s_tg_a"]
    else:
        loss_consistency = torch.tensor(0.0, device=s_tg_hat.device)

    # --------------------------
    # 2) Supervised loss: s_tg_hat vs target_s_tg
    # --------------------------
    if target_s_tg is not None:
        if target_s_tg.dim() > 2:
            target_s_tg = target_s_tg.flatten(1)
        losses["cos_s_tg_true"] = cosine_distance(s_tg_hat, target_s_tg).mean()
        losses["l1_s_tg_true"] = F.l1_loss(s_tg_hat, target_s_tg)
        loss_supervised = 0.5 * (losses["cos_s_tg_true"] + losses["l1_s_tg_true"])
    else:
        loss_supervised = torch.tensor(0.0, device=s_tg_hat.device)

    # --------------------------
    # 3) VIC regularization
    # --------------------------
    losses["vic_s_y"] = vic_reg_loss(out["s_y"])
    losses["vic_z_ca"] = vic_reg_loss(out["z_ca"])

    # --------------------------
    # Total loss (optional, can weight outside trainer)
    # --------------------------
    losses["total"] = loss_consistency + loss_supervised + 0.1 * losses["vic_s_y"] + 0.1 * losses["vic_z_ca"]

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
