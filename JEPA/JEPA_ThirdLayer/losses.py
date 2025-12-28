import torch
import torch.nn.functional as F
from .utils import cosine_distance, vic_reg_loss, info_nce_loss_temp_free, info_nce_loss_temp_schedule


def global_encoding_losses(
    glob_out,
    step,
    total_steps,
    use_temp_free=False,
    w_var=0.1,
    w_cov=0.01,
):
    """
    JEPA-3 Global Encoding Loss

    Main objective:
        D(pred_tar, s_tar)

    Regularization:
        Full VICReg on pred_tar to prevent collapse
    """
    pred_tar = glob_out["pred_tar"]  # z
    s_tar = glob_out["s_tar"]        # teacher target

    if s_tar is None:
        raise ValueError("s_tar is None, cannot compute JEPA-3 loss")

    # -----------------------------
    # Invariance losses
    # -----------------------------
    #cos_loss = cosine_distance(pred_tar, s_tar).mean()
    # choose contrastive loss
    if use_temp_free:
        cos_loss = info_nce_loss_temp_free(pred_tar, s_tar)
    else:
        cos_loss = info_nce_loss_temp_schedule(
            pred_tar,
            s_tar,
            step=step,
            total_steps=total_steps
        )
    # -----------------------------
    # VICReg regularization (FULL)
    # -----------------------------
    vic_loss = vic_reg_loss(
        pred_tar,
        var_weight=w_var,
        cov_weight=w_cov,
    )

    # -----------------------------
    # Total
    # -----------------------------
    total = (cos_loss + vic_loss)

    return {
        "total": total,
        "cos_pred_tar": cos_loss,
        "vic_pred_tar": vic_loss,
    }

