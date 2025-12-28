import torch
import torch.nn.functional as F
from .utils import cosine_distance, vic_reg_loss, info_nce_loss_temp_free, info_nce_loss_temp_schedule

def global_encoding_losses(
    glob_out,
    global_step: int,
    use_temp_free: bool = False,
    lambda_cos: float = 1.0,
    lambda_n_max: float = 0.5,
    warmup_steps: int = 300,
    w_var: float = 0.1,
    w_cov: float = 0.01,
):
    """
    JEPA-3 Global Encoding Loss with:
      - Cosine alignment
      - InfoNCE (temperature-scheduled or temp-free)
      - Full VICReg regularization
    """
    pred_tar = glob_out["pred_tar"]  # predicted embeddings
    s_tar = glob_out["s_tar"]        # teacher target embeddings

    if s_tar is None:
        raise ValueError("s_tar is None, cannot compute JEPA-3 loss")

    # -----------------------------
    # Cosine alignment loss
    # -----------------------------
    cos_loss = cosine_distance(pred_tar, s_tar).mean()

    # -----------------------------
    # InfoNCE loss
    # -----------------------------
    if use_temp_free:
        loss_nce = info_nce_loss_temp_free(pred_tar, s_tar)
        tau = torch.tensor(float('nan'), device=pred_tar.device)  # no tau in temp-free
    else:
        # horizon-free temperature schedule
        loss_nce, tau = info_nce_loss_temp_schedule(
            pred_tar,
            s_tar,
            step=global_step,
        )
    
    # -----------------------------
    # λ warmup for InfoNCE
    # -----------------------------
    lambda_n = min(global_step / warmup_steps, 1.0) * lambda_n_max

    # -----------------------------
    # VICReg regularization
    # -----------------------------
    vic_loss = vic_reg_loss(pred_tar, var_weight=w_var, cov_weight=w_cov)

    # -----------------------------
    # Total loss
    # -----------------------------
    total = lambda_cos * cos_loss + lambda_n * loss_nce + vic_loss

    return {
        "total": total,
        "cos": cos_loss.detach(),
        "nce": loss_nce.detach(),
        "vicreg": vic_loss.detach(),
        "tau": tau.detach() if tau is not None else tau,
    }
