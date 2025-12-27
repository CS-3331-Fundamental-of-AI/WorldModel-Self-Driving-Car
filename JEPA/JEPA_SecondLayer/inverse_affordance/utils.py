import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

# ------------------------
# EMA helper
# ------------------------
@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, momentum: float):
    # Update ema_model parameters to be: ema = ema * m + (1-m) * param
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)

# Distance / Loss helpers
# ---------------------------------------------------------------------------
import math
from typing import Optional, Tuple

def cosine_distance(a, b):
    # a,b: (B, D) or (B, N, D) -> return 1 - cos_sim
    if a.dim() == 3:
        a = a.flatten(0, 1)
        b = b.flatten(0, 1)
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    cos = (a_norm * b_norm).sum(dim=-1)
    return 1.0 - cos


#---------------------------------------------------------
# VICReg loss
# ---------------------------------------------------------
def vic_reg_loss(x, eps=1e-4, var_weight=1.0, cov_weight=1.0):
    """
    Full VICReg regularization on embeddings x
    Args:
        x: [B, D] embedding tensor
        eps: small number to avoid sqrt(0)
        var_weight: weight for variance term
        cov_weight: weight for covariance term
    Returns:
        scalar VICReg loss
    """
    # Centered embeddings
    x = x - x.mean(dim=0, keepdim=True)  # [B, D]

    # -----------------------------
    # Variance term: encourage std > 1
    # -----------------------------
    std = torch.sqrt(x.var(dim=0) + eps)  # [D]
    var_loss = torch.mean(F.relu(1 - std))

    # -----------------------------
    # Covariance term: decorrelate features
    # -----------------------------
    B, D = x.shape
    cov = (x.T @ x) / (B - 1)  # [D, D]
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / D

    return var_weight * var_loss + cov_weight * cov_loss
