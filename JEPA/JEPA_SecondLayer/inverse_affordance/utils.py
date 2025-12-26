import torch
import torch.nn as nn

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


# VIC-Reg placeholder (regularization on embeddings)
def vic_reg_loss(x):
    # x: (B, D)
    # use simple variance and covariance penalties as in VICReg
    x = x - x.mean(dim=0, keepdim=True)
    var = x.var(dim=0).mean()
    cov = (x.T @ x) / (x.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / x.shape[1]
    return var * 0.1 + cov_loss * 0.1
    