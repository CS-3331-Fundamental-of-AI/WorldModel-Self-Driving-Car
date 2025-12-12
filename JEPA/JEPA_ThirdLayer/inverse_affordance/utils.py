import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class EMAHelper:
    """Simple EMA helper for slow update of target network weights.
    Usage: ema = EMAHelper(decay=0.999)
           ema.register(model)
           ema.update(model)
           ema.assign_to(target_model)  # copy to target
    """

    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}

    def register(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                new = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                self.shadow[name] = new.clone()

    def assign_to(self, target: nn.Module):
        for name, p in target.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

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
    
# ---------------------------------------------------------------------------
# Loss calculation function to use externally
# ---------------------------------------------------------------------------

def inverse_affordance_losses(out: dict, target_s_tg: torch.Tensor, target_s_tg_2: Optional[torch.Tensor] = None):
    """Compute losses used in architecture: D(s_tg^, s_a) and D(s_tg^, s_tg) etc.
    - Cosine / L1 between s_tg_hat and s_a (path1)
    - Cosine / L1 between s_tg_hat and target_s_tg (path2)
    - VIC-Reg on embeddings
    Return dict of losses.
    """
    losses = {}
    s_tg_hat = out["s_tg_hat"]
    s_a = out["s_a_detached"]  # frozen

    # reduce to vector if needed
    if s_tg_hat.dim() > 2:
        s_tg_hat = s_tg_hat.flatten(1)
    if s_a.dim() > 2:
        s_a = s_a.flatten(1)

    # Cosine distance
    losses["cos_s_tg_a"] = cosine_distance(s_tg_hat, s_a).mean()
    losses["l1_s_tg_a"] = F.l1_loss(s_tg_hat, s_a)

    # compare with true target (if available)
    if target_s_tg is not None:
        if target_s_tg.dim() > 2:
            target_s_tg = target_s_tg.flatten(1)
        losses["cos_s_tg_true"] = cosine_distance(s_tg_hat, target_s_tg).mean()
        losses["l1_s_tg_true"] = F.l1_loss(s_tg_hat, target_s_tg)

    # VIC-Reg on s_y and z_ca
    losses["vic_s_y"] = vic_reg_loss(out["s_y"])
    losses["vic_z_ca"] = vic_reg_loss(out["z_ca"])

    # total (simple weighted sum - tune weights externally)
    total = (
        1.0 * losses["cos_s_tg_a"] + 0.5 * losses["l1_s_tg_a"] +
        (0.5 * losses.get("cos_s_tg_true", 0.0)) + 0.5 * losses.get("l1_s_tg_true", 0.0) +
        0.1 * losses["vic_s_y"] + 0.1 * losses["vic_z_ca"]
    )
    losses["total"] = total
    return losses
