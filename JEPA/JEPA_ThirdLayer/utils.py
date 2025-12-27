import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Freeze module
# ---------------------------------------------------------
def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


# ---------------------------------------------------------
# EMA helper
# ---------------------------------------------------------
class EMAHelper:
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}

    def register(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name] = (1 - self.decay) * p.detach() + self.decay * self.shadow[name]

    @torch.no_grad()
    def assign_to(self, target: nn.Module):
        for name, p in target.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name])

# ---------------------------------------------------------
# Cosine distance
# ---------------------------------------------------------
def cosine_distance(a, b):
    if a.dim() == 3:
        a = a.flatten(0, 1)
        b = b.flatten(0, 1)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1 - (a * b).sum(dim=-1)


# ---------------------------------------------------------
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
