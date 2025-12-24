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
# VICReg placeholder
# ---------------------------------------------------------
def vic_reg_loss(x):
    x = x - x.mean(dim=0, keepdim=True)
    var = x.var(dim=0).mean()

    cov = (x.T @ x) / (x.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / x.size(1)

    return 0.1 * var + 0.1 * cov_loss
