import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

def info_nce_loss_temp_schedule(
    z_pred,
    z_tar,
    step: int,
    tau_start: float = 0.2,
    tau_end: float = 0.07,
    tau_timescale: int = 2000,
):
    """
    Cosine-similarity InfoNCE with horizon-free temperature schedule.

    - z_pred: [B, D] predicted embeddings
    - z_tar:  [B, D] target embeddings
    - step: current training step
    - tau_start: initial temperature (high → exploration)
    - tau_end: final temperature (low → sharper alignment)
    - tau_timescale: timescale for temperature annealing (horizon-free)
    
    Returns:
        loss: InfoNCE loss
        tau:  current temperature
    """
    # normalize embeddings
    z_pred = F.normalize(z_pred, dim=-1)
    z_tar  = F.normalize(z_tar, dim=-1)

    # cosine similarity matrix
    sim = torch.matmul(z_pred, z_tar.T)
    sim = sim.clamp(-1 + 1e-6, 1 - 1e-6)  # optional safety clamp

    # horizon-free cosine annealing
    progress = torch.tensor(min(step / tau_timescale, 1.0), device=z_pred.device)
    tau = tau_end + 0.5 * (tau_start - tau_end) * (1 + torch.cos(torch.pi * progress))

    # compute InfoNCE loss
    logits = sim / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    loss = F.cross_entropy(logits, labels)

    return loss, tau

def info_nce_loss_temp_free(z_pred, z_tar, eps=1e-6):
    """
    Temperature-free InfoNCE using 2 * arctanh(cos similarity).
    """
    # normalize
    z_pred = F.normalize(z_pred, dim=-1)
    z_tar = F.normalize(z_tar, dim=-1)
    
    # cosine similarity matrix
    sim = torch.matmul(z_pred, z_tar.T)

    # clamp for numerical stability (cosine in (-1+eps, 1-eps))
    sim = sim.clamp(-1 + eps, 1 - eps)

    # logit mapping: 2 * arctanh(sim)
    logits = 2 * torch.atanh(sim)

    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(logits, labels)

def temperature_schedule(
    step: int,
    tau_start: float = 0.2,
    tau_end: float = 0.07,
    tau_timescale: int = 5000,
):
    """
    Horizon-free cosine annealing temperature.

    - High τ early → exploration
    - Low τ later → sharper alignment
    - No assumption about total training length
    """
    progress = min(step / tau_timescale, 1.0)

    tau = tau_end + 0.5 * (tau_start - tau_end) * (
        1 + torch.cos(torch.pi * progress)
    )
    return tau


def info_nce_loss_with_tau(
    z_pred,
    z_tar,
    tau,
):
    """
    Cosine-similarity InfoNCE with external temperature.
    """
    z_pred = F.normalize(z_pred, dim=-1)
    z_tar  = F.normalize(z_tar, dim=-1)

    # cosine similarity matrix
    sim = torch.matmul(z_pred, z_tar.T)

    # optional safety clamp (recommended with EMA teachers)
    sim = sim.clamp(-1 + 1e-6, 1 - 1e-6)

    logits = sim / tau
    labels = torch.arange(sim.size(0), device=sim.device)

    return F.cross_entropy(logits, labels)
