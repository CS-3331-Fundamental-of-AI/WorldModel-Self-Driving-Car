# ============================================================
#  TRAJECTORY TOKENIZER – LOSS FUNCTIONS (FSQ + VICREG)
# ============================================================

import torch
import torch.nn.functional as F
import math

# ------------------------------------------------------------
# 1. Trajectory reconstruction losses
# ------------------------------------------------------------
def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard MSE across the trajectory window.
    Forces decoder outputs to track the physical motion.
    """
    return F.mse_loss(recon, target)


def temporal_smoothness_loss(traj: torch.Tensor) -> torch.Tensor:
    """
    Penalize large frame-to-frame jumps in reconstructed trajectories.

    Encourages:
      - physically plausible movement
      - smoother outputs

    Too much weight here → overly flat trajectories.
    """
    diff = traj[:, 1:, :] - traj[:, :-1, :]
    return (diff ** 2).mean()


# ------------------------------------------------------------
# 2. VICReg: invariance between clean & augmented latents
# ------------------------------------------------------------
def vicreg_invariance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Encourages the model to map clean and augmented trajectories
    to the same latent region.
    """
    return F.mse_loss(z1, z2)


# ------------------------------------------------------------
# 3. VICReg: variance term (anti-collapse)
# ------------------------------------------------------------
def vicreg_variance(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    Forces each latent dimension to maintain non-trivial variance.

    Prevents:
      - representation collapse
      - encoder shrinking all dimensions
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    penalty = torch.relu(gamma - std)
    return penalty.mean()


# ------------------------------------------------------------
# 4. VICReg: covariance term (decorrelation)
# ------------------------------------------------------------
def vicreg_covariance(z: torch.Tensor) -> torch.Tensor:
    """
    Penalize redundancy across latent dimensions.

    Encourages:
      - diverse latent axes
      - improved codebook spreading for FSQ inputs
    """
    B, D = z.shape
    z_norm = z - z.mean(dim=0, keepdim=True)
    cov = (z_norm.T @ z_norm) / (B - 1)

    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()


# ------------------------------------------------------------
# 5. FSQ token usage regularization
# ------------------------------------------------------------
def fsq_token_usage_loss(tokens: torch.Tensor, num_levels: int) -> torch.Tensor:
    """
    Encourage balanced usage across per-dimension discrete FSQ levels.

    tokens:     [B, d_q] (each entry in [0, L-1])
    num_levels: number of scalar levels (L)
    """
    B, d_q = tokens.shape
    L = num_levels

    # Frequency estimate via differentiable histogram
    one_hot = F.one_hot(tokens, num_classes=L).float()   # [B, d_q, L]
    freq = one_hot.mean(dim=(0, 1))                     # [L]

    # Uniform distribution target
    target = torch.full_like(freq, 1.0 / L)

    # KL divergence: freq || uniform
    usage_loss = (freq * (freq.add(1e-8).log() - target.log())).sum()
    return usage_loss


# ============================================================
# 6. TOTAL LOSS (FSQ + VICREG + Smoothness)
# ============================================================
def total_tokenizer_loss_fsq(
    out_clean: dict,
    out_aug: dict,
    target_traj: torch.Tensor,
    fsq_levels: int,
    lambda_recon=1.0,
    lambda_smooth=0.1,
    lambda_inv=1.0,
    lambda_var=1.0,
    lambda_cov=0.04,
    lambda_usage=0.1,
):
    """
    Combines:
      - reconstruction (clean only)
      - temporal smoothness (clean only)
      - VICReg invariance (clean ↔ aug)
      - VICReg variance (clean + aug)
      - VICReg covariance (clean + aug)
      - FSQ level usage regularization
    """

    # --------------------------------------------------------
    # Reconstruction losses on clean view
    # --------------------------------------------------------
    L_recon  = reconstruction_loss(out_clean["recon"], target_traj)
    L_smooth = temporal_smoothness_loss(out_clean["recon"])

    # --------------------------------------------------------
    # VICReg latent losses (z0 from clean & augmented)
    # --------------------------------------------------------
    zc = out_clean["z0"]
    za = out_aug["z0"]

    L_inv = vicreg_invariance(zc, za)
    L_var = vicreg_variance(zc) + vicreg_variance(za)
    L_cov = vicreg_covariance(zc) + vicreg_covariance(za)

    # --------------------------------------------------------
    # FSQ token-level usage regularization
    # --------------------------------------------------------
    L_usage = fsq_token_usage_loss(out_clean["tokens"], fsq_levels)

    # --------------------------------------------------------
    # Total combined loss
    # --------------------------------------------------------
    L_total = (
        lambda_recon * L_recon +
        lambda_smooth * L_smooth +
        lambda_inv * L_inv +
        lambda_var * L_var +
        lambda_cov * L_cov +
        lambda_usage * L_usage
    )

    # Dictionary for logging and monitoring
    return {
        "loss_total": L_total,
        "loss_recon": L_recon,
        "loss_smooth": L_smooth,
        "vic_inv": L_inv,
        "vic_var": L_var,
        "vic_cov": L_cov,
        "fsq_usage": L_usage,
    }


# ------------------------------
# 7. Temperature schedule
# ------------------------------

def cosine_tau_schedule(
    step: int,
    total_steps: int,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
    warmup_frac: float = 0.1,
    anneal_frac: float = 0.8,
) -> float:
    """
    Cosine temperature annealing as in the PDF:
    - warmup [0, warmup_frac]: tau = tau_start
    - anneal [warmup_frac, anneal_frac]: cosine from tau_start -> tau_end
    - cooldown [anneal_frac, 1.0]: tau = tau_end
    """
    t = step / max(1, total_steps)

    if t <= warmup_frac:
        return tau_start

    if t >= anneal_frac:
        return tau_end

    # normalized progress in [0,1] for the anneal segment
    t_norm = (t - warmup_frac) / (anneal_frac - warmup_frac)
    tau = tau_end + 0.5 * (tau_start - tau_end) * (1.0 + math.cos(math.pi * t_norm))
    return tau