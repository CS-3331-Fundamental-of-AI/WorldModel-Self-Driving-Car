import torch
import torch.nn.functional as F

"""
Zhu’s AD-L-JEPA keeps only the essentials for predictive BEV learning:

Kept:
  - JEPA masked embedding prediction (predict embeddings, not points)
  - VICReg variance term only (simple + effective collapse prevention)
  - EMA target encoder (stabilizes learning; replaces invariance)
  - BEV-grid masking (empty + non-empty)

Dropped:
  - Invariance term (unnecessary: JEPA already aligns context→target)
  - Covariance term (redundant + expensive; EMA + variance suffice)
  - Contrastive negatives (not needed in predictive JEPA)
  - Pixel/point-cloud reconstruction (not needed in embedding JEPA)

When to use FULL VICReg (invariance + variance + covariance):
  - Dual-view Siamese SSL (two augmentations to align)
  - Multi-modal alignment (image↔text, audio↔video)
  - No EMA or predictor used → full VICReg needed to avoid collapse
  - When whitening / decorrelation improves downstream tasks

When to use HALF VICReg (variance term only):
  - JEPA-style predictive models (masked tokens, future embeddings)
  - Architectures with EMA teacher → invariance unnecessary
  - BEV / spatial-grid encoders where covariance is costly + low benefit
  - Large-scale masking pretraining where efficiency matters
"""


def jepa_loss(
    s_c,          # predicted embedding (B, N, D)
    s_t,          # target embedding (B, N, D)
    z_c,          # context encoder embedding AFTER empty/mask token replace (B,N,D)
    mask_empty,   # (B, N)   masked empty grids (P)
    mask_nonempty,# (B, N)   masked non-empty grids (Q)
    alpha0=0.25,
    alpha1=0.75,
    beta1=1.0,
    beta2=1.0,
    lambda_jepa=1.0,
    lambda_reg=1.0,
    gamma=1.0
):
    B, N, D = s_c.shape
    device = s_c.device

    # ---------------------------------------------------------
    # 1. JEPA cosine loss on masked indices
    # ---------------------------------------------------------

    # -- EMPTY masked (P set)
    mask_P = mask_empty.bool()                        # (B,N)
    if mask_P.any():
        s_c_P = s_c[mask_P].view(B, -1, D)
        s_t_P = s_t[mask_P].view(B, -1, D)
        cos_P = F.cosine_similarity(s_c_P, s_t_P, dim=-1)
        loss_P = (1 - cos_P).mean()
    else:
        loss_P = torch.tensor(0.0, device=device)

    # -- NON-EMPTY masked (Q set)
    mask_Q = mask_nonempty.bool()
    if mask_Q.any():
        s_c_Q = s_c[mask_Q]        # (Kq, D)
        s_t_Q = s_t[mask_Q]        # (Kq, D)
        cos_Q = F.cosine_similarity(s_c_Q, s_t_Q, dim=-1)
        loss_Q = (1 - cos_Q).mean()
    else:
        loss_Q = torch.tensor(0.0, device=device)

    L_jepa = alpha0 * loss_P + alpha1 * loss_Q

    # ---------------------------------------------------------
    # 2. Variance Regularization (VICReg-style)
    # Only on non-empty masked grids K = Q
    # ---------------------------------------------------------

    L_reg = torch.tensor(0.0, device=device)

    for b in range(B):
        idx = mask_Q[b]   # indices of non-empty masked grids in sample b

        if idx.any():
            # Context embeddings z_c[K]
            zc_K = z_c[b][idx]     # (M, D)
            # Predictor embeddings s_c[Q]
            sc_Q = s_c[b][idx]     # (M, D)

            vr1 = variance_regularization(zc_K, gamma=gamma)
            vr2 = variance_regularization(sc_Q, gamma=gamma)

            # cr1 = covariance_regularization(zc_K)
            # cr2 = covariance_regularization(sc_Q)

            # L_reg += beta1 * (vr1 + cr1) + beta2 * (vr2 + cr2) # Update with the covariance loss (for adding embedding diversity)
            L_reg += beta1 * (vr1 ) + beta2 * (vr2) # Drop the covariance because it's worsen the embedding quality


    L_reg = L_reg / B       # important: average per sample (per paper)

    # ---------------------------------------------------------
    # 3. Total JEPA loss
    # ---------------------------------------------------------
    loss = lambda_jepa * L_jepa + lambda_reg * L_reg

    return {
        "loss_total": loss,
        "loss_jepa": L_jepa,
        "loss_reg": L_reg,
        "loss_P_empty": loss_P,
        "loss_Q_nonempty": loss_Q
    }

def variance_regularization(z, gamma=1.0, eps=1e-4):
    """
    z: (M, D)
    Computes per-dimension variance hinge loss.
    """
    if z.numel() == 0:
        return torch.tensor(0.0, device=z.device)

    std = torch.sqrt(z.var(dim=0) + eps)  # (D,)
    return torch.mean(F.relu(gamma - std))

def drift_loss(z_c, prev_z_c):
    if prev_z_c is None:
        return torch.tensor(0.0, device=z_c.device)
    return F.mse_loss(z_c, prev_z_c)

def covariance_regularization(z):
    """
    z: (M, D)
    Returns Gram-matrix decorrelation penalty.
    """
    if z.numel() == 0:
        return torch.tensor(0.0, device=z.device)

    z_centered = z - z.mean(dim=0)
    M, D = z_centered.shape

    cov = (z_centered.T @ z_centered) / (M - 1)
    cov = cov.clone()
    cov.fill_diagonal_(0)

    return (cov.pow(2).sum() / D)

def vicreg_loss(z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0):
    sim_loss = F.mse_loss(z1, z2)
    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    N, D = z1.shape
    cov_z1 = (z1_centered.T @ z1_centered) / (N - 1)
    cov_z2 = (z2_centered.T @ z2_centered) / (N - 1)
    cov_z1.fill_diagonal_(0)
    cov_z2.fill_diagonal_(0)
    cov_loss = cov_z1.pow(2).sum() / D + cov_z2.pow(2).sum() / D
    return sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss

def compute_jepa_loss(    s_c, s_t,
                          z_c,
                          mask_empty,
                          mask_nonempty,
                          alpha0=0.25,
                          alpha1=0.75,
                          beta1=1.0,
                          beta2=1.0,
                          lambda_jepa=1.0,
                          lambda_reg=1.0,
                          gamma=1.0):

        return jepa_loss(
            s_c=s_c,
            s_t=s_t,
            z_c=z_c,
            mask_empty=mask_empty, # B, N
            mask_nonempty=mask_nonempty, # B, N
            alpha0=alpha0,
            alpha1=alpha1,
            beta1=beta1,
            beta2=beta2,
            lambda_jepa=lambda_jepa,
            lambda_reg=lambda_reg,
            gamma=gamma
        )
