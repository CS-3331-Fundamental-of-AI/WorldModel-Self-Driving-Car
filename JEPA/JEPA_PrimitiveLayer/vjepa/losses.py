import torch
import torch.nn.functional as F


def jepa_embedding_loss(
    z_hat,
    z_target,
    alpha=1.0,   # alignment weight
    beta=1.0,    # variance weight
    gamma=0.1,   # covariance weight
    eps=1e-4,
):
    """
    JEPA / VICReg-style embedding loss.

    Args:
        z_hat:    [B, N, D] predicted embeddings
        z_target: [B, N, D] target embeddings (stop-grad or frozen encoder)
    Returns:
        total_loss, loss_dict
    """

    # --------------------------------------------------
    # 1) Alignment loss (JEPA core)
    # --------------------------------------------------
    loss_align = F.l1_loss(z_hat, z_target)

    # --------------------------------------------------
    # 2) Variance loss (anti-collapse)
    # --------------------------------------------------
    def variance_loss(z):
        # flatten tokens: [B*N, D]
        z = z.reshape(-1, z.size(-1))
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(1.0 - std))

    loss_var = variance_loss(z_hat)

    # --------------------------------------------------
    # 3) Covariance loss (decorrelation)
    # --------------------------------------------------
    def covariance_loss(z):
        z = z.reshape(-1, z.size(-1))   # [B*N, D]
        z = z - z.mean(dim=0, keepdim=True)

        N = z.size(0)
        cov = (z.T @ z) / (N - 1)

        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / z.size(-1)

    loss_cov = covariance_loss(z_hat)

    # --------------------------------------------------
    # 4) Total loss
    # --------------------------------------------------
    total_loss = (
        alpha * loss_align
        + beta * loss_var
        + gamma * loss_cov
    )

    loss_dict = {
        "loss_total": total_loss,
        "loss_align": loss_align,
        "loss_var": loss_var,
        "loss_cov": loss_cov,
    }

    return total_loss, loss_dict