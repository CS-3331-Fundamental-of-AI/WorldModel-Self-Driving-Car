import torch
import torch.nn as nn
# ---------------------------------------------------------------------------
# Spatial Branch - apply convolutional encoder + FiLM modulation
# ---------------------------------------------------------------------------
class SpatialEncoderFiLM(nn.Module):
    def __init__(self, in_ch=64, base_ch=128, out_dim=256, n_res=4, film_dim=128):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_ch, base_ch, 3, padding=1),
                nn.BatchNorm2d(base_ch),
                nn.GELU(),
                nn.Conv2d(base_ch, base_ch, 3, padding=1),
                nn.BatchNorm2d(base_ch)
            )
            for _ in range(n_res)
        ])

        self.film_beta_proj = nn.ModuleList([
            nn.Linear(film_dim, base_ch) for _ in range(n_res)
        ])
        self.film_gamma_proj = nn.ModuleList([
            nn.Linear(film_dim, base_ch) for _ in range(n_res)
        ])

        self.project = nn.Conv2d(base_ch, out_dim, 1)
        self.act = nn.GELU()

    def forward(self, x, beta_list, gamma_list):
        """
        beta_list[i]: (B,film_dim)   
        gamma_list[i]: (B,film_dim)  ← this is global gamma from GCNNBlock
        """
        x = self.conv_in(x)

        for i, rb in enumerate(self.res_blocks):
            res = rb(x)

            beta = self.film_beta_proj[i](beta_list[i])     # (B,C)
            gamma = self.film_gamma_proj[i](gamma_list[i])  # (B,C)

            x = self.act(x + res)
            x = x * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)

        return self.project(x)