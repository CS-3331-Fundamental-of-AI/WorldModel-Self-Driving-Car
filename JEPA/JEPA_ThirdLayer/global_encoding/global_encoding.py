import torch
import torch.nn as nn
from typing import Optional

from .gcn_pyg import GCN_PYG
from .cube_mlp import CubeMLP
from ..shared.predictors import PredictorMLP


class JEPA_Tier3_GlobalEncoding(nn.Module):
    def __init__(
        self,
        cube_L: int = 6,
        cube_M: int = 4,
        cube_D: int = 128,
        cube_out: int = 128,
        s_c_dim: int = 256,  # incoming context feature dim (from predictor/proj)
    ):
        super().__init__()

        self.L = cube_L
        self.M = cube_M
        self.D = cube_D

        # Cube modules
        self.cube_online = CubeMLP(L=cube_L, M=cube_M, D=cube_D, out_dim=cube_out)
        self.cube_target = CubeMLP(L=cube_L, M=cube_M, D=cube_D, out_dim=cube_out)

        # Global map GCN -> produce node embeddings of dim cube_D
        self.global_gcn = GCN_PYG(in_feats=32, hidden=128, out_feats=cube_D, pool=None)

        # Small predictor from s_ctx -> s_tar (aux)
        self.pred_from_ctx = PredictorMLP(in_dim=cube_out, out_dim=cube_out)

        # Project incoming s_c features to cube_D
        self.s_c_dim = s_c_dim
        self.s_c_proj = nn.Linear(s_c_dim, cube_D)

    def forward(
        self,
        s_tg_hat: torch.Tensor,               # [B, D] or [B, L, D] OR pass tokens separately via tokens_final arg
        s_c: torch.Tensor,                    # (B, C) or (B, C, H, W)
        global_nodes: Optional[torch.Tensor] = None,
        global_edges: Optional[torch.Tensor] = None,
        tokens_final: Optional[torch.Tensor] = None # optional (B, T, D) from inverse module
    ):
        B = s_tg_hat.shape[0]

        # -----------------------------
        # Prepare tokens for x (preferred: tokens_final if provided)
        # -----------------------------
        if tokens_final is not None:
            # use provided temporal tokens directly (preferred)
            x_tokens = tokens_final[:, :self.L, :self.D]  # [B, L, D]
        else:
            # if s_tg_hat is already (B, L, D)
            if s_tg_hat.ndim == 3:
                x_tokens = s_tg_hat[:, :self.L, :self.D]
            elif s_tg_hat.ndim == 2:
                # Expand single vector to L repeated tokens (simple fallback)
                x_tokens = s_tg_hat.unsqueeze(1).expand(-1, self.L, -1)[:, :self.L, :self.D]
            else:
                raise ValueError("s_tg_hat must be (B,D) or (B,L,D) when tokens_final not provided")

        # -----------------------------
        # Prepare s_c -> y_modal and z_channels
        # -----------------------------
        # s_c may be (B, C) or (B, C, H, W)
        if s_c.ndim == 3:           # NEW
            s_c_pool = s_c.mean(dim=1)  # collapse patches → [B, D]
        elif s_c.ndim == 4:
            s_c_pool = s_c.mean(dim=(2, 3))
        elif s_c.ndim == 2:
            s_c_pool = s_c
        else:
            raise ValueError(f"s_c has unsupported ndim={s_c.ndim}")

        s_c_proj = self.s_c_proj(s_c_pool)  # (B, D)
        # Build M modality slots by simple replication; alternative: learned chunking
        y_modal = s_c_proj.unsqueeze(1).expand(B, self.M, self.D).contiguous()   # [B, M, D]
        z_channels = y_modal  # reuse; you could also apply a separate proj if desired

        # -----------------------------
        # Online cube -> target cube output
        # -----------------------------
        s_tar = self.cube_online(x_tokens, y_modal, z_channels)  # [B, cube_out]

        # -----------------------------
        # Global map path via GCN -> context cube (stop-grad)
        # -----------------------------
        if global_nodes is not None and global_edges is not None:
            g_out = self.global_gcn(global_nodes, global_edges)  # [B, N, D]
            N = g_out.shape[1]

            # Partition nodes into M groups (robust to N < M)
            step = max(1, N // self.M)
            y_ctx_list = []
            for i in range(self.M):
                start = i * step
                end = min(N, (i + 1) * step)
                if start >= end:
                    # fallback: zeros
                    y_ctx_list.append(torch.zeros(B, self.D, device=g_out.device, dtype=g_out.dtype))
                else:
                    chunk = g_out[:, start:end, :]        # [B, chunk, D]
                    y_ctx_list.append(chunk.mean(dim=1))  # [B, D]
            y_ctx = torch.stack(y_ctx_list, dim=1)  # [B, M, D]

            x_ctx = x_tokens[:, :self.cube_target.L, :self.D]
            z_ctx = y_ctx
            s_ctx = self.cube_target(x_ctx, y_ctx, z_ctx).detach()
        else:
            s_ctx = torch.zeros_like(s_tar)

        # -----------------------------
        # Auxiliary prediction: s_ctx -> s_tar
        # -----------------------------
        pred_tar = self.pred_from_ctx(s_ctx)

        return {
            "s_tar": s_tar,
            "s_ctx": s_ctx,
            "pred_tar": pred_tar,
        }
