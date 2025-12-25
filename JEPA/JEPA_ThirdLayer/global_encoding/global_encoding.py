import torch
import torch.nn as nn
from typing import Optional

from .gcn_pyg import GCN_PYG
from .cube_mlp import CubeMLP
from ..shared.predictors import PredictorMLP

from JEPA_ThirdLayer.utils import EMAHelper, freeze

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

        # -----------------------------
        # Cube modules (online / target)
        # -----------------------------
        self.cube_online = CubeMLP(L=cube_L, M=cube_M, D=cube_D, out_dim=cube_out)
        self.cube_target = CubeMLP(L=cube_L, M=cube_M, D=cube_D, out_dim=cube_out)

        # -----------------------------
        # Global map encoder
        # -----------------------------
        self.node_embed = nn.Linear(3, 32)  # map 3-dim to GCN input dim
        self.global_gcn = GCN_PYG(in_feats=32, hidden=128, out_feats=cube_D, pool=None)
        
        # EMA from global cube online -> attention & modal fusing target
        self.ema_helper = EMAHelper(decay=0.999)
        freeze(self.cube_target)

        # -----------------------------
        # Aux predictor
        # -----------------------------
        self.pred_from_ctx = PredictorMLP(in_dim=cube_out, out_dim=cube_out)

        # Project incoming s_c features to cube_D
        self.s_c_proj = nn.Linear(s_c_dim, cube_D)
        
    # =====================================================
    # EMA UPDATE 
    # =====================================================
    @torch.no_grad()
    def update_ema(self):
        self.ema_helper.update(self.cube_online)
        self.ema_helper.assign_to(self.cube_target)
        
    @torch.no_grad()
    def init_ema(self):
        self.ema_helper.register(self.cube_online)
        self.ema_helper.assign_to(self.cube_target)

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(
        self,
        s_y: torch.Tensor,          # inverse affordance latent
        s_c: torch.Tensor,          # context
        s_tg: torch.Tensor,         # EMA future target (JEPA-2)
        global_nodes: Optional[torch.Tensor] = None,
        global_edges: Optional[torch.Tensor] = None,
        tokens_final: Optional[torch.Tensor] = None # optional (B, T, D) from inverse module
    ):
        B = s_y.shape[0]
        # --------------------------------------------------
        # x_tokens ← s_y (primary hypothesis)
        # --------------------------------------------------
        if tokens_final is not None:
            # use provided temporal tokens directly (preferred)
            x_tokens = tokens_final[:, :self.L, :self.D]  # [B, L, D]
        else:
            # if s_y is already (B, L, D)
            if s_y.ndim == 3:
                x_tokens = s_y[:, :self.L, :self.D]
            elif s_y.ndim == 2:
                # Expand single vector to L repeated tokens (simple fallback)
                x_tokens = s_y.unsqueeze(1).expand(-1, self.L, -1)[:, :self.L, :self.D]
            else:
                raise ValueError("s_y must be (B,D) or (B,L,D) when tokens_final not provided")

        # --------------------------------------------------
        # y_modal ← s_c (context → modal slots)
        # --------------------------------------------------
        if s_c.ndim == 3:           
            s_c_pool = s_c.mean(dim=1)  # collapse patches → [B, D]
        elif s_c.ndim == 4:
            s_c_pool = s_c.mean(dim=(2, 3))
        elif s_c.ndim == 2:
            s_c_pool = s_c
        else:
            raise ValueError(f"s_c has unsupported ndim={s_c.ndim}")

        s_c_proj = self.s_c_proj(s_c_pool)  # (B, D)
        y_modal = s_c_proj.unsqueeze(1).expand(B, self.M, self.D)  # [B, M, D]
        
        # --------------------------------------------------
        # z_channels ← s_tg (EMA future → gating)
        # --------------------------------------------------
        if s_tg.ndim == 2:
            z_channels = s_tg.unsqueeze(1).expand(B, self.M, self.D)
        elif s_tg.ndim == 3:
            z_channels = s_tg[:, :self.M, :self.D]
        else:
            raise ValueError("s_tg must be (B,D) or (B,M,D)")

        # -----------------------------
        # Online cube -> target cube output
        # -----------------------------
        s_tar = self.cube_online(x_tokens, y_modal, z_channels)  # [B, cube_out]

        # -----------------------------
        # Global map path via GCN -> context cube (stop-grad)
        # -----------------------------
        if global_nodes is not None and global_edges is not None:
            # global_nodes: [B, N, 3]
            x_nodes = self.node_embed(global_nodes)  # [B, N, 32]
            g_out = self.global_gcn(x_nodes, global_edges)  # [B, N, cube_D]
            
            N = g_out.shape[1]
            step = max(1, N // self.M)  # Partition nodes into M groups (robust to N < M)
            
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

            s_ctx = self.cube_target(
            x_tokens[:, :self.cube_target.L],
            y_ctx,
            z_channels
        ).detach()
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

def load_pretrained_gcn(self, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt.get("state", ckpt)
    self.gcn.load_state_dict(state, strict=True)

    # 🔒 freeze GCN
    for p in self.gcn.parameters():
        p.requires_grad = False

    self.gcn.eval()
    print("🧊 Global GCN loaded & frozen")
