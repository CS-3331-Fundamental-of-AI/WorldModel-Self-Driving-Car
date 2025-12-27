#JEPA/JEPA_ThirdLayer/global_encoding.py
import torch
import torch.nn as nn
from typing import Optional

from .gcn_pyg import GCN_PYG
from .cube_mlp import CubeMLP
from .predictors import PredictorMLP

from JEPA_ThirdLayer.utils import EMAHelper, freeze

class JEPA_Tier3_GlobalEncoding(nn.Module):
    """
    JEPA-3 Global Encoding

    Student (online):
        s_ctx = cube_online(s_y, s_c, s_tg)

    Teacher (EMA, stop-grad):
        s_tar = cube_target(GCN(global_graph))

    Loss:
        pred(s_ctx) ≈ s_tar
    """

    def __init__(
        self,
        cube_L: int = 6,
        cube_M: int = 4,
        cube_D: int = 128,
        cube_out: int = 128,
        s_c_dim: int = 256,
        ema_decay: float = 0.999,
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
        self.ema_helper = EMAHelper(decay=ema_decay)
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
    # PRETRAINED GCN LOADER
    # =====================================================
    def load_pretrained_gcn(
        self,
        ckpt_path: str,
        freeze_gcn: bool = True,
        strict: bool = False,
        verbose: bool = True,
    ):
        """
        Load pretrained global GCN weights (Stage-0 pretraining).

        Expected checkpoint format:
        {
            "state": state_dict
        }

        Only loads:
        - node_embed
        - global_gcn

        Args:
            ckpt_path: path to *.pt checkpoint
            freeze_gcn: whether to freeze GCN after loading
            strict: strict loading (False recommended)
            verbose: print loaded / skipped keys
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "state" not in ckpt:
            raise KeyError("Checkpoint missing 'state' key")

        state = ckpt["state"]

        # --------------------------------------------------
        # Filter compatible keys
        # --------------------------------------------------
        own_state = self.state_dict()
        load_state = {}
        skipped = []

        for k, v in state.items():
            if k.startswith("node_embed.") or k.startswith("global_gcn."):
                if k in own_state and own_state[k].shape == v.shape:
                    load_state[k] = v
                else:
                    skipped.append(k)

        missing, unexpected = self.load_state_dict(load_state, strict=False)

        # --------------------------------------------------
        # Freeze if requested
        # --------------------------------------------------
        if freeze_gcn:
            freeze(self.node_embed)
            freeze(self.global_gcn)

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        if verbose:
            print(f"✅ Loaded pretrained GCN from: {ckpt_path}")
            print(f"   Loaded keys: {len(load_state)}")
            print(f"   Keys loaded are for 'node_embed' and 'global_gcn' only.")

            if skipped:
                print(f"   ⚠️ Skipped (shape mismatch): {len(skipped)}")

            if missing:
                print(f"   ℹ️ Missing keys: {len(missing)}")
                print(f"   Ignore missing keys of other modules.")

            if unexpected:
                print(f"   ℹ️ Unexpected keys: {len(unexpected)}")
                
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
        # x_tokens ← s_y (primary hypothesis) from IA
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
        # y_modal ← s_c (context → modal slots) from JEPA-1
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
        # z_channels ← s_tg (EMA future → gating) from JEPA-2
        # --------------------------------------------------
        if s_tg.ndim == 2:
            z_channels = s_tg.unsqueeze(1).expand(B, self.M, self.D)
        elif s_tg.ndim == 3:
            z_channels = s_tg[:, :self.M, :self.D]
        else:
            raise ValueError("s_tg must be (B,D) or (B,M,D)")

        # ==================================================
        # STUDENT: s_ctx (cube_online)
        # ==================================================
        s_ctx = self.cube_online(
            x_tokens,
            y_modal,
            z_channels,
        )

        has_graph = global_nodes is not None and global_edges is not None
        
        # ==================================================
        # TEACHER: s_tar (GCN → cube_target, stop-grad)
        # ==================================================
        if has_graph:
            with torch.no_grad():
                x_nodes = self.node_embed(global_nodes)
                g_out = self.global_gcn(x_nodes, global_edges)  # [B, N, D]

                # pool nodes → M slots
                N = g_out.shape[1]
                step = max(1, N // self.M)

                y_tar = torch.stack(
                    [
                        g_out[:, i * step : min(N, (i + 1) * step)].mean(dim=1)
                        for i in range(self.M)
                    ],
                    dim=1,
                )

                s_tar = self.cube_target(
                    x_tokens[:, :self.cube_target.L],
                    y_tar,
                    z_channels,
                )
        else: 
            s_tar = None

        # Auxiliary prediction: s_ctx -> s_tar
        # -----------------------------
        pred_tar = self.pred_from_ctx(s_ctx) #z

        return {
            "s_tar": s_tar,
            "s_ctx": s_ctx,
            "pred_tar": pred_tar,
        }
