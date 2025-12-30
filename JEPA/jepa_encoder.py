import torch
import torch.nn as nn

from JEPA_PrimitiveLayer.vjepa.model import PrimitiveLayerJEPA
from .JEPA_SecondLayer import (
    JEPA_Tier2_PhysicalAffordance,
    JEPA_Tier2_InverseAffordance,
)
from .JEPA_ThirdLayer import JEPA_Tier3_GlobalEncoding


class JEPA_Encoder(nn.Module):
    """
    Full JEPA World Encoder (inference-safe)

    JEPA-1: Visual primitives
    JEPA-2a: Physical affordance (trajectory + graph)
    JEPA-2b: Inverse affordance (action-conditioned)
    JEPA-3: Global world encoding
    """

    def __init__(self, vision_encoder):
        super().__init__()

        # -------------------------------------------------
        # JEPA-1
        # -------------------------------------------------
        self.jepa1 = PrimitiveLayerJEPA(
            encoder=vision_encoder,
            grid_h=16,
            grid_w=16,
            enc_dim=1024,
            prim_dim=128,
        )

        # -------------------------------------------------
        # JEPA-2 Physical
        # -------------------------------------------------
        self.jepa2_phys = JEPA_Tier2_PhysicalAffordance(
            traj_dim=256,
            traj_out=128,
            node_feat_dim=13,
            gcn_hidden=128,
            gcn_out=128,
            fusion_hidden=256,
        )

        # -------------------------------------------------
        # JEPA-2 Inverse
        # -------------------------------------------------
        self.jepa2_inv = JEPA_Tier2_InverseAffordance(
            action_dim=2,
            kin_state_dim=64,
            kin_k=6,
            token_dim=128,
            film_dim=128,
            pred_dim=128,
        )

        # -------------------------------------------------
        # JEPA-3 Global
        # -------------------------------------------------
        self.jepa3 = JEPA_Tier3_GlobalEncoding(
            cube_L=6,
            cube_M=4,
            cube_D=128,
            cube_out=128,
            s_c_dim=128,
        )

    # =====================================================
    # Forward
    # =====================================================
    def forward(
        self,
        pixel_values,      # [B, C, H, W]
        traj,              # [B, T, 6]
        adj,               # [B, N, N]
        x_graph,           # [B, N, 13]
        action,            # [B, 2]
        global_nodes=None,
        global_edges=None,
    ):
        B = pixel_values.size(0)
        # -------------------------------------------------
        # JEPA-1: primitives
        # -------------------------------------------------
        # Determine input shape
        if pixel_values.dim() == 4:
            x_flat = pixel_values.permute(0, 3, 1, 2)  # [B, C, H, W]
            is_sequence = False
        elif pixel_values.dim() == 5:
            B, T, H, W, C = pixel_values.shape
            x = pixel_values.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
            is_sequence = True
            # Flatten sequence in **chunks**
            chunk_size = 64  # frames per chunk
            s_c_tokens_chunks = []
            s_c_proj_chunks = []
            for i in range(0, B*T, chunk_size):
                x_chunk = x.reshape(B*T, C, H, W)[i:i+chunk_size]  # [chunk, C, H, W]
                s_c_tokens_chunk, s_c_proj_chunk = self.jepa1(x_chunk)
                s_c_tokens_chunks.append(s_c_tokens_chunk)
                s_c_proj_chunks.append(s_c_proj_chunk)
            s_c_tokens_flat = torch.cat(s_c_tokens_chunks, dim=0)
            s_c_proj_flat = torch.cat(s_c_proj_chunks, dim=0)
            N, D = s_c_tokens_flat.shape[1], s_c_tokens_flat.shape[2]
            s_c_tokens = s_c_tokens_flat.view(B, T, N, D)
            s_c_proj = s_c_proj_flat.view(B, T, N, D)
        else:
            raise ValueError("pixel_values must be 4D or 5D")

        if not is_sequence:
            s_c_tokens, s_c_proj = s_c_tokens_flat, s_c_proj_flat


        # -------------------------------------------------
        # JEPA-2a: physical affordance
        # -------------------------------------------------
        phys_out = self.jepa2_phys(traj, adj, x_graph)
        s_traj = phys_out["traj_emb"]        # [B, 128]
        s_tg = phys_out["fusion"]            # [B, 256]

        # -------------------------------------------------
        # JEPA-2b: inverse affordance
        # -------------------------------------------------
        inv_out = self.jepa2_inv(
            action=action,
            s_c=s_c_tokens,
        )

        s_y = inv_out["s_y"]                 # [B, 128]
        tokens_final = inv_out["tokens"]     # [B, T, 128]

        # -------------------------------------------------
        # JEPA-3: global world encoding
        # -------------------------------------------------
        glob_out = self.jepa3(
            s_y=s_y,
            s_c=s_c_tokens,
            s_tg=s_tg,
            global_nodes=global_nodes,
            global_edges=global_edges,
            tokens_final=tokens_final,
        )

        return {
            # primitives
            "s_c_tokens": s_c_tokens,

            # tier2
            "traj_emb": s_traj,
            "s_tg": s_tg,
            "s_y": s_y,

            # world
            "world_tgt": glob_out["s_tar"],   # target (EMA, if graph exists)
            "world_ctx": glob_out["s_ctx"],     # student
            "world_latent": glob_out["pred_tar"],   # predicted target (RSSM input)
        }

