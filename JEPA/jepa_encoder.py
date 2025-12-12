import torch
import torch.nn as nn

from .JEPA_PrimitiveLayer import PrimitiveLayer
from .JEPA_SecondLayer import Tier2Module
from .JEPA_ThirdLayer import JEPA_Tier3_InverseAffordance, JEPA_Tier3_GlobalEncoding



class JEPA_Encoder(nn.Module):
    """
    Full 3-layer JEPA World Model.

    JEPA-1: visual primitive encoder
    JEPA-2: trajectory + local graph encoder
    JEPA-3: inverse affordance + global world encoder
    """

    def __init__(self):
        super().__init__()

        # -------------------------
        # JEPA-1 (Primitive)
        # -------------------------
        self.jepa1 = PrimitiveLayer()

        # -------------------------
        # JEPA-2 (Trajectory + Graph)
        # -------------------------
        self.jepa2 = Tier2Module(
            traj_dim=256,
            traj_out=128,
            node_feat_dim=13,
            gcn_hidden=128,
            gcn_out=128,
            fusion_hidden=256,
        )

        # -------------------------
        # JEPA-3a (Inverse Affordance)
        # -------------------------
        self.jepa3_inv = JEPA_Tier3_InverseAffordance(
            action_dim=2,
            kin_state_dim=64,
            kin_k=6,
            token_dim=128,
            spatial_in_ch=64,
            spatial_feat_dim=256,
            film_dim=128,
            pred_dim=128,
        )

        # -------------------------
        # JEPA-3b (Global Encoding)
        # -------------------------
        self.jepa3_glob = JEPA_Tier3_GlobalEncoding(
            cube_L=6,
            cube_M=4,
            cube_D=128,
            cube_out=128,
            s_c_dim=256,
        )


    def forward(
        self,
        masked_img,
        unmasked_img,
        mask_empty,
        mask_non,
        mask_any,
        traj,
        adj,
        x_graph,
        action,
        global_nodes=None,
        global_adj=None,
    ):
        # -------------------------
        # JEPA-1
        # -------------------------
        _, s_c_1, tokens_1 = self.jepa1(
            masked_img,
            unmasked_img,
            mask_empty,
            mask_non,
            mask_any,
        )

        # -------------------------
        # JEPA-2
        # -------------------------
        tier2_out = self.jepa2(traj, adj, x_graph)
        s_traj = tier2_out["traj_emb"]
        s_fusion = tier2_out["fusion"]

        # -------------------------
        # JEPA-3 Inverse
        # -------------------------
        inv_out = self.jepa3_inv(action, s_c_1)

        # -------------------------
        # JEPA-3 Global
        # -------------------------
        glob_out = self.jepa3_glob(
            s_tg_hat=inv_out["s_tg_hat"],
            s_c=s_fusion,
            global_nodes=global_nodes,
            global_adj=global_adj,
            tokens_final=inv_out["tokens_final"],
        )

        return {
            "s_c_1": s_c_1,
            "tokens_1": tokens_1,

            "traj_emb": s_traj,
            "fusion_emb": s_fusion,

            "s_tg_hat": inv_out["s_tg_hat"],

            "world_latent": glob_out["s_tar"],  # final world embedding
            "world_context": glob_out["s_ctx"],
            "world_pred": glob_out["pred_tar"],
        }
