# jepa_tier3.py
import torch
import torch.nn as nn

# import the two branch implementations (adjust paths to your layout)
from .inverse_affordance.inverse_affordance import JEPA_Tier3_InverseAffordance
from .global_encoding.global_encoding import JEPA_Tier3_GlobalEncoding

class JEPA_Tier3(nn.Module):
    """
    Top-level orchestrator for Tier-3: runs
      - inverse-affordance branch: action + s_c -> s_y
      - global-encoding branch (s_y, s_c, s_tg) + world map -> world latent
    Returns a unified dict for losses / downstream usage.
    """

    def __init__(self, inv_cfg=None, glob_cfg=None):
        super().__init__()
        inv_cfg = inv_cfg or {}
        glob_cfg = glob_cfg or {}

        # instantiate branches (pass config dicts as kwargs)
        self.inv = JEPA_Tier3_InverseAffordance(**inv_cfg)
        self.glob = JEPA_Tier3_GlobalEncoding(**glob_cfg)

    def forward(
        self,
        # inputs for inverse-affordance
        action: torch.Tensor,
        s_c: torch.Tensor,   
        # inputs for global encoding (s_tg should come from JEPA-2)
        s_tg: torch.Tensor,
        global_nodes: torch.Tensor = None,
        global_edges: torch.Tensor = None,
        # optionally accept tokens to feed to global branch
        tokens_for_global: torch.Tensor = None,
        **kwargs
    ):
        # --------------------------------------------------
        # 1) Inverse affordance
        # --------------------------------------------------
        inv_out = self.inv(action=action, s_c=s_c)
        # inv_out contains keys: "s_y","s_a","z_ca","tokens","beta_t","gamma_t"
        s_y = inv_out["s_y"]

        # --------------------------------------------------
        # 2) Global encoding
        # --------------------------------------------------
        glob_out = self.glob(
            s_y=s_y,
            s_c=s_c,
            s_tg=s_tg,
            global_nodes=global_nodes,
            global_edges=global_edges,
            tokens_final=tokens_for_global if tokens_for_global is not None else inv_out.get("tokens", None)
        )

        # --------------------------------------------------
        # 3) Aggregate outputs
        # --------------------------------------------------
        out = {
            "inv": inv_out,
            "glob": glob_out,
            "s_y": s_y,
            "z_ca": inv_out["z_ca"],
            "s_tar": glob_out["s_tar"],
            "s_ctx": glob_out["s_ctx"],
            "pred_tar": glob_out.get("pred_tar", None),
        }
        return out
