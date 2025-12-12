# jepa_tier3.py
import torch
import torch.nn as nn

# import the two branch implementations (adjust paths to your layout)
from inverse_affordance.inverse_affordance import JEPA_Tier3_InverseAffordance
from global_encoding.global_encoding import JEPA_Tier3_GlobalEncoding

class JEPA_Tier3(nn.Module):
    """
    Top-level orchestrator for Tier-3: runs
      - inverse-affordance branch (action -> s_tg_hat, tokens, ...)
      - global-encoding branch (fuses s_tg_hat + world map -> s_tar, s_ctx)
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
        spatial_x: torch.Tensor,
        # inputs for global encoding (s_c should come from JEPA-1 or JEPA-2)
        s_c: torch.Tensor,
        global_nodes: torch.Tensor = None,
        global_adj: torch.Tensor = None,
        # optionally accept tokens to feed to global branch
        tokens_for_global: torch.Tensor = None,
        **kwargs
    ):
        # 1) run inverse-affordance branch
        inv_out = self.inv(action=action, spatial_x=spatial_x)
        # inv_out contains keys: "s_y","s_tg_hat","z_ca","tokens_final","beta_t","gamma_t"

        # 2) run global encoding branch using s_tg_hat (and optional tokens)
        glob_out = self.glob(
            s_tg_hat=inv_out["s_tg_hat"],
            s_c=s_c,
            global_nodes=global_nodes,
            global_adj=global_adj,
            tokens_final=tokens_for_global if tokens_for_global is not None else inv_out.get("tokens_final", None)
        )

        # 3) aggregate useful outputs for downstream (training / metrics)
        out = {
            "inv": inv_out,
            "glob": glob_out,
            # convenience top-level things
            "s_tg_hat": inv_out["s_tg_hat"],
            "z_ca": inv_out["z_ca"],
            "s_tar": glob_out["s_tar"],
            "s_ctx": glob_out["s_ctx"],
            "pred_tar": glob_out.get("pred_tar", None),
        }
        return out
