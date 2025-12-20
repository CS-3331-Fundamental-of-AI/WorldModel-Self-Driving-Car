import torch
import torch.nn as nn
import copy

from .kinematics import DeterministicKinematicBicycle
from .temporal_encoder import TemporalActionEncoder
from ..shared.predictors import PredictorMLP
from .utils import freeze

class JEPA_Tier3_InverseAffordance(nn.Module):
    """
    JEPA Tier-3 Inverse Affordance (IA):
      - Rollout kinematics (action → latent states)
      - Temporal token encoding
      - FiLM-modulated spatial latent (s_c)
      - Fusion with action embedding → z_ca
      - Predict s_y and s_tg_hat

    Note: IA only uses s_c internally; it does NOT prepare s_c for the global encoder.
    """

    def __init__(self,
                 action_dim=2,
                 kin_state_dim=64,
                 kin_k=6,
                 token_dim=128,
                 s_c_dim=128,
                 film_dim=128,
                 pred_dim=128,
                 n_res_blocks=4,
                 ema_decay=0.995):
        super().__init__()

        # Kinematics
        self.kin = DeterministicKinematicBicycle(
            state_dim=kin_state_dim,
            action_dim=action_dim,
            k=kin_k
        )

        # Temporal encoder
        self.temporal_enc = TemporalActionEncoder(
            state_dim=kin_state_dim,
            token_dim=token_dim
        )

        # FiLM projections
        self.film_beta_proj = nn.Linear(token_dim, film_dim)
        self.film_gamma_proj = nn.Linear(token_dim, film_dim)

        # Action embedding
        self.action_proj = nn.Linear(token_dim, film_dim)

        # Spatial latent projection
        self.s_c_proj = nn.Linear(s_c_dim, film_dim)

        # Joint latent
        self.z_proj = nn.Sequential(
            nn.Linear(film_dim * 2, pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, pred_dim),
        )

        # Predictors
        self.pred_sy = PredictorMLP(film_dim + pred_dim, pred_dim)
        self.pred_tg_1 = PredictorMLP(pred_dim, pred_dim)
        self.pred_tg_2 = PredictorMLP(pred_dim, pred_dim)

        # EMA target encoder
        self.ema_target = copy.deepcopy(self.temporal_enc)
        freeze(self.ema_target)

        self.n_res_blocks = n_res_blocks

    def forward(self, action, s_c):
        B = action.size(0)

        # 1. Kinematics rollout
        kin_high = self.kin(action)

        # 2. Temporal encoding
        layer_outputs = self.temporal_enc(kin_high)
        tokens_final, beta_t, gamma_t = layer_outputs[-1]  # (B,T,D)

        # 3. FiLM projections
        beta_t_film = self.film_beta_proj(beta_t)           # (B,T,film_dim)
        beta = beta_t_film.mean(dim=1)                      # (B,film_dim)

        gamma_global = gamma_t[:, 0, :]                     # (B,token_dim)
        gamma = self.film_gamma_proj(gamma_global)          # (B,film_dim)

        # 4. FiLM-modulated spatial latent (s_c only for IA)
        s_c_pooled = s_c.mean(dim=(2,3))   # pool H,W → (B, C)
        s_c_proj = self.s_c_proj(s_c_pooled)  # (B, film_dim)
        s_c_mod = s_c_proj * (1 + gamma) + beta

        # 5. Action embedding
        s_a = self.action_proj(tokens_final.mean(1))        # (B,film_dim)

        # 6. Joint latent (combine raw s_c_proj and FiLM-modulated version)
        z_ca = self.z_proj(torch.cat([s_c_proj, s_c_mod], dim=-1))

        # 7. Predictors
        feat_flat = torch.cat([s_c_mod, z_ca], dim=-1)
        s_y = self.pred_sy(feat_flat)
        s_tg_hat = self.pred_tg_2(self.pred_tg_1(s_y))

        return {
            "s_y": s_y,
            "s_tg_hat": s_tg_hat,
            "s_a_detached": s_a.detach(),
            "z_ca": z_ca,
            "tokens_final": tokens_final,
            "beta_t": beta_t,
            "gamma_t": gamma_t
        }
