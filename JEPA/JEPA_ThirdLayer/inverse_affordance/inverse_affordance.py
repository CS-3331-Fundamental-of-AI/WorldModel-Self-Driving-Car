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
    """

    def __init__(self,
                 action_dim=2,
                 kin_state_dim=64,
                 kin_k=6,
                 token_dim=128,
                 s_c_dim=4096,
                 film_dim=128,
                 pred_dim=128,):
        super().__init__()

        # -------------------------------------------------
        # Kinematics + Temporal Encoder
        # -------------------------------------------------
        self.kin = DeterministicKinematicBicycle(
            state_dim=kin_state_dim,
            action_dim=action_dim,
            k=kin_k
        )

        self.temporal_enc = TemporalActionEncoder(
            state_dim=kin_state_dim,
            token_dim=token_dim
        )

        # -------------------------------------------------
        # FiLM projections
        # -------------------------------------------------
        self.beta_proj  = nn.Linear(token_dim, film_dim)
        self.gamma_proj = nn.Linear(token_dim, film_dim)

        # -------------------------------------------------
        # Latent projections
        # -------------------------------------------------
        
        # Action embedding
        self.action_proj = nn.Linear(token_dim, film_dim)

        # Spatial latent projection
        self.s_c_proj = nn.Linear(s_c_dim, film_dim)

        # Joint latent
        self.z_proj = nn.Sequential(
            nn.Linear(film_dim, pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, pred_dim),
        )

        # -------------------------------------------------
        # Predictors
        # -------------------------------------------------
        self.pred_sy = PredictorMLP(pred_dim, pred_dim)

    def forward(self, action, s_c):
        # -----------------------------------------------
        # 1. Rollout kinematics
        # -----------------------------------------------
        kin_seq = self.kin(action)

        # -----------------------------------------------
        # 2. Temporal encoding (last layer only)
        # -----------------------------------------------
        tokens, beta_t, gamma_t = self.temporal_enc(kin_seq)[-1]

        beta  = self.beta_proj(beta_t.mean(dim=1))
        gamma = self.gamma_proj(gamma_t.mean(dim=1))  # (B,T,D)

        
        # 3. FiLM projections
        #beta_t_film = self.film_beta_proj(beta_t)           # (B,T,film_dim)
        #beta = beta_t_film.mean(dim=1)                      # (B,film_dim)

        #gamma_global = gamma_t[:, 0, :]                     # (B,token_dim)
        #gamma = self.film_gamma_proj(gamma_global)          # (B,film_dim)
        
        # -----------------------------------------------
        # 3. Pool and project s_c
        # -----------------------------------------------
        if s_c.ndim == 4:
            s_c_pooled = s_c.mean(dim=(2, 3))
        elif s_c.ndim == 3:
            s_c_pooled = s_c.mean(dim=2)
        elif s_c.ndim == 2:
            # already B, C → use as is
            s_c_pooled = s_c
        else:
            raise ValueError(f"s_c has unsupported ndim={s_c.ndim}")    

        s_c_proj = self.s_c_proj(s_c_pooled)  # (B, film_dim)
        
        # -----------------------------------------------
        # 4. FiLM modulation
        # -----------------------------------------------
        s_c_mod = s_c_proj * (1.0 + gamma) + beta

        # -----------------------------------------------
        # 5. Action embedding
        # -----------------------------------------------
        s_a = self.action_proj(tokens.mean(dim=1))  

        # -----------------------------------------------
        # 6. Joint latent
        # -----------------------------------------------
        z_ca = self.z_proj(s_c_mod)
        
        # -----------------------------------------------
        # 7. Predict
        # -----------------------------------------------
        s_y = self.pred_sy(z_ca)

        return {
            "s_y": s_y,
            "s_a_detached": s_a.detach(),
            "z_ca": z_ca,
            "tokens": tokens,
            "beta_t": beta_t,
            "gamma_t": gamma_t
        }
