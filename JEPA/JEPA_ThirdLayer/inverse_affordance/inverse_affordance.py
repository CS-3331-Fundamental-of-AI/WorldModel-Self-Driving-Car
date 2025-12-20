import torch
import torch.nn as nn
import copy

# ----------------------------------------------------------------------
# Import project-level modules
# ----------------------------------------------------------------------
from .kinematics import DeterministicKinematicBicycle
from .temporal_encoder import TemporalActionEncoder
from ..shared.predictors import PredictorMLP
from .utils import freeze
# ----------------------------------------------------------------------

class JEPA_Tier3_InverseAffordance(nn.Module):
    """
    JEPA Tier-3 module implementing:
      - Deterministic kinematic rollout (action → latent state sequence)
      - Temporal token encoder (GCNN-based)
      - FiLM-modulated spatial encoder (spatial_x → spatial_feat)
      - Fusion (s_c ⊕ s_a → z_ca)
      - Predictors for s_y and s_tg_hat

    OUTPUTS:
      s_y        : intermediate action-conditioned spatial latent
      s_tg_hat   : predicted target latent
      z_ca       : fused latent from action+spatial streams
      tokens     : last-layer temporal tokens
      beta_t     : raw per-timestep FiLM β before projection
      gamma_t    : raw per-timestep FiLM γ before projection
    """

    def __init__(
        self,
        action_dim=2,
        kin_state_dim=64,
        kin_k=6,
        token_dim=128,
        spatial_in_ch=64,
        spatial_feat_dim=256,
        film_dim=128,
        pred_dim=128,
        n_res_blocks=4,
        ema_decay=0.995,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Kinematic model — produces high-dim latent state sequence
        # ------------------------------------------------------------------
        self.kin = DeterministicKinematicBicycle(
            state_dim=kin_state_dim,
            action_dim=action_dim,
            k=kin_k
        )

        # ------------------------------------------------------------------
        # 2. Temporal encoder — produces per-timestep tokens + raw FiLM
        #    TemporalActionEncoder output:
        #      tokens_final : (B,T,D)
        #      beta_t       : (B,T,D)
        #      gamma_t      : (B,T,D)
        # ------------------------------------------------------------------
        self.temporal_enc = TemporalActionEncoder(
            state_dim=kin_state_dim,
            token_dim=token_dim,
        )

        # ------------------------------------------------------------------
        # 3. FiLM projection heads
        #    - beta per token → per-block FiLM beta
        #    - gamma uses GLOBAL gamma (taken from gamma_t[:,0])
        # ------------------------------------------------------------------
        self.film_beta_proj = nn.Linear(token_dim, film_dim)
        self.film_gamma_proj = nn.Linear(token_dim, film_dim)
        
        # ------------------------------------------------------------------
        # 4. Action → embedding
        # ------------------------------------------------------------------
        self.action_proj = nn.Linear(token_dim, film_dim)

        # ------------------------------------------------------------------
        # 5. Merge spatial & action features → z_ca
        # ------------------------------------------------------------------
        self.z_proj = nn.Sequential(
            nn.Linear(film_dim * 2, pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, pred_dim),
        )

        # ------------------------------------------------------------------
        # 6. Predictors
        # ------------------------------------------------------------------
        # Predict intermediate representation s_y
        self.pred_sy = PredictorMLP(film_dim + pred_dim, pred_dim)

        # Predict target representation s_tg_hat
        self.pred_tg_1 = PredictorMLP(pred_dim, pred_dim)
        self.pred_tg_2 = PredictorMLP(pred_dim, pred_dim)

        # ------------------------------------------------------------------
        # 7. EMA target encoder (BYOL/JEPA style)
        # ------------------------------------------------------------------
        self.ema_target = copy.deepcopy(self.temporal_enc)
        freeze(self.ema_target)

        self.n_res_blocks = n_res_blocks

    # ======================================================================
    # FORWARD PASS
    # ======================================================================
    def forward(self, action, s_c):
        """
        Forward computation:
        - Rollout the kinematic model
        - Encode temporal tokens
        - Produce FiLM beta per layer, global gamma
        - Modulate spatial latent (s_c) using FiLM
        - Predict s_y and s_tg^
        """
        B = action.size(0)

        # --------------------------------------------------------------
        # 1. Kinematic rollout → (B, T, state_dim)
        # --------------------------------------------------------------
        kin_high = self.kin(action)

        # --------------------------------------------------------------
        # 2. Temporal encoder → last layer output
        #    Returns list of layers: take final
        # --------------------------------------------------------------
        layer_outputs = self.temporal_enc(kin_high)
        tokens_final, beta_t, gamma_t = layer_outputs[-1]  # each (B,T,D)

        B, T, D = tokens_final.shape

        # --------------------------------------------------------------
        # 3. β (per token → FiLM beta)
        # --------------------------------------------------------------
        beta_t_film = self.film_beta_proj(beta_t)  # (B,T,film_dim)

        # --------------------------------------------------------------
        # 4. γ uses only GLOBAL step
        #    gamma_t[:,0] is the "global" gamma vector
        # --------------------------------------------------------------
        gamma_global = gamma_t[:, 0, :]                   # (B,token_dim)
        gamma_film = self.film_gamma_proj(gamma_global)   # (B,film_dim)

        # --------------------------------------------------------------
        # 5. Apply FiLM directly on spatial latent s_c
        #    (s_c comes from JEPA-1, NOT image)
        # --------------------------------------------------------------
        beta = beta_t_film.mean(dim=1)    # (B,film_dim)
        gamma = gamma_film                # (B,film_dim)
        
        B = s_c.size(0)
        s_c_flat = s_c.reshape(B, -1)                      # flatten all but batch, shape [B, C*H*W]
        if not hasattr(self, "_s_c_proj_init"):
            # dynamically initialize projection if spatial dims unknown at init
            self.s_c_proj = nn.Linear(s_c_flat.size(1), self.s_c_proj.out_features).to(s_c.device)
            self._s_c_proj_init = True

        s_c_proj = self.s_c_proj(s_c_flat)          # project to [B, film_dim]
        s_c_mod = s_c_proj * (1 + gamma) + beta  

        # --------------------------------------------------------------
        # 6. Action embedding (mean pool tokens)
        # --------------------------------------------------------------
        s_a = self.action_proj(tokens_final.mean(1))  # (B,film_dim)

        # --------------------------------------------------------------
        # 7. Joint latent z_ca
        # --------------------------------------------------------------
        z_ca = self.z_proj(torch.cat([s_c_mod, s_a], dim=-1))

        # --------------------------------------------------------------
        # 8. Predict s_y
        # --------------------------------------------------------------
        feat_flat = torch.cat([s_c_mod, z_ca], dim=-1)
        s_y = self.pred_sy(feat_flat)

        # --------------------------------------------------------------
        # 9. Predict s_tg_hat
        # --------------------------------------------------------------
        s_tg_hat = self.pred_tg_2(self.pred_tg_1(s_y))

        # --------------------------------------------------------------
        # Return complete bundle for loss computation
        # --------------------------------------------------------------
        return {
            "s_y": s_y,
            "s_tg_hat": s_tg_hat,
            "z_ca": z_ca,
            "tokens_final": tokens_final,
            "beta_t": beta_t,
            "gamma_t": gamma_t,
        }
