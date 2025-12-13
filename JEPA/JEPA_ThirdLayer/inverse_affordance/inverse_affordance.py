import torch
import torch.nn as nn
import copy

# ----------------------------------------------------------------------
# Import project-level modules
# ----------------------------------------------------------------------
from .kinematics import DeterministicKinematicBicycle
from .temporal_encoder import TemporalActionEncoder
from .spatial_film import SpatialEncoderFiLM
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
        # 4. Spatial Encoder with FiLM applied INSIDE convolutional blocks
        # ------------------------------------------------------------------
        self.spatial_encoder = SpatialEncoderFiLM(
            in_ch=spatial_in_ch,
            base_ch=128,
            out_dim=spatial_feat_dim,
            n_res=n_res_blocks,
            film_dim=film_dim
        )

        # Project spatial features to FiLM dim for fusion with tokens
        self.spatial_proj = nn.Conv2d(spatial_feat_dim, film_dim, 1)

        # ------------------------------------------------------------------
        # 5. Action → embedding
        # ------------------------------------------------------------------
        self.action_proj = nn.Linear(token_dim, film_dim)

        # ------------------------------------------------------------------
        # 6. Merge spatial & action features → z_ca
        # ------------------------------------------------------------------
        self.z_proj = nn.Sequential(
            nn.Linear(film_dim * 2, pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, pred_dim),
        )

        # ------------------------------------------------------------------
        # 7. Predictors
        # ------------------------------------------------------------------
        # Predict intermediate representation s_y
        self.pred_sy = PredictorMLP(spatial_feat_dim + pred_dim, pred_dim)

        # Predict target representation s_tg_hat
        self.pred_tg_1 = PredictorMLP(pred_dim, pred_dim)
        self.pred_tg_2 = PredictorMLP(pred_dim, pred_dim)

        # ------------------------------------------------------------------
        # 8. EMA target encoder (BYOL/JEPA style)
        # ------------------------------------------------------------------
        self.ema_target = copy.deepcopy(self.temporal_enc)
        freeze(self.ema_target)

        self.n_res_blocks = n_res_blocks

    # ======================================================================
    # FORWARD PASS
    # ======================================================================
    def forward(self, action, spatial_x):
        """
        Forward computation:
          - Rollout the kinematic model
          - Encode temporal tokens
          - Produce FiLM beta per layer, global gamma
          - Run spatial encoder with FiLM modulation
          - Predict s_y and s_tg^
        """
        B = action.size(0)

        # Ensure spatial input has expected channel count (FiLM backbone expects conv_in channels)
        if spatial_x.dim() == 3:
            spatial_x = spatial_x.unsqueeze(1)
        if spatial_x.shape[1] == 1:
            spatial_x = spatial_x.repeat(1, self.spatial_encoder.conv_in.in_channels, 1, 1)

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
        # 3. β (per token → per ResBlock FiLM beta)
        # --------------------------------------------------------------
        beta_t_film = self.film_beta_proj(beta_t)  # (B,T,film_dim)

        # --------------------------------------------------------------
        # 4. γ uses only GLOBAL step
        #    gamma_t[:,0] is the "global" gamma vector
        # --------------------------------------------------------------
        gamma_global = gamma_t[:, 0, :]              # (B,token_dim)
        gamma_film = self.film_gamma_proj(gamma_global)  # (B,film_dim)

        # Create per-block FiLM beta list
        if T == self.n_res_blocks:
            beta_list = [beta_t_film[:, i, :] for i in range(T)]
        else:
            # Uniform temporal sampling if blocks != tokens count
            idxs = torch.linspace(0, T - 1, steps=self.n_res_blocks).long().to(beta_t.device)
            beta_list = [beta_t_film[:, idx, :] for idx in idxs]

        # Global gamma shared across all blocks
        gamma_list = [gamma_film for _ in range(self.n_res_blocks)]

        # --------------------------------------------------------------
        # 5. Spatial encoder with FiLM modulation
        # --------------------------------------------------------------
        spatial_feat = self.spatial_encoder(spatial_x, beta_list, gamma_list)

        # Spatial feature pooled and projected
        s_c = self.spatial_proj(spatial_feat).mean([2, 3])  # (B,film_dim)

        # --------------------------------------------------------------
        # 6. Action embedding (mean pool tokens)
        # --------------------------------------------------------------
        s_a = self.action_proj(tokens_final.mean(1))  # (B,film_dim)

        # --------------------------------------------------------------
        # 7. Joint latent z_ca
        # --------------------------------------------------------------
        z_ca = self.z_proj(torch.cat([s_c, s_a], dim=-1))

        # --------------------------------------------------------------
        # 8. Predict s_y
        # --------------------------------------------------------------
        feat_flat = torch.cat([spatial_feat.mean([2, 3]), z_ca], -1)
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
