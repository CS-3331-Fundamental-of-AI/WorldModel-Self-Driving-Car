import math
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 1. Gated CNN Encoder / Decoder
# ------------------------------

class GatedConvBlock(nn.Module):
    """
    1D gated convolution block:
    out = tanh(Wx) * sigmoid(Vx)
    Used for both encoder and decoder backbones.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv_feat = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        feat = self.conv_feat(x)
        gate = self.conv_gate(x)
        y = torch.tanh(feat) * torch.sigmoid(gate)
        y = self.norm(y)
        return y

class GatedCNNEncoder(nn.Module): # currently low-variability with the trajectories data
    """
    Encoder: trajectory window -> latent vector z0
    Input: [B, T, traj_dim]  (traj_dim=4: Δx, Δy, Δspeed, Δyaw)
    Output: [B, latent_dim]
    """
    def __init__(self, traj_dim: int = 6, latent_dim: int = 128, num_layers: int = 3, hidden_channels: int = 64):
        super().__init__()
        self.traj_dim = traj_dim
        self.latent_dim = latent_dim

        layers = []
        in_ch = traj_dim
        for _ in range(num_layers):
            layers.append(GatedConvBlock(in_ch, hidden_channels, kernel_size=3, padding=1))
            in_ch = hidden_channels

        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over time dimension
        self.proj = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, traj_dim] -> [B, traj_dim, T]
        x = x.transpose(1, 2)
        h = self.conv_stack(x)  # [B, hidden, T]
        h = self.pool(h).squeeze(-1)  # [B, hidden]
        z0 = self.proj(h)  # [B, latent_dim]
        return z0

class GatedCNNDecoder(nn.Module):
    """
    Decoder: latent (concatenated RVQ levels) -> reconstructed trajectory window
    Input: [B, num_levels * latent_dim]
    Output: [B, T, traj_dim]
    """
    def __init__(self, traj_dim: int = 6, latent_dim: int = 128, num_levels: int = 4, hidden_channels: int = 64, T: int = 5):
        super().__init__()
        self.traj_dim = traj_dim
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.T = T

        in_dim = num_levels * latent_dim
        self.fc = nn.Linear(in_dim, hidden_channels * T)

        # simple conv stack to refine trajectory over time
        self.deconv_stack = nn.Sequential(
            GatedConvBlock(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            GatedConvBlock(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Conv1d(hidden_channels, traj_dim, kernel_size=1)

    def forward(self, z_levels: List[torch.Tensor]) -> torch.Tensor:
        """
        z_levels: list of [B, latent_dim] for each level
        returns recon: [B, T, traj_dim]
        """
        z_concat = torch.cat(z_levels, dim=-1)  # [B, L * D]
        B = z_concat.size(0)

        h = self.fc(z_concat)             # [B, hidden * T]
        h = h.view(B, -1, self.T)         # [B, hidden, T]
        h = self.deconv_stack(h)          # [B, hidden, T]
        out = self.out_conv(h)            # [B, traj_dim, T]
        out = out.transpose(1, 2)         # [B, T, traj_dim]
        return out

# ------------------------------
# 2. SoftVQ Level & Residual RVQ
# ------------------------------

class SoftVQLevel(nn.Module):
    """
    A single SoftVQ level:
    - codebook: [K, D]
    - soft assignments with temperature tau
    """
    def __init__(self, latent_dim: int, num_codes: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        # codebook ~ N(0, 1/sqrt(D))
        self.codebook = nn.Parameter(torch.randn(num_codes, latent_dim) / math.sqrt(latent_dim))

    def forward(self, z_resid: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft quantization
        z_resid: [B, D]
        tau: scalar temperature
        Returns:
          z_soft: [B, D] (soft quantized latent)
          p: [B, K] (assignment probs)
        """
        B, D = z_resid.shape
        # distances: [B, K]
        # ||z - c||^2 = ||z||^2 + ||c||^2 - 2 z·c
        z_sq = (z_resid ** 2).sum(dim=-1, keepdim=True)          # [B, 1]
        c_sq = (self.codebook ** 2).sum(dim=-1).unsqueeze(0)     # [1, K]
        dot = z_resid @ self.codebook.t()                        # [B, K]
        dists = z_sq + c_sq - 2.0 * dot                          # [B, K]

        # softmax(-d / tau)
        p = F.softmax(-dists / tau, dim=-1)                      # [B, K]
        z_soft = p @ self.codebook                               # [B, D]
        return z_soft, p

    def hard_quantize(self, z_resid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hard nearest-neighbor quantization for inference.
        z_resid: [B, D]
        Returns:
          z_q: [B, D]
          indices: [B] in [0, K-1]
        """
        # [B, K]
        z_sq = (z_resid ** 2).sum(dim=-1, keepdim=True)
        c_sq = (self.codebook ** 2).sum(dim=-1).unsqueeze(0)
        dot = z_resid @ self.codebook.t()
        dists = z_sq + c_sq - 2.0 * dot

        indices = torch.argmin(dists, dim=-1)  # [B]
        z_q = self.codebook[indices]           # [B, D]
        return z_q, indices

class SoftResidualVQ(nn.Module):
    """
    Multi-level Soft Residual VQ (RVQ) with whitening W.
    - L levels, K codes per level, latent_dim D.
    - Whitening is a learnable linear transform W: [D -> D]. 
    """
    def __init__(self, latent_dim: int = 128, num_levels: int = 4, num_codes: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.num_codes = num_codes

        # Whitening matrix W (SimVQ-style latent basis)
        self.whitening = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.eye_(self.whitening.weight)   # Identity whitening

        # Codebooks per level
        self.levels = nn.ModuleList([
            SoftVQLevel(latent_dim, num_codes=num_codes)
            for _ in range(num_levels)
        ])

    def forward(self, z0: torch.Tensor, tau: float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Soft RVQ forward (training).
        z0: [B, D] (encoder latent)
        tau: current temperature
        Returns:
          z_soft_levels: list of [B, D] per level
          p_levels: list of [B, K] per level
        """
        # # Whitening
        z_resid = self.whitening(z0)  # [B, D]

        # Dirrect Feed
        # z_resid =  z0 # self.whitening(z0)  # [B, D]
        z_soft_levels = []
        p_levels = []

        for level in self.levels:
            z_soft, p = level(z_resid, tau)  # [B,D], [B,K]
            z_soft_levels.append(z_soft)
            p_levels.append(p)
            # residual for next level
            z_resid = z_resid - z_soft

        return z_soft_levels, p_levels

    @torch.no_grad()
    def encode_hard_tokens(self, z0: torch.Tensor) -> torch.Tensor:
        """
        Hard RVQ encoding for inference, following the pseudocode in the PDF.
        Returns:
          tokens: [B, num_levels] with global token IDs:
                  token_i = k_star + (i-1)*K
        """
        B, D = z0.shape
        z_resid = self.whitening(z0)   # whitening [B, D] - drop for test
        # z_resid = z0 # self.whitening(z0)   # whitening [B, D]
        tokens = []

        for i, level in enumerate(self.levels):
            z_q, indices = level.hard_quantize(z_resid)  # indices in [0, K-1]
            # Wang-style offset to make IDs unique per level
            token_i = indices + i * self.num_codes       # [B]
            tokens.append(token_i)
            # residual update
            z_resid = z_resid - z_q

        tokens = torch.stack(tokens, dim=-1)  # [B, L]
        return tokens

    @torch.no_grad()
    def cvq_update(self, z0, z_soft_levels, p_levels, alphas):
        """
        Perform CVQ update for each level.
        z0:            [B, D]
        z_soft_levels: list of [B, D]
        p_levels:      list of [B, K]
        alphas:        list of float (per level)
        """

        # 1. Compute residuals exactly like forward pass:
        z_resid = self.whitening(z0)                     # r0
        # z_resid = z0 #self.whitening(z0)                     # r0

        residuals = []
        for z_soft in z_soft_levels:
            residuals.append(z_resid)
            z_resid = z_resid - z_soft                  # r_{l+1}

        # fallback anchors (strongest distribution)
        fallback = z0.detach()

        # 2. Apply CVQ per level:
        for lvl, level in enumerate(self.levels):
            inputs = residuals[lvl]                     # r_l
            p      = p_levels[lvl]                      # [B,K]
            alpha  = alphas[lvl]                        # avg: [0.1 → 0.6]

            cvq_update_codebook(
                level.codebook,
                inputs=inputs,
                p=p,
                alpha=alpha,
                fallback_inputs=fallback
            )

@torch.no_grad()
def cvq_update_codebook(codebook, inputs, p, alpha, fallback_inputs=None):
    """
    Online CVQ codebook update (per level).
    codebook:        [K, D] nn.Parameter
    inputs:          [B, D] or [B, T, D] flattened
    p:               [B, K]
    alpha:           drift strength (0.1–0.6)
    fallback_inputs: used when residual variance too small
    """

    # Flatten inputs if needed
    if inputs.dim() == 3:
        B, T, D = inputs.shape
        inputs = inputs.reshape(B*T, D)
        p = p.repeat_interleave(T, dim=0)

    B, D = inputs.shape
    K = codebook.shape[0]

    # --- 1. Compute usage per code ---
    usage = p.sum(dim=0) + 1e-8                      # [K]
    usage_norm = usage / usage.sum()

    # Dead code detection (5% of uniform)
    dead_mask = usage_norm < (0.05 / K)

    if dead_mask.sum() == 0:
        return  # No dead codes

    # --- 2. Compute feature anchors (centroids) ---
    # numerator: [K, D] = Σ p[b,k] * inputs[b]
    numerator = p.T @ inputs                          # [K, D]
    anchors = numerator / usage.unsqueeze(-1)         # [K, D]

    # --- 3. Fallback to richer signal if residual tiny ---
    if inputs.std() < 1e-3 and fallback_inputs is not None:
        anchors = fallback_inputs.mean(dim=0, keepdim=True).repeat(K, 1)

    # --- 4. EMA update for dead codes ---
    cb = codebook.data                                # [K, D]
    cb[dead_mask] = (1 - alpha) * cb[dead_mask] + alpha * anchors[dead_mask]
    codebook.data.copy_(cb)

# ------------------------------
# 3. FSQ - Finite Scale Quantizer
# ------------------------------

class FSQ(nn.Module):
    """
    Finite Scalar Quantization for d-dimensional latent.

    - Input:  z in [-1, 1]^d  (we'll tanh before calling this)
    - Levels: L discrete values per dimension.
      If L is odd, grid is symmetric, e.g. L=7 -> [-1, -2/3, -1/3, 0, 1/3, 2/3, 1].

    Returns:
      z_q:   quantized latent (with straight-through estimator)
      idx:   integer indices in [0, L-1] of shape [B, d]
    """

    def __init__(self, dim: int, levels: int = 7):
        super().__init__()
        assert levels >= 2, "FSQ needs at least 2 levels"
        self.dim = dim
        self.levels = levels

        # Precompute denominator (L-1) as buffer
        self.register_buffer("denom", torch.tensor(float(levels - 1)))

    def forward(self, z: torch.Tensor):
        """
        z: [B, d], values should be roughly in [-1, 1].
        """
        # Safety clamp
        z_clipped = torch.clamp(z, -1.0, 1.0)

        # Map [-1, 1] -> [0, L-1]
        scaled = (z_clipped + 1.0) * (self.levels - 1) / 2.0  # [B, d]

        # Round to nearest integer grid index
        idx = torch.round(scaled).long()                      # [B, d]
        idx = torch.clamp(idx, 0, self.levels - 1)

        # Map indices back to [-1, 1]
        idx_f = idx.float()
        z_q = 2.0 * idx_f / self.denom - 1.0                  # [B, d]

        # Straight-through estimator: forward uses z_q, backward sees z
        z_q_ste = z + (z_q - z).detach()

        return z_q_ste, idx

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Pure quantization (no STE), used for inference.
        """
        z_clipped = torch.clamp(z, -1.0, 1.0)
        scaled = (z_clipped + 1.0) * (self.levels - 1) / 2.0
        idx = torch.round(scaled).long()
        idx = torch.clamp(idx, 0, self.levels - 1)
        return idx

    @torch.no_grad()
    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Map integer codes back to grid values in [-1, 1].
        """
        idx_f = idx.float()
        z_q = 2.0 * idx_f / self.denom - 1.0
        return z_q

class TrajectoryTokenizerFSQ(nn.Module):
    def __init__(
        self,
        traj_dim: int = 6,
        T: int = 8,
        enc_latent_dim: int = 128,   # encoder output dim (z0)
        d_q: int = 6,                # FSQ bottleneck dimension
        fsq_levels: int = 7,         # L (number of scalar levels)
        enc_layers: int = 3,
        dec_hidden: int = 64,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.traj_dim = traj_dim
        self.T = T
        self.enc_latent_dim = enc_latent_dim
        self.d_q = d_q
        self.fsq_levels = fsq_levels
        self.use_layernorm = use_layernorm

        # Encoder: same as before
        self.encoder = GatedCNNEncoder(
            traj_dim=traj_dim,
            latent_dim=enc_latent_dim,
            num_layers=enc_layers,
            hidden_channels=dec_hidden,
        )

        # Optional: LayerNorm on encoder output before bottleneck
        if self.use_layernorm:
            self.ln = nn.LayerNorm(enc_latent_dim)

        # Bottleneck: 128 → d_q
        self.bottleneck = nn.Linear(enc_latent_dim, d_q)

        # FSQ quantizer on d_q dims
        self.quantizer = FSQ(dim=d_q, levels=fsq_levels)

        # Decoder
        self.decoder = GatedCNNDecoder(
            traj_dim=traj_dim,
            latent_dim=d_q,
            num_levels=1,
            hidden_channels=dec_hidden,
            T=T,
        )

    def forward(self, traj_window: torch.Tensor, tau=None) -> Dict[str, torch.Tensor]:
        # 1) Encode trajectories
        z0 = self.encoder(traj_window)  # [B, enc_latent_dim]

        # OPTIONAL: LayerNorm to stabilize activations before quantization
        if self.use_layernorm:
            z0 = self.ln(z0)

        # 2) Bottleneck + tanh → quantizer
        zq_in = self.bottleneck(z0)     # [B, d_q]
        zq_in = torch.tanh(zq_in)       # bounding to [-1,1]

        # 3) FSQ quantization
        zq_hat, tokens = self.quantizer(zq_in)

        # 4) Decode (single level)
        recon = self.decoder([zq_hat])

        return {
            "recon": recon,
            "z0": z0,            # normalized encoder output
            "zq_in": zq_in,
            "zq_hat": zq_hat,
            "tokens": tokens,
        }

    @torch.no_grad()
    def encode_tokens(self, traj_window: torch.Tensor) -> torch.Tensor:
        z0 = self.encoder(traj_window)
        if self.use_layernorm:
            z0 = self.ln(z0)
        zq_in = torch.tanh(self.bottleneck(z0))
        tokens = self.quantizer.encode(zq_in)
        return tokens
