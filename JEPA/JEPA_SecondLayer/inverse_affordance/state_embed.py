import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---------------------------------------------------------------------------
# Mapping from B x T x 4 -> B x T x 32 (notes: if batch-size small / variest -> worsen the performance)
# ---------------------------------------------------------------------------
"""

	•	Small / variable batch sizes: BN relies on batch statistics — if your batch size is small (or varies), statistics may be noisy, harming stability or generalization. 
        In those cases, alternatives like Layer Normalization (LN) or Group Normalization might be more stable.  ￼
        
	•	Inference-time behavior: During evaluation / deployment, batch statistics are replaced by running averages, 
        which may differ from train-time statistics — ensure careful handling. This is standard BN caveat.  ￼
        
	•	Over-normalization risk / loss of physical meaning: If you normalize too aggressively or embed too deeply, 
        you might lose the interpretability of raw physical values (x, y, yaw, v) — the embedding becomes abstract. 
        Depending on your downstream needs (e.g. planning, interpretable predictions), that may or may not be acceptable.
        
	•	Added overhead: Even a small linear + BN adds parameters and compute; but given small input dims and small 
        embed_dim, this overhead is minimal and usually worth it.

"""
class StateEmbedGN(nn.Module):
    def __init__(self, low_dim=4, embed_dim=32, num_groups=8):
        super().__init__()
        self.proj = nn.Linear(low_dim, embed_dim, bias=False)
        # ensure embed_dim divisible by num_groups
        assert embed_dim % num_groups == 0, "embed_dim must be divisible by num_groups"
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim, affine=True)
        # optionally add a nonlinearity
        # self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, low_dim)
        B, T, _ = x.shape
        h = self.proj(x)                     # (B, T, embed_dim)
        # GN expects (N, C, *), so treat T as “length”, embed_dim as “channels”
        h = h.transpose(1, 2)               # (B, embed_dim, T)
        h = self.gn(h)                      # normalized
        h = h.transpose(1, 2)               # back to (B, T, embed_dim)
        # h = self.act(h)  # optional non-linearity
        return h