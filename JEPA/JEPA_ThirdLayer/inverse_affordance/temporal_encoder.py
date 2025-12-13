import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Gated Conv 1D block (two-filter conv + gating) that produces token-wise temporal features
# ---------------------------------------------------------------------------

class Conv1D_Gated(nn.Module):
    """Two parallel 1D conv filters + gating (elementwise multiply with sigmoid of second)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv_a = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.conv_b = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # x: (B, C, T) -> conv -> (B, out_ch, T)
        a = self.conv_a(x)
        b = self.conv_b(x)
        gated = a * torch.sigmoid(b)
        return gated

class GCNNBlock(nn.Module):
    """
    One GCNN layer = gated conv + FiLM (beta, global gamma)
    Inputs:  (B, T, D)
    Outputs: tokens_j, beta_j, gamma_j
    """
    def __init__(self, in_dim, token_dim, conv_hidden=128):
        super().__init__()

        self.conv = Conv1D_Gated(in_ch=in_dim, out_ch=conv_hidden)
        self.to_token = nn.Conv1d(conv_hidden, token_dim, kernel_size=1)

        # FiLM
        self.beta_proj = nn.Linear(token_dim, token_dim)
        self.global_lin = nn.Linear(token_dim, token_dim)
        self.gamma_summary = nn.Linear(token_dim, token_dim)

    def forward(self, x):
        # x: (B,T,D)
        h = x.transpose(1,2)
        h = self.conv(h)
        h = self.to_token(h)
        tokens = h.transpose(1,2)    # (B,T,token_dim)

        # βᵢ — directional, per-token
        beta = self.beta_proj(tokens)    # (B,T,token_dim)

        # γᵢ — global, shared per layer
        gp = tokens.mean(dim=1)          # (B,token_dim)
        gctx = F.relu(self.global_lin(gp))
        gamma = self.gamma_summary(gctx) # (B,token_dim)
        gamma = gamma.unsqueeze(1).expand_as(beta)

        return tokens, beta, gamma

# ---------------------------------------------------------------------------
# TemporalActionEncoder (4-layer GCNN Stack)
# ---------------------------------------------------------------------------
class TemporalActionEncoder(nn.Module):
    def __init__(self, state_dim, token_dim=128, conv_hidden=128, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = state_dim
        for _ in range(num_layers):
            self.layers.append(GCNNBlock(in_dim, token_dim, conv_hidden))
            in_dim = token_dim

    def forward(self, state_seq):
        outputs = []
        x = state_seq
        for layer in self.layers:
            tokens, beta, gamma = layer(x)
            outputs.append((tokens, beta, gamma))
            x = tokens
        return outputs
