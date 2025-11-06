import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------
# GCN LAYER (unchanged)
# ------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: [B, N, D], adj: [B, N, N]
        h = torch.bmm(adj, x)
        return F.relu(self.fc(h))


# ------------------------------------
# GRAPH ENCODER (dynamic dimensions)
# ------------------------------------
class GraphEncoder(nn.Module):
    """
    GCN encoder + Graph Transformer.
    Automatically adapts to input node dimension.
    """
    def __init__(self, gcn_dim=128, latent_dim=256):
        super().__init__()
        self.gcn_dim = gcn_dim
        self.latent_dim = latent_dim

        # These layers will be created dynamically on first forward()
        self.gcn1 = None
        self.gcn2 = None

        # Transformer block (dimension: gcn_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gcn_dim,
            nhead=4,
            dim_feedforward=256,
            batch_first=True  # ✅ easier, no transpose needed
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Linear pooling to graph-level embedding
        self.fc = nn.Linear(gcn_dim, latent_dim)

    # ------------------------------------
    # Build GCN layers on-the-fly
    # ------------------------------------
    def _build_gcn_if_needed(self, in_dim):
        if self.gcn1 is None:
            self.gcn1 = GCNLayer(in_dim, self.gcn_dim)
            self.gcn2 = GCNLayer(self.gcn_dim, self.gcn_dim)

    # ------------------------------------
    # Forward pass
    # ------------------------------------
    def forward(self, x, adj):
        """
        x:   [B, N, D] (node features)
        adj: [B, N, N] (adjacency matrix)
        """
        B, N, D = x.shape

        # Build GCN layers dynamically using actual node dimension
        self._build_gcn_if_needed(D)

        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)

        # h shape: [B, N, gcn_dim]
        # batch_first=True → no transpose needed
        h = self.transformer(h)

        # Pool to graph-level vector
        graph_latent = self.fc(h.mean(dim=1))

        return graph_latent, h
