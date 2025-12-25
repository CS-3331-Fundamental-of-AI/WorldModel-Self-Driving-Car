import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn_pyg import GCN_PYG

class GCNPretrainModel(nn.Module):
    def __init__(self, node_dim=3, hidden=128, out_dim=128):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, 32)
        self.gcn = GCN_PYG(
            in_feats=32,
            hidden=hidden,
            out_feats=out_dim,
            pool=None
        )
        self.decoder = nn.Linear(out_dim, node_dim)

    def forward(self, nodes, adj):
        """
        nodes: [B, N, 3]
        adj:   [B, N, N]
        """
        x = self.node_embed(nodes)
        h = self.gcn(x, adj)          # [B, N, D]
        recon = self.decoder(h)       # [B, N, 3]
        return recon, h


