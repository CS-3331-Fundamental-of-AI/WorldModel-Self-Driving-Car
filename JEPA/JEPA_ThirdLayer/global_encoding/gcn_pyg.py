import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

class GCN_PYG(nn.Module):
    """
    GCN using torch_geometric’s GCNConv.

    Accepts dense batched adjacency matrices and converts them to PyG batch internally.

    Inputs:
        x:   [B, N, F]
        adj: [B, N, N]
    Output:
        If pool is 'mean' or 'sum': [B, out_feats]
        If pool is None:             [B, N, out_feats]
    """

    def __init__(self, in_feats, hidden, out_feats, pool="mean"):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)
        self.pool = pool

    def dense_to_pyg_batch(self, x, adj):
        """
        Convert dense adjacency + features to PyG batch format
        Returns: x_batch [sum_N, F], edge_index [2, E_total], batch_index [sum_N]
        """
        B, N, F_dim = x.shape
        device = x.device
        node_offsets = torch.arange(B, device=device) * N

        x_list, edge_list, batch_list = [], [], []

        for b in range(B):
            x_list.append(x[b])
            edge = adj[b].nonzero(as_tuple=False).t()
            edge_list.append(edge + node_offsets[b])
            batch_list.append(torch.full((N,), b, dtype=torch.long, device=device))

        x_batch = torch.cat(x_list, dim=0)
        edge_index = torch.cat(edge_list, dim=1)
        batch_index = torch.cat(batch_list, dim=0)

        return x_batch, edge_index, batch_index

    def forward(self, x, adj):
        """
        x:   [B, N, F]
        adj: [B, N, N]
        """
        # Convert to PyG batch
        x_batch, edge_index, batch_index = self.dense_to_pyg_batch(x, adj)

        # GCN layers
        h = F.relu(self.conv1(x_batch, edge_index))
        h = self.conv2(h, edge_index)

        # Pooling or reshape
        if self.pool == "mean":
            out = global_mean_pool(h, batch_index)   # [B, out_feats]
        elif self.pool == "sum":
            out = global_add_pool(h, batch_index)    # [B, out_feats]
        else:
            B, N, _ = x.shape
            out = h.view(B, N, -1)                   # [B, N, out_feats]

        return out
