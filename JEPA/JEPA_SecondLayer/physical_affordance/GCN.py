import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

class GCN(nn.Module):
    """
    - Uses GCNConv (Kipf & Welling) (Graph Convolutional Network)
    - Supports batched inputs
    - Input:  x  [B, N, F]
             adj[B, N, N]
    - Output: pooled graph embedding [B, F_out]
              or per-node embeddings [B, N, F_out]
    - If Pooling Happens -> need Graph-masking (else - no need)
    """

    def __init__(self, in_feats, hidden, out_feats, pool="mean"):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)
        self.pool = pool

    # -------------------------------------------------------------
    # Convert dense adjacency + features → PyG graph batch
    # -------------------------------------------------------------
    def dense_to_pyg_batch(self, x, adj):
        """
        x:   [B, N, F]
        adj: [B, N, N]
        return:
            x_batch:      [sum_N, F]
            edge_index:   [2, E_total]
            batch_index:  [sum_N]   (graph ID per node)
        """
        B, N, F = x.shape
        device = x.device

        node_offsets = torch.arange(B, device=device) * N   # start idx for each graph

        # Build edge indices for every graph
        # ------------------------------------
        edge_index_list = []
        x_list = []
        batch_list = []

        for b in range(B):
            # features
            x_list.append(x[b])     # (N, F)

            # adjacency → edge list
            edge = adj[b].nonzero(as_tuple=False).t()   # [2, E_b]
            edge_index_list.append(edge + node_offsets[b])

            # batch index per node
            batch_list.append(torch.full((N,), b, dtype=torch.long, device=device))

        x_batch = torch.cat(x_list, dim=0)            # [B*N, F]
        edge_index = torch.cat(edge_index_list, dim=1) # [2, sum(E)]
        batch_index = torch.cat(batch_list, dim=0)     # [B*N]

        return x_batch, edge_index, batch_index

    # -------------------------------------------------------------
    # forward
    # -------------------------------------------------------------
    def forward(self, x, adj):
        """
        x:   [B, N, F]
        adj: [B, N, N]
        """
        # convert dense → pyg format
        x_batch, edge_index, batch_index = self.dense_to_pyg_batch(x, adj)

        # Layer 1
        h = F.relu(self.conv1(x_batch, edge_index))

        # Layer 2
        h = self.conv2(h, edge_index)   # [sum_nodes, out_feats]

        # Pooling
        if self.pool == "mean":
            out = global_mean_pool(h, batch_index)   # [B, out_feats]
        elif self.pool == "sum":
            out = global_add_pool(h, batch_index)    # [B, out_feats]
        else:
            # reshape back to [B, N, F_out]
            B, N, _ = x.shape
            out = h.view(B, N, -1)

        return out