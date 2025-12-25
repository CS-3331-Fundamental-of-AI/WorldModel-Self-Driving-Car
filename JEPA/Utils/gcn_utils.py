# Utils/gcn_utils.py

import torch

def edges_to_adj(edges, num_nodes):
    """
    edges: [E, 2]
    returns [N, N] normalized adjacency
    """
    adj = torch.zeros(num_nodes, num_nodes, device=edges.device)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    adj[edges[:, 1], edges[:, 0]] = 1.0
    adj += torch.eye(num_nodes, device=edges.device)

    deg = adj.sum(dim=1, keepdim=True)
    return adj / deg.clamp(min=1.0)

def mask_nodes(nodes, mask_ratio=0.3):
    """
    nodes: [B, N, D]
    """
    B, N, D = nodes.shape
    mask = torch.rand(B, N, device=nodes.device) < mask_ratio

    masked = nodes.clone()
    masked[mask] = 0.0

    return masked, mask
