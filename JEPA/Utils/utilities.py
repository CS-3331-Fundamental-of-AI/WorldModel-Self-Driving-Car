
import torch

import networkx as nx

# ----------------------------
# Upsample Helper
# ----------------------------
def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

def move_j1_to_device(batch_j1, device):
    """
    Move JEPA-1 batch data to the target device (CPU / GPU).

    Parameters
    ----------
    batch_j1 : tuple
        A tuple where each element corresponds to one JEPA-1 field.
        Each field is typically:
        - a list of torch.Tensors (batch dimension not stacked yet), or
        - a list of non-tensor data (numpy arrays, ints, metadata, etc.)

    Returns
    -------
    tuple
        Same structure as batch_j1, but with tensor lists moved to `device`.
        No stacking is done here.
    """

    new_batch = []

    for item in batch_j1:
        # Case 1: this field is a list/tuple of torch.Tensors
        # Example: [Tensor, Tensor, Tensor, ...]
        if isinstance(item, (list, tuple)) and isinstance(item[0], torch.Tensor):
            # Move each tensor in the list to the target device
            new_batch.append([x.to(device) for x in item])

        # Case 2: non-tensor data (numpy arrays, ints, metadata, etc.)
        # Keep it unchanged
        else:
            new_batch.append(item)

    # Return the same structure as input
    return tuple(new_batch)

def build_x_adj(G, type2id, category2id, layer2id):
    """
    Input:
        G    : networkx graph
    Output:
        x    : [N, 13] node feature matrix
        adj  : [N, N] adjacency matrix
    """
    N = G.number_of_nodes()
    F = 13  # fixed
    
    x = torch.zeros((N, F), dtype=torch.float32)

    for i, (node, data) in enumerate(G.nodes(data=True)):
        # --- Numeric features ---
        pos = data.get("pos", (0,0))
        global_pos = data.get("global_pos", (0,0))
        heading = data.get("heading", 0.0)
        speed = data.get("speed", 0.0)
        accel = data.get("acceleration", 0.0)
        size = data.get("size", [0.0,0.0,0.0])
        
        # --- Categorical IDs ---
        type_id = type2id.get(data.get("type"), 0)
        category_id = category2id.get(data.get("category"), 0)
        layer_id = layer2id.get(data.get("layer"), 0)
        
        x[i] = torch.tensor([
            pos[0], pos[1],
            global_pos[0], global_pos[1],
            heading,
            speed,
            accel,
            size[0], size[1], size[2],
            float(type_id),
            float(category_id),
            float(layer_id)
        ])

    # --- adjacency (dense) ---
    A = nx.to_numpy_array(G)
    adj = torch.tensor(A, dtype=torch.float32)

    return x, adj

def build_graph_batch(graph_list, type2id, category2id, layer2id):
    xs, adjs = [], []
    maxN = max([g.number_of_nodes() for g in graph_list])
    
    for G in graph_list:
        x, adj = build_x_adj(G, type2id, category2id, layer2id)
        
        # pad node features
        padN = maxN - x.size(0)
        if padN > 0:
            x = torch.cat([x, torch.zeros(padN, x.size(1))], dim=0)
            adj = torch.cat([
                torch.cat([adj, torch.zeros(adj.size(0), padN)], dim=1),
                torch.zeros(padN, maxN)
            ], dim=0)
        
        xs.append(x)
        adjs.append(adj)
    
    x_batch = torch.stack(xs, dim=0)     # [B, maxN, 13]
    adj_batch = torch.stack(adjs, dim=0) # [B, maxN, maxN]
    
    return x_batch, adj_batch

def batch_global_graphs(global_nodes_list, global_adj_list, device):
    B = len(global_nodes_list)
    N_max = max([g.shape[0] for g in global_nodes_list])
    F = global_nodes_list[0].shape[1]

    nodes_batch = torch.zeros(B, N_max, F, device=device)
    adj_batch   = torch.zeros(B, N_max, N_max, device=device)

    for i, (nodes, adj) in enumerate(zip(global_nodes_list, global_adj_list)):
        n = nodes.shape[0]
        nodes_batch[i, :n, :] = nodes
        if adj.ndim == 2:
            adj_batch[i, :n, :n] = adj
        else:
            raise ValueError("Adjacency must be [N,N]")

    return nodes_batch, adj_batch
