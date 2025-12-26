import torch
import torch.nn as nn
import torch.nn.functional as F

class JointSynthesizerFusion(nn.Module):
    def __init__(self, traj_dim=128, gcn_dim=128, hidden=256):
        super().__init__()

        # auto detect if dimensions match
        self.same_dim = (traj_dim == gcn_dim)
        self.traj_dim = traj_dim
        self.gcn_dim = gcn_dim
        D = gcn_dim

        # if dims are different, project trajectory to graph dim
        if not self.same_dim:
            self.traj_proj = nn.Linear(traj_dim, gcn_dim)

        # Synthesizer scoring MLP (scalar attention)
        self.score_mlp = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # Value projection (node value vectors)
        self.value_proj = nn.Linear(D, D)

    def forward(self, s_traj, s_gcn, graph_mask=None):
        """
        s_traj:      [B, D_t]
        s_gcn:       [B, N, D_g]
        graph_mask:  [B, N]   (1 = real node, 0 = padded)
        """

        B, N, Dg = s_gcn.shape

        # -----------------------------------------------------
        # (1) Unify dimensionality of traj & graph embeddings
        # -----------------------------------------------------
        if self.same_dim:
            traj = s_traj.unsqueeze(1)   # [B,1,D]
            nodes = s_gcn               # [B,N,D]
            D = s_traj.size(-1)
        else:
            traj = self.traj_proj(s_traj).unsqueeze(1)
            nodes = s_gcn
            D = nodes.size(-1)

        # -----------------------------------------------------
        # (2) Concatenate: [traj ; node_1 ; node_2 ; ... node_N]
        # -----------------------------------------------------
        Z = torch.cat([traj, nodes], dim=1)     # [B, 1+N, D]

        # -----------------------------------------------------
        # (3) Synthesizer MLP for scalar attention scores
        # -----------------------------------------------------
        raw_scores = self.score_mlp(Z).squeeze(-1)   # [B, 1+N]
        node_scores = raw_scores[:, 1:]              # [B, N]

        # -----------------------------------------------------
        # (4) Apply mask to padded nodes BEFORE softmax
        # -----------------------------------------------------
        if graph_mask is not None:
            # graph_mask: 1 for real nodes, 0 for padded
            node_scores = node_scores.masked_fill(graph_mask == 0, -1e9)

        # masked softmax over nodes
        A = F.softmax(node_scores, dim=-1)           # [B, N]

        # -----------------------------------------------------
        # (5) Value projection for nodes
        # -----------------------------------------------------
        V = self.value_proj(nodes)                   # [B, N, D]

        # -----------------------------------------------------
        # (6) Final fused embedding (weighted sum)
        # -----------------------------------------------------
        fused = (A.unsqueeze(1) @ V).squeeze(1)      # [B, D]

        # -----------------------------------------------------
        # (7) Case-A contribution map: per-node × per-feature
        # -----------------------------------------------------
        node_contrib = A.unsqueeze(-1) * V           # [B, N, D]

        return fused, A, node_contrib
