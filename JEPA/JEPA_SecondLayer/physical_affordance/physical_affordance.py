import torch
import torch.nn as nn
import torch.nn.functional as F
from .TrajTokenizer import TrajectoryTokenizerFSQ
from .TokenEmbedding import TokenEmbedding
from .TrajectoryEnc import TrajEncoder
from .GCN import GCN
from .JointSynthesizerFusion import JointSynthesizerFusion

# ---- Freeze utility ----
def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

# ==========================================
# Tier2Module (Trajectory Encoder + GCN + Fusion)
# ==========================================

class JEPA_Tier2_PhysicalAffordance(nn.Module):
    def __init__(
        self,
        traj_dim=256,       # token embedding dim
        traj_out=128,       # trajectory encoder dim
        node_feat_dim=13,
        gcn_hidden=128,
        gcn_out=128,
        fusion_hidden=256,
    ):
        super().__init__()

        # ------------------------------
        # 1. FSQ Tokenizer (Frozen)
        # ------------------------------
        self.trajectory_tok = TrajectoryTokenizerFSQ()
        freeze(self.trajectory_tok)

        # B × d_q → B × d_q × traj_dim
        self.token_emb = TokenEmbedding(d_model=traj_dim)

        # ------------------------------
        # 2. Trajectory Encoder (takes ORIGINAL temporal traj)
        # ------------------------------
        # input: [B, T, 6]
        # output: [B, 128]
        self.traj_enc = TrajEncoder(
            traj_dim=6,
            conv_channels=64,
            kernel=3,
            out_dim=traj_out,
        )

        # ------------------------------
        # 3. Local Graph Encoder (GCN)
        # ------------------------------
        self.gcn = GCN(
            in_feats=node_feat_dim,
            hidden=gcn_hidden,
            out_feats=gcn_out,
            pool=None,
        )

        # ------------------------------
        # 4. Trajectory ↔ Graph Fusion
        # ------------------------------
        self.joint_synthesizer = JointSynthesizerFusion(
            traj_dim=traj_out,
            gcn_dim=gcn_out,
            hidden=fusion_hidden
        )


    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, traj, adj, x_graph, traj_mask=None, graph_mask=None):
        """
        traj:        [B, T, 6]
        traj_mask:   [B, T]
        adj:         [B, N, N]
        x_graph:     [B, N, node_feat_dim]
        graph_mask:  [B, N]
        """

        B = traj.shape[0]

        # ------------------------------
        # 1. FSQ semantic tokenization (NOT used for temporal encoding)
        # ------------------------------
        fsq_tokens = self.trajectory_tok.encode_tokens(traj)        # [B, d_q]
        fsq_emb    = self.token_emb(fsq_tokens)                     # [B, d_q, traj_dim]

        # NOTE:
        # fsq_emb is semantic "style" vector
        # traj_enc(traj) is temporal-motion embedding

        # ------------------------------
        # 2. Encode ORIGINAL temporal trajectory
        # ------------------------------
        s_traj = self.traj_enc(traj, traj_mask)   # [B, 128]

        # ------------------------------
        # 3. Encode Graph
        # ------------------------------
        gcn_out = self.gcn(x_graph, adj)    # [B, N, 128]

        if graph_mask is not None:
            gcn_out = gcn_out * graph_mask.unsqueeze(-1)

        # ------------------------------
        # 4. Joint Fusion
        # ------------------------------
        fused_out, attn_weights, node_contrib = self.joint_synthesizer(
            s_traj, gcn_out, graph_mask=graph_mask
        )

        return {
            "traj_emb": s_traj,         # main trajectory embedding
            "graph_emb": gcn_out,       # per-node graph embedding
            "fusion": fused_out,        # [B, 128] #s_tg
            "node_level": node_contrib, # per-node contributions
            "attn": attn_weights,       # [B, N]
            "fsq_tokens": fsq_tokens,   # (optional for logging)
            "fsq_emb": fsq_emb,
        }

