# trainers/trainer_jepa2.py

import torch
from JEPA_SecondLayer.losses import *
from JEPA_SecondLayer.utils import update_ema
from config.config import CLIP_NORM, EMA_JEPA2


class JEPA2Trainer:
    def __init__(self, model, ema_model, optimizer, fsq_levels=16, aug_fn=None):
        """
        fsq_levels: number of discrete levels for FSQ token regularization
        aug_fn: function to augment trajectories for VICReg invariance
        """
        self.model = model
        self.ema_model = ema_model
        self.opt = optimizer
        self.fsq_levels = fsq_levels
        self.aug_fn = aug_fn

    def step(self, traj, x_graph, adj, traj_mask=None, graph_mask=None):
        """
            traj (torch.Tensor): [B, T, 6] — input trajectories (e.g., delta positions/velocities) for the batch.
            x_graph (torch.Tensor): [B, N, F] — node features for the graph, where N is number of nodes, F is feature dim.
            adj (torch.Tensor): [B, N, N] — adjacency matrices representing graph connectivity for each batch.
            traj_mask (torch.Tensor, optional): [B, T] — mask indicating valid trajectory timesteps.
            graph_mask (torch.Tensor, optional): [B, N] — mask indicating valid graph nodes.
        """

        # -----------------------------
        # 1. Clean trajectory forward
        # -----------------------------
        out_clean = self.model(
            traj=traj,
            adj=adj,
            x_graph=x_graph,
            traj_mask=traj_mask,
            graph_mask=graph_mask
        )

        # -----------------------------
        # 2. Augmented trajectory forward
        # -----------------------------
        if self.aug_fn is not None:
            traj_aug = self.aug_fn(traj)
        else:
            # small Gaussian noise as default augmentation
            traj_aug = traj + 0.01 * torch.randn_like(traj)

        out_aug = self.model(
            traj=traj_aug,
            adj=adj,
            x_graph=x_graph,
            traj_mask=traj_mask,
            graph_mask=graph_mask
        )

        # -----------------------------
        # 3. Compute total loss
        # -----------------------------
        loss_dict = total_tokenizer_loss_fsq(
            out_clean=out_clean,
            out_aug=out_aug,
            target_traj=traj,
            fsq_levels=self.fsq_levels
        )
        loss = loss_dict["loss_total"]

        # -----------------------------
        # 4. Backward + optimization
        # -----------------------------
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_NORM)
        self.opt.step()

        # -----------------------------
        # 5. EMA update
        # -----------------------------
        update_ema(self.ema_model, self.model, EMA_JEPA2)

        return {
            "loss": loss.detach(),
            "s_tg": out_clean["fusion"].detach(),  # latent target for JEPA-3
            "out_clean": out_clean,
            "out_aug": out_aug,
            "loss_dict": loss_dict
        }
