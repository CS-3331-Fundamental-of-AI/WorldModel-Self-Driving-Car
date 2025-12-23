import torch
from Utils.utilities import batch_global_graphs


class JEPAPipeline:
    """
    Steady-state parallel JEPA training pipeline.

    Invariants:
    - Each JEPA tower updates ONLY its own parameters
    - No cross-tower gradient flow
    - Cross-tower interaction only via detached representations
    """

    def __init__(self, t1, t2, t3, adapter):
        self.t1 = t1  # JEPA-1 (V-JEPA-2 based)
        self.t2 = t2  # JEPA-2 (trajectory / graph)
        self.t3 = t3  # JEPA-3 (inverse / global)
        self.adapter = adapter
        
    @torch.no_grad()
    def _get_s_c(self, batch):
        """Get frozen JEPA-1 representation"""
        if "j1" not in batch or batch["j1"] is None:
            return None
        out = self.t1.forward_only(batch["j1"])
        return out["s_c"]

    def step(self, batch):
        # --------------------------------------------------
        # 0) Adapt unified batch
        # --------------------------------------------------
        batch = self.adapter.adapt(batch)

        out1, out2, out3 = None, None, None

        # ==================================================
        # JEPA-1 (NEW: V-JEPA-2)
        # ==================================================
        # ----------------------------
        # (frozen)
        # ----------------------------
        s_c = self._get_s_c(batch)


        # ==================================================
        # JEPA-2 (trajectory / graph)
        # ==================================================
        if "j2" in batch and batch["j2"] is not None:
            batch_j2 = batch["j2"]
            device = next(self.t2.model.parameters()).device

            traj = batch_j2["clean_deltas"].to(device)
            traj_mask = batch_j2["traj_mask"].to(device)
            x_graph = batch_j2["graph_feats"].to(device)
            adj = batch_j2["graph_adj"].to(device)
            graph_mask = batch_j2["graph_mask"].to(device)

            out2 = self.t2.step(
                traj=traj,
                x_graph=x_graph,
                adj=adj,
                graph_mask=graph_mask,
                traj_mask=traj_mask,
            )

        # ==================================================
        # Stop-gradient barrier
        # ==================================================
        s_c = out1["s_c"].detach() if out1 is not None else None
        s_tg = out2["s_tg"].detach() if out2 is not None else None

        # ==================================================
        # JEPA-3 (contextual inverse + global)
        # ==================================================
        out3 = None
        has_context = (s_c is not None) and (s_tg is not None)
        has_j3 = ("j3" in batch) and (batch["j3"] is not None)

        if has_context and has_j3:
            batch_j3 = batch["j3"]
            device = s_c.device if s_c is not None else s_tg.device

            # -------- Global graph --------
            global_nodes_list = batch_j3.get("global_nodes")
            global_edges_list = batch_j3.get("global_edges")

            if global_nodes_list is not None and global_edges_list is not None:
                # Use utility function to batch graphs
                global_nodes_batch, global_edges_batch = batch_global_graphs(
                    global_nodes_list,
                    global_edges_list,
                    device,
                )
            else:
                global_nodes_batch, global_edges_batch = None, None

            # -------- JEPA-3 step --------
            out3 = self.t3.step(
                action=batch_j3.get("action"),
                s_c=s_c,
                s_tg=s_tg,
                global_nodes=global_nodes_batch,
                global_edges=global_edges_batch,
            )

        # ==================================================
        # Aggregate losses (no backward here!)
        # ==================================================
        loss_j2 = out2["loss"] if out2 is not None else 0.0
        loss_j2_inv = out2.get("loss_inv", 0.0) if out2 else 0.0
        loss_j2_var = out2.get("loss_var", 0.0) if out2 else 0.0
        loss_j2_cov = out2.get("loss_cov", 0.0) if out2 else 0.0

        loss_j3 = out3["loss"] if out3 is not None else 0.0
        loss_j3_inv = out3.get("loss_inv", 0.0) if out3 else 0.0
        loss_j3_glob = out3.get("loss_glob", 0.0) if out3 else 0.0

        loss_j1 = 0.0   # JEPA-1 frozen in this stage
        total_loss = loss_j2 + loss_j3


        return {
            "loss": total_loss,
            "loss_j1": loss_j1,
            # JEPA-2 VICReg losses
            "loss_j2": loss_j2,
            "loss_j2_inv": loss_j2_inv,
            "loss_j2_var": loss_j2_var,
            "loss_j2_cov": loss_j2_cov,
            # JEPA-3 task losses
            "loss_j3": loss_j3,
            "loss_j3_inv": loss_j3_inv,
            "loss_j3_glob": loss_j3_glob,
        }