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
        self.t1 = t1  # JEPA-1 (V-JEPA-2)
        self.t2 = t2  # JEPA-2 (PA + IA + EMA)
        self.t3 = t3  # JEPA-3
        self.adapter = adapter

    @torch.no_grad()
    def _get_s_c(self, batch):
        if "j1" not in batch or batch["j1"] is None:
            return None
        out = self.t1.forward_only(batch["j1"])
        return {"s_c": out["s_c"]}

    def step(self, batch):
        # --------------------------------------------------
        # 0) Adapt unified batch
        # --------------------------------------------------
        batch = self.adapter.adapt(batch)

        out1, out2, out3 = None, None, None

        # ==================================================
        # JEPA-1 (frozen)
        # ==================================================
        out1 = self._get_s_c(batch)

        # ==================================================
        # JEPA-2 (PA + IA)
        # ==================================================
        if batch.get("j2") is not None:
            batch_j2 = batch["j2"]

            device = next(self.t2.pa.parameters()).device

            traj       = batch_j2["clean_deltas"].to(device)
            traj_mask = batch_j2["traj_mask"].to(device)
            x_graph   = batch_j2["graph_feats"].to(device)
            adj       = batch_j2["graph_adj"].to(device)
            graph_mask = batch_j2["graph_mask"].to(device)

            # IA input
            action = batch_j2["action"].to(device)

            s_c = out1["s_c"].detach() if out1 else None

            out2 = self.t2.step(
                traj=traj,
                x_graph=x_graph,
                adj=adj,
                traj_mask=traj_mask,
                graph_mask=graph_mask,
                action=action,
                s_c=s_c,
            )

        # ==================================================
        # Stop-gradient barrier
        # ==================================================
        s_c  = out1["s_c"].detach() if out1 else None
        s_tg = out2["s_tg"].detach() if out2 else None
        s_y  = out2["s_y"].detach() if out2 else None

        # ==================================================
        # JEPA-3
        # ==================================================
        if s_c is not None and s_tg is not None and batch.get("j3") is not None:
            batch_j3 = batch["j3"]
            device = s_c.device

            global_nodes_list = batch_j3.get("global_nodes")
            global_edges_list = batch_j3.get("global_edges")

            if global_nodes_list is not None:
                global_nodes, global_edges = batch_global_graphs(
                    global_nodes_list,
                    global_edges_list,
                    device,
                )
            else:
                global_nodes = global_edges = None

            out3 = self.t3.step(
                s_c=s_c,
                s_tg=s_tg,
                s_y=s_y,
                global_nodes=global_nodes,
                global_edges=global_edges,
            )

        # ==================================================
        # Aggregate losses
        # ==================================================
        loss_j2 = out2["loss"] if out2 else 0.0
        loss_j2_pa = out2.get("loss_pa", 0.0) if out2 else 0.0
        loss_j2_ia = out2.get("loss_ia", 0.0) if out2 else 0.0
        # ---- JEPA-2 diagnostics ----
        grad_j2_pa = out2.get("grad_norm_pa", 0.0) if out2 else 0.0
        grad_j2_ia = out2.get("grad_norm_ia", 0.0) if out2 else 0.0

        vic_pa_var = out2.get("vicreg_pa_var", 0.0) if out2 else 0.0
        vic_ia_var = out2.get("vicreg_ia_var", 0.0) if out2 else 0.0

        
        # JEPA-3 losses
        loss_j3      = out3.get("total", 0.0)
        loss_j3_cos  = out3.get("cos", 0.0)
        loss_j3_nce  = out3.get("nce", 0.0)
        loss_j3_vic  = out3.get("vicreg", 0.0)
        tau          = out3.get("tau", float('nan'))

        total_loss = loss_j2 + loss_j3

        return {
            "loss": total_loss,
            "loss_j1": 0.0,
            "loss_j2": loss_j2,
            "loss_j2_pa": loss_j2_pa,
            "loss_j2_ia": loss_j2_ia,
            "loss_j3": loss_j3,
            "loss_j3_cos": loss_j3_cos,
            "loss_j3_nce": loss_j3_nce,
            "loss_j3_vic": loss_j3_vic,
            "tau": tau,
            
            # ----------------
            # diagnostics
            # ----------------
            "grad_j2_pa": grad_j2_pa,
            "grad_j2_ia": grad_j2_ia,
            "vic_j2_pa_var": vic_pa_var,
            "vic_j2_ia_var": vic_ia_var,
        }
