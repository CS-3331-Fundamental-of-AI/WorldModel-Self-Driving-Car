import torch
from Utils.utilities import build_graph_batch 

class JEPAPipeline:
    """
    Steady-state parallel JEPA training pipeline compatible with UnifiedDataset.

    Invariants:
    - Each JEPA tower updates ONLY its own student parameters
    - No cross-tower gradient flow
    - Cross-tower interaction happens only via detached representations
      and slow EMA evolution
    """ 

    def __init__(self, t1, t2, t3, adapter):
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.adapter = adapter

    def step(self, batch):
        batch = self.adapter.adapt(batch)
        out1, out2 = None, None

        # ----------------------------
        # JEPA-1
        # ----------------------------
        if "j1" in batch and batch["j1"] is not None:
            (
                bev_list,
                mask_emp_list,
                mask_non_emp_list,
                mask_union_list,
                mask_emp_np_list,
                mask_non_emp_np_list,
                mask_union_np_list,
                ph_list,
                pw_list,
                img_list
            ) = batch["j1"]

            device = next(self.t1.model.parameters()).device
            bev = torch.stack(bev_list).to(device)
            mask_emp = torch.stack(mask_emp_list).to(device)
            mask_non_emp = torch.stack(mask_non_emp_list).to(device)
            mask_union = torch.stack(mask_union_list).to(device)

            mask_emp_np = torch.stack([torch.from_numpy(x).bool() for x in mask_emp_np_list]).to(device)
            mask_non_emp_np = torch.stack([torch.from_numpy(x).bool() for x in mask_non_emp_np_list]).to(device)
            mask_union_np = torch.stack([torch.from_numpy(x).bool() for x in mask_union_np_list]).to(device)

            out1 = self.t1.step((mask_emp, mask_non_emp, mask_union, mask_emp_np, mask_non_emp_np, mask_union_np, bev))

        # ----------------------------
        # JEPA-2
        # ----------------------------
        if "j2" in batch and batch["j2"] is not None:
            batch_j2 = batch["j2"]
            device = next(self.t2.model.parameters()).device

            traj = batch_j2["clean_deltas"].to(device)
            traj_mask = batch_j2["traj_mask"].to(device)
            x_graph = batch_j2["graph_feats"].to(device)
            adj = batch_j2["graph_adj"].to(device)
            graph_mask = batch_j2["graph_mask"].to(device)

            out2 = self.t2.step(traj=traj, x_graph=x_graph, adj=adj, graph_mask=graph_mask, traj_mask=traj_mask)

        # ----------------------------
        # Stop-gradient
        # ----------------------------
        s_c, s_tg = None, None
        if out1 is not None:
            s_c = out1["s_c"].detach()
        if out2 is not None:
            s_tg = out2["s_tg"].detach()

        # ----------------------------
        # JEPA-3
        # ----------------------------
        out3 = None
        has_context = (s_c is not None) or (s_tg is not None)
        has_j3 = ("j3" in batch) and (batch["j3"] is not None)

        if has_context and has_j3:
            batch_j3 = batch["j3"]
            device = s_c.device if s_c is not None else s_tg.device

            global_nodes_list = batch_j3.get("global_nodes")
            global_adj_list = batch_j3.get("global_adj")
            if global_nodes_list is not None and global_adj_list is not None:
                # Use utility function to batch graphs
                global_nodes_batch, global_adj_batch = build_graph_batch(
                    global_nodes_list, global_adj_list, device
                )
            else:
                global_nodes_batch, global_adj_batch = None, None

            out3 = self.t3.step(
                action=batch_j3.get("action"),
                s_c=s_c,
                s_tg=s_tg,
                global_nodes=global_nodes_batch,
                global_adj=global_adj_batch,
            )

        # ----------------------------
        # Aggregate losses
        # ----------------------------
        total_loss = 0.0
        loss_j1 = out1["loss"] if out1 else 0.0
        loss_j2 = out2["loss"] if out2 else 0.0
        loss_j3 = out3["loss"] if out3 else 0.0
        loss_j3_inv = out3.get("loss_inv", 0.0) if out3 else 0.0
        loss_j3_glob = out3.get("loss_glob", 0.0) if out3 else 0.0
        total_loss = loss_j1 + loss_j2 + loss_j3

        return {
            "loss": total_loss,
            "loss_j1": loss_j1,
            "loss_j2": loss_j2,
            "loss_j3": loss_j3,
            "loss_j3_inv": loss_j3_inv,
            "loss_j3_glob": loss_j3_glob
        }
