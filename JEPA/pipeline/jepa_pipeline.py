class JEPAPipeline:
    """
    Steady-state parallel JEPA training pipeline compatible with UnifiedDataset.

    Invariants:
    - Each JEPA tower updates ONLY its own student parameters
    - No cross-tower gradient flow
    - Cross-tower interaction happens only via detached representations
      and slow EMA evolution
    """

    def __init__(self, t1, t2, t3):
        self.t1 = t1  # JEPA-1 trainer
        self.t2 = t2  # JEPA-2 trainer
        self.t3 = t3  # JEPA-3 trainer

    def step(self, batch):
        """
        batch: dict with keys "j1" and/or "j2"
        - "j1" → tuple from MapDataset (JEPA-1)
        - "j2" → dict from Tier2Dataset (JEPA-2)
        """
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

            # Stack lists → tensors (THIS is what default collate used to do)
            bev = torch.stack(bev_list).to(device)
            mask_emp = torch.stack(mask_emp_list).to(device)
            mask_non_emp = torch.stack(mask_non_emp_list).to(device)
            mask_union = torch.stack(mask_union_list).to(device)

            mask_emp_np = torch.stack(mask_emp_np_list).to(device)
            mask_non_emp_np = torch.stack(mask_non_emp_np_list).to(device)
            mask_union_np = torch.stack(mask_union_np_list).to(device)

            out1 = self.t1.step(
                (
                    mask_emp,
                    mask_non_emp,
                    mask_union,
                    mask_emp_np,
                    mask_non_emp_np,
                    mask_union_np,
                    bev
                )
            )

        # ----------------------------
        # JEPA-2
        # ----------------------------
        if "j2" in batch and batch["j2"] is not None:
            batch_j2 = batch["j2"]
            device = next(self.t2.model.parameters()).device

            traj = batch_j2["clean_deltas"].to(device)
            traj_mask = batch_j2["traj_mask"].to(device)
            graph_feats = batch_j2["graph_feats"].to(device)
            graph_adj   = batch_j2["graph_adj"].to(device)
            graph_mask  = batch_j2["graph_mask"].to(device)

            out2 = self.t2.step(
                traj=traj,
                graph=(graph_feats, graph_adj, graph_mask),
                traj_mask=traj_mask,
                graph_mask=graph_mask
            )  # updates JEPA-2 student + EMA

        # ----------------------------
        # Stop-gradient: prevent cross-tower backprop
        # ----------------------------
        s_c, s_tg = None, None
        if out1 is not None:
            s_c = out1["s_c"].detach()
        if out2 is not None:
            s_tg = out2["s_tg"].detach()

        # ----------------------------
        # JEPA-3: inverse affordance + global consistency
        # ----------------------------
        # NOTE: adapt the input dict keys as per your actual JEPA-3 requirement
        batch_j3 = batch.get("j3", {})  # optional
        out3 = self.t3.step(
            action=batch_j3.get("action"),
            spatial_x=batch_j3.get("spatial_x"),
            s_c=s_c,
            s_tg=s_tg,
            graph=batch_j2 if "j2" in batch else None
        )

        # ----------------------------
        # Aggregate losses (for logging only)
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
