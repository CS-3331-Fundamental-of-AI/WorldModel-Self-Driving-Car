# pipeline/jepa_pipeline.py

class JEPAPipeline:
    """
    Steady-state parallel JEPA training pipeline.

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
        # --------------------------------------------------
        # JEPA-1: context encoder (student vs frozen teacher)
        # --------------------------------------------------
        out1 = self.t1.step(batch["masks"])  # updates JEPA-1 student only

        # --------------------------------------------------
        # JEPA-2: trajectory / graph encoder + EMA target
        # --------------------------------------------------
        out2 = self.t2.step(
            batch["traj"],
            batch.get("graph")
        )  # updates JEPA-2 student only + EMA

        # --------------------------------------------------
        # Stop-gradient: prevent cross-tower backprop
        # --------------------------------------------------
        s_c  = out1["s_c"].detach()   # JEPA-1 → JEPA-3 (no grad)
        s_tg = out2["s_tg"].detach()  # JEPA-2 → JEPA-3 (no grad)

        # --------------------------------------------------
        # JEPA-3: inverse affordance + global consistency
        # --------------------------------------------------
        out3 = self.t3.step(
            action=batch["action"],
            spatial_x=batch["spatial_x"],
            s_c=s_c,
            s_tg=s_tg,
            graph=batch.get("graph")
        )  # updates JEPA-3 student only + optional EMA

        # --------------------------------------------------
        # Aggregate losses (for logging only)
        # --------------------------------------------------
        total_loss = out1["loss"] + out2["loss"] + out3["loss"]

        return {
            "loss": total_loss,
            "loss_j1": out1["loss"],
            "loss_j2": out2["loss"],
            "loss_j3": out3["loss"],
            "loss_j3_inv": out3["loss_inv"],
            "loss_j3_glob": out3["loss_glob"],
        }
