# --------------------------
# JEPA Input Adapter (FIXED)
# --------------------------
from Utils.utilities import build_graph_batch

class JEPAInputAdapter:
    def __init__(self, device, type2id, category2id, layer2id):
        self.device = device
        self.type2id = type2id
        self.category2id = category2id
        self.layer2id = layer2id

    def adapt(self, batch):
        out = dict(batch)

        # =========================
        # JEPA-1 (vision, frozen)
        # =========================
        if out.get("j1") is not None:
            j1 = out["j1"]
            if isinstance(j1, dict) and "pixel_values" in j1:
                j1["pixel_values"] = j1["pixel_values"].to(self.device)

        # =========================
        # JEPA-2 (PA + IA)
        # =========================
        if out.get("j2") is not None:
            j2 = out["j2"]

            graph_feats, graph_adj = build_graph_batch(
                j2["graphs"],
                self.type2id,
                self.category2id,
                self.layer2id,
            )

            assert graph_feats.shape[-1] == 13, \
                f"Expected 13 graph features, got {graph_feats.shape[-1]}"

            out["j2"] = {
                "clean_deltas": j2["clean_deltas"].to(self.device),
                "traj_mask": j2["traj_mask"].to(self.device),
                "action": j2["action"].to(self.device),   
                "graph_feats": graph_feats.to(self.device),
                "graph_adj": graph_adj.to(self.device),
                "graph_mask": j2["graph_mask"].to(self.device),
            }

        # =========================
        # JEPA-3 (global only)
        # =========================
        if out.get("j3") is not None:
            j3 = out["j3"]
            out["j3"] = {
                "global_nodes": [g.to(self.device) for g in j3["global_nodes"]],
                "global_edges": [g.to(self.device) for g in j3["global_edges"]],
            }

        return out
