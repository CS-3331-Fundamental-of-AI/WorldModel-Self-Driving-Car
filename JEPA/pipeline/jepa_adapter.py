from Utils.utilities import move_j1_to_device, build_graph_batch

class JEPAInputAdapter:
    def __init__(self, device, type2id, category2id, layer2id):
        self.device = device
        self.type2id = type2id
        self.category2id = category2id
        self.layer2id = layer2id

    def adapt(self, batch):
        out = dict(batch)

        # -----------------
        # JEPA-1
        # -----------------
        if out.get("j1") is not None:
            # leave as-is (lists); pipeline will stack
            pass

        # -----------------
        # JEPA-2
        # -----------------
        if out.get("j2") is not None:
            j2 = out["j2"]
            graph_feats, graph_adj = build_graph_batch(
                j2["graphs"],
                self.type2id,
                self.category2id,
                self.layer2id
            )

            assert graph_feats.shape[-1] == 13, \
                f"Expected 13 graph features, got {graph_feats.shape[-1]}"

            j2 = {
                "clean_deltas": j2["clean_deltas"].to(self.device),
                "traj_mask": j2["traj_mask"].to(self.device),
                "graph_feats": graph_feats.to(self.device),
                "graph_adj": graph_adj.to(self.device),
                "graph_mask": j2["graph_mask"].to(self.device),
            }
            out["j2"] = j2

        # -----------------
        # JEPA-3
        # -----------------
        if out.get("j3") is not None:
            j3 = out["j3"]
            out["j3"] = {
                "action": j3["action"].to(self.device),
                "spatial_x": j3["global_map"].to(self.device),
            }

        return out
