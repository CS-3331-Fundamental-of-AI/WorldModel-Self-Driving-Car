import os
import json
import torch
from torch.utils.data import Dataset
from dotenv import load_dotenv

load_dotenv(".env")

# --------------------------
# Paths
# --------------------------
DATAROOT = os.getenv(
    "DATAROOT",
    "/kaggle/input/a-crude-data-set-converted-from-nuscene/metadata"
)

DATASET_PATH = os.getenv(
    "DATASET_PATH",
    "/kaggle/input/a-crude-data-set-converted-from-nuscene"
)

GLOBAL_GRAPH_ROOT = os.path.join(DATASET_PATH, "global_graphs", "json")


# ============================================================
# Tier 3 Dataset
# ============================================================

class Tier3Dataset(Dataset):
    """
    JEPA-3 Dataset:
      - action = [acceleration, steering_delta]
      - global_graph = graph JSON from dataset
    """

    def __init__(self, scene_map):
        super().__init__()
        self.scene_map = scene_map

        # Flatten (scene, frame)
        self.master_index = []
        for scene_token, frames in scene_map.items():
            for i in range(1, len(frames)):  # start from 1 for delta
                self.master_index.append((scene_token, i))

        # Cache global graphs
        self.global_graphs = {}

        print(f"[Tier3Dataset] Total samples: {len(self.master_index)}")

    def _load_global_graph(self, scene_name):
        if scene_name not in self.global_graphs:
            graph_path = os.path.join(GLOBAL_GRAPH_ROOT, f"{scene_name}.json")
            if not os.path.exists(graph_path):
                raise FileNotFoundError(graph_path)

            with open(graph_path, "r") as f:
                data = json.load(f)

            # Use 'root' if exists, else top-level dict
            graph_data = data.get("root", data)

            # Convert list of dicts to float tensor
            nodes_list = [
                [node["x"], node["y"], node.get("z", 0.0)]  # default z=0 if missing
                for node in graph_data["nodes"]
            ]
            nodes = torch.tensor(nodes_list, dtype=torch.float32)
            
            edges_list = [[e["source"], e["target"]] for e in graph_data["edges"]]
            edges = torch.tensor(edges_list, dtype=torch.long)
            edge_weights = torch.tensor([e.get("weight", 1.0) for e in graph_data["edges"]], dtype=torch.float)
            
            self.global_graphs[scene_name] = {"nodes": nodes, "edges": edges}

        return self.global_graphs[scene_name]


    def __len__(self):
        return len(self.master_index)

    def __getitem__(self, idx):
        scene_token, t = self.master_index[idx]

        js_t   = self.scene_map[scene_token][t]
        js_tm1 = self.scene_map[scene_token][t - 1]

        # --------------------------
        # ACTION (metadata)
        # --------------------------
        accel = js_t["ego_state"].get("acceleration_ms2", 0.0)

        yaw_t   = js_t["ego_state"]["rotation"]["yaw"]
        yaw_tm1 = js_tm1["ego_state"]["rotation"]["yaw"]
        steering = yaw_t - yaw_tm1

        action = torch.tensor([accel, steering], dtype=torch.float32)

        # --------------------------
        # GLOBAL GRAPH
        # --------------------------
        scene_name = js_t["global_context"]["scene_name"]
        global_graph = self._load_global_graph(scene_name)

        return {
            "action": action,
            "global_graph": global_graph,
        }

def tier3_collate_fn(batch):
    return {
        "action": torch.stack([b["action"] for b in batch]),
        "global_graph": [b["global_graph"] for b in batch],  # list of dicts with nodes & edges
    }
