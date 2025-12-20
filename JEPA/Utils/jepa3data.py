import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
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

GLOBAL_MAP_ROOT = os.path.join(DATASET_PATH, "global_maps")


# ============================================================
# Tier 3 Dataset
# ============================================================

class Tier3Dataset(Dataset):
    """
    JEPA-3 Dataset:
      - action (acceleration, steering)
      - global map image
    """

    def __init__(self, scene_map):
        """
        scene_map: same mapping used by Tier2
                   scene_token -> list of json frames
        """
        super().__init__()
        self.scene_map = scene_map

        # Flatten index
        self.master_index = []
        for scene_token, frames in scene_map.items():
            for i in range(len(frames)):
                self.master_index.append((scene_token, i))

        # Image transform (match JEPA-1 style)
        self.transform = T.Compose([
            T.Grayscale(),          # or remove if RGB desired
            T.Resize((256, 256)),
            T.ToTensor(),           # [1, H, W]
        ])

        print(f"[Tier3Dataset] Total samples: {len(self.master_index)}")

    def __len__(self):
        return len(self.master_index)

    def __getitem__(self, idx):
        scene_token, frame_idx = self.master_index[idx]
        js = self.scene_map[scene_token][frame_idx]

        # --------------------------
        # 1. ACTION (from metadata)
        # --------------------------
        ego = js["ego_state"]

        accel = ego.get("acceleration_ms2", 0.0)
        steering = ego.get("rotation", {}).get("yaw", 0.0)  # fallback

        action = torch.tensor(
            [accel, steering],
            dtype=torch.float32
        )

        # --------------------------
        # 2. GLOBAL MAP IMAGE
        # --------------------------
        scene_name = js["global_context"]["scene_name"]
        map_file = f"{scene_name}.png"
        map_path = os.path.join(GLOBAL_MAP_ROOT, map_file)

        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Missing global map: {map_path}")

        img = Image.open(map_path)
        global_map = self.transform(img)  # [1, H, W]

        return {
            "action": action,
            "global_map": global_map,
            "meta": js
        }

def tier3_collate_fn(batch):
    actions = torch.stack([b["action"] for b in batch])         # [B, 2]
    global_maps = torch.stack([b["global_map"] for b in batch]) # [B, C, H, W]
    metas = [b["meta"] for b in batch]

    return {
        "action": actions,
        "global_map": global_maps,
        "meta": metas
    }
