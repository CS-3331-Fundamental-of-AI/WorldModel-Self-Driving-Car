import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoVideoProcessor

# --------------------------------------------------
# Global processor (same as Kaggle)
# --------------------------------------------------
processor = AutoVideoProcessor.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256"
)

# --------------------------------------------------
# Build DataFrame (same as Kaggle)
# --------------------------------------------------
def build_map_dataframe(root):
    files = sorted([
        f for f in os.listdir(root)
        if f.endswith(".png")
    ])

    df = pd.DataFrame({
        "path": [os.path.join(root, f) for f in files],
        "name": files,
    })

    df["map_id"] = df["name"].str.extract(r"(\d+)").astype(int)
    return df


# --------------------------------------------------
# Dataset (same as Kaggle)
# --------------------------------------------------
class MapDataset(Dataset):
    def __init__(self, df, image_size=256, patch_size=16, mask_ratio=0.5):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.Hp = image_size // patch_size
        self.Wp = image_size // patch_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["path"]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.NEAREST)

        # uniform random patch mask
        N = self.Hp * self.Wp
        num_mask = int(self.mask_ratio * N)

        mask = torch.zeros(N, dtype=torch.bool)
        mask[torch.randperm(N)[:num_mask]] = True
        patch_mask = mask.view(self.Hp, self.Wp)

        return {
            "image": img,
            "patch_mask": patch_mask,
        }


# --------------------------------------------------
# Collate (same as Kaggle)
# --------------------------------------------------
def collate_maps(batch):
    images = [b["image"] for b in batch]
    patch_masks = torch.stack([b["patch_mask"] for b in batch])

    # HF video processor expects List[List[PIL]]
    videos = [[img] for img in images]

    px = processor(
        videos=videos,
        return_tensors="pt"
    )["pixel_values_videos"]   # [B, 1, 3, H, W]

    return {
        "pixel_values": px,
        "patch_mask": patch_masks,
    }
