from Utils.jepa1data import MapDataset # replace with actual dataset class
import torch
from torch.utils.data import DataLoader

# Load dataset
dataset = MapDataset(csv_file="/kaggle/working/WorldModel-Self-Driving-Car/JEPA/JEPA_PrimitiveLayer/map_files_15k.csv")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Take first batch
masked_img, unmasked_img, mask_empty_lat, mask_non_lat, mask_any_lat = next(iter(dataloader))

from JEPA_PrimitiveLayer import PrimitiveLayer

layer = PrimitiveLayer(embed_dim=128)
layer.eval()

with torch.no_grad():
    z_c, s_c, z_t = layer(masked_img, unmasked_img, mask_empty_lat, mask_non_lat, mask_any_lat)
    
    B, N, C = s_c.shape
seq = s_c[0]  # first image in batch
right_neighbor = seq[1:]
left = seq[:-1]

cos = torch.nn.functional.cosine_similarity(left, right_neighbor, dim=-1)
print("Mean cosine similarity between sequential patches:", cos.mean().item())

