import os
import torch
from JEPA_PrimitiveLayer import PrimitiveLayer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------
# Environment variables
# -------------------------------
map_csv = os.getenv("MAP_CSV", "/kaggle/working/WorldModel-Self-Driving-Car/JEPA/JEPA_PrimitiveLayer/map_files_15k.csv")
map_root = os.getenv("MAP_ROOT", "/kaggle/input/a-crude-data-set-converted-from-nuscene/local_maps/")

# -------------------------------
# Try to use real dataset
# -------------------------------
use_dummy = True
try:
    from Utils.jepa1data import MapDataset
    from torch.utils.data import DataLoader

    dataset = MapDataset(map_csv_file=map_csv)
    dataset.root_dir = map_root  # override root_dir to correct path

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # MapDataset returns: bev, mask_emp, mask_non, mask_union, mask_emp_np, mask_non_np, mask_union_np, ph, pw, img
    batch = next(iter(dataloader))
    bev, mask_emp, mask_non, mask_union, mask_emp_np, mask_non_np, mask_union_np, ph, pw, img = batch

    # Convert to tensors expected by PrimitiveLayer
    masked_img   = bev.float()
    unmasked_img = bev.float()
    mask_empty_lat = mask_emp.flatten(start_dim=1)
    mask_non_lat   = mask_non.flatten(start_dim=1)
    mask_any_lat   = mask_union.flatten(start_dim=1)

    use_dummy = False
    print("✅ Using real dataset batch")

except Exception as e:
    print(f"⚠️ Real dataset not available, using dummy input: {e}")

# -------------------------------
# Fallback: dummy input
# -------------------------------
if use_dummy:
    B, C, H, W = 1, 3, 64, 64
    Hc, Wc = 8, 8
    masked_img   = torch.randn(B, C, H, W)
    unmasked_img = torch.randn(B, C, H, W)
    mask_empty_lat = torch.zeros(B, Hc*Wc, dtype=torch.int)
    mask_non_lat   = torch.zeros(B, Hc*Wc, dtype=torch.int)
    mask_any_lat   = torch.zeros(B, Hc*Wc, dtype=torch.int)
    print("✅ Using dummy input")

# -------------------------------
# Instantiate PrimitiveLayer
# -------------------------------
layer = PrimitiveLayer(embed_dim=128)
layer.eval()

# -------------------------------
# Forward pass
# -------------------------------
with torch.no_grad():
    z_c, s_c, z_t = layer(masked_img, unmasked_img, mask_empty_lat, mask_non_lat, mask_any_lat)

# -------------------------------
# Correlation test
# -------------------------------
B, N, C = s_c.shape
seq = s_c[0]              # first image in batch
right_neighbor = seq[1:]
left = seq[:-1]

cos = F.cosine_similarity(left, right_neighbor, dim=-1)
print("Mean cosine similarity between sequential patches:", cos.mean().item())

# -------------------------------
# Heat map visualization
# -------------------------------
Hc = Wc = int(N ** 0.5)  # assume square latent map
feat = s_c[0].cpu().numpy()  # first image
feat_rgb = PCA(n_components=3).fit_transform(feat)
feat_rgb = feat_rgb.reshape(Hc, Wc, 3)

# Normalize to [0,1]
feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min())

plt.figure(figsize=(4,4))
plt.imshow(feat_rgb)
plt.title("Heat map of s_c (PCA → RGB)")
plt.axis('off')
plt.show()
