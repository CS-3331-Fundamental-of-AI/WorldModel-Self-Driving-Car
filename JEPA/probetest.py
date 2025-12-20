import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import Image, display
import numpy as np

# -------------------------------
# Environment variables
# -------------------------------
map_csv = os.getenv(
    "MAP_CSV",
    "/kaggle/working/WorldModel-Self-Driving-Car/JEPA/JEPA_PrimitiveLayer/map_files_15k.csv"
)
map_root = os.getenv(
    "MAP_ROOT",
    "/kaggle/input/a-crude-data-set-converted-from-nuscene/local_maps/"
)

# -------------------------------
# Load dataset
# -------------------------------
from Utils.jepa1data import MapDataset
from Utils.mask import apply_mask
from JEPA_PrimitiveLayer import PrimitiveLayer
from Utils.utilities import up2  # for upsampling in JEPA1

dataset = MapDataset(map_csv_file=map_csv)
dataset.root_dir = map_root
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Take one batch
batch = next(iter(dataloader))
bev, mask_emp, mask_non, mask_union, mask_emp_np, mask_non_np, mask_union_np, ph, pw, img = batch

B = bev.shape[0]
if bev.ndim == 5:
    masked_img = bev.squeeze(1).float()
    unmasked_img = bev.squeeze(1).float()
else:
    masked_img = bev.float()
    unmasked_img = bev.float()

mask_empty_lat = up2(mask_emp_np.squeeze(1)).view(B, -1)
mask_non_lat   = up2(mask_non_np.squeeze(1)).view(B, -1)
mask_any_lat   = up2(mask_union_np.squeeze(1)).view(B, -1)

print("✅ Loaded real dataset batch")

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
# Cosine similarity check
# -------------------------------
B, N, C = s_c.shape
seq = s_c[0]
cos = F.cosine_similarity(seq[:-1], seq[1:], dim=-1)
print("Mean cosine similarity between sequential patches:", cos.mean().item())

# -------------------------------
# Heat map visualization: enhanced
# -------------------------------
Hc = Wc = int(N ** 0.5)
feat = s_c[0].cpu().numpy()

# PCA to 3 components (RGB)
feat_rgb = PCA(n_components=3).fit_transform(feat)
feat_rgb = feat_rgb.reshape(Hc, Wc, 3)

# Stretch contrast
feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min() + 1e-8)

# Show as RGB heatmap
plt.figure(figsize=(4,4))
plt.imshow(feat_rgb)
plt.axis('off')
plt.title("s_c Heatmap (PCA → RGB)")
plt.savefig("/kaggle/working/s_c_heatmap_rgb.png")
plt.close()
display(Image("/kaggle/working/s_c_heatmap_rgb.png"))

# Show first PCA component with colormap
plt.figure(figsize=(4,4))
plt.imshow(feat_rgb[:,:,0], cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("s_c First PCA Component")
plt.savefig("/kaggle/working/s_c_heatmap_component0.png")
plt.close()
display(Image("/kaggle/working/s_c_heatmap_component0.png"))

# -------------------------------
# Patch-to-patch difference maps
# -------------------------------
seq_2d = seq.reshape(Hc, Wc, C)  # (Hc, Wc, C)

# Horizontal differences (row-wise)
diff_h = np.linalg.norm(seq_2d[:,1:,:] - seq_2d[:,:-1,:], axis=-1)  # (Hc, Wc-1)
plt.figure(figsize=(4,4))
plt.imshow(diff_h, cmap='magma', interpolation='nearest')
plt.colorbar()
plt.title("Patch-to-Patch Horizontal Difference")
plt.savefig("/kaggle/working/s_c_diff_h.png")
plt.close()
display(Image("/kaggle/working/s_c_diff_h.png"))

# Vertical differences (column-wise)
diff_v = np.linalg.norm(seq_2d[1:,:,:] - seq_2d[:-1,:,:], axis=-1)  # (Hc-1, Wc)
plt.figure(figsize=(4,4))
plt.imshow(diff_v, cmap='magma', interpolation='nearest')
plt.colorbar()
plt.title("Patch-to-Patch Vertical Difference")
plt.savefig("/kaggle/working/s_c_diff_v.png")
plt.close()
display(Image("/kaggle/working/s_c_diff_v.png"))
