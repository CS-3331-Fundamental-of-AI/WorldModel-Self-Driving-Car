import torch
from JEPA_PrimitiveLayer import PrimitiveLayer
import torch.nn.functional as F

# -------------------------------
# Dummy input
# -------------------------------
B, C, H, W = 2, 3, 64, 64          # batch size 2, 64x64 RGB images
Hc, Wc = 8, 8                       # example latent map size
masked_img   = torch.randn(B, C, H, W)
unmasked_img = torch.randn(B, C, H, W)
mask_empty_lat = torch.zeros(B, Hc*Wc, dtype=torch.int)
mask_non_lat   = torch.zeros(B, Hc*Wc, dtype=torch.int)
mask_any_lat   = torch.zeros(B, Hc*Wc, dtype=torch.int)

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
# Correlation test: neighbor cosine similarity
# -------------------------------
B, N, C = s_c.shape
seq = s_c[0]              # first image in batch
right_neighbor = seq[1:]
left = seq[:-1]

cos = F.cosine_similarity(left, right_neighbor, dim=-1)
print("Mean cosine similarity between sequential patches:", cos.mean().item())
