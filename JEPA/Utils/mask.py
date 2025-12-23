from config.config import MASK_RATIO, PATCH_SIZE
import torch
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from .color_analysis import analyze_bev_colors
from .patch_util import patchify, unpatchify
import torch.nn.functional as F
import math
from torchvision import transforms

def create_bev_mask_grid_occupancy(bev_rgb, mask_ratio=0.5, patch_size=16):
    """
    BEV Masking using TOLERANT semantic occupancy:

    EMPTY (P-set)       = patch is >= 95% background white
    NON-EMPTY (Q-set)   = patch contains > 5% non-white pixels
    """

    device = bev_rgb.device
    B, C, H, W = bev_rgb.shape

    # -----------------------------------------------------
    # 1. Tolerant white-background detection
    # -----------------------------------------------------
    bg = torch.tensor([255, 255, 255], device=device).view(1, 3, 1, 1)

    # Per-pixel difference (sum of abs diff across channels)
    diff = (bev_rgb.float() - bg.float()).abs().sum(dim=1)

    # A pixel counts as background if diff < 15 (tolerance)
    is_background = (diff < 15)        # (B,H,W)

    # Occupancy = NOT background
    occupancy = (~is_background).float().unsqueeze(1)   # (B,1,H,W)

    # -----------------------------------------------------
    # 2. Patchify occupancy map
    # -----------------------------------------------------
    tokens, ph, pw = patchify(occupancy, patch_size)
    N = ph * pw

    # tokens shape: (B, N, 1, p, p)
    tokens = tokens.view(B, N, 1, patch_size, patch_size)

    # Count non-white pixels inside each patch
    patch_occ = tokens.sum(dim=(2, 3, 4))       # (B,N)

    # Tolerant patch classification:
    patch_empty = (patch_occ < (patch_size*patch_size * 0.05))   # < 5% occupancy
    patch_nonempty = ~patch_empty

    # -----------------------------------------------------
    # 3. Random P / Q masking
    # -----------------------------------------------------
    mask_empty = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask_nonempty = torch.zeros(B, N, dtype=torch.bool, device=device)

    num_mask_total = int(mask_ratio * N)

    for b in range(B):

        empty_idx = patch_empty[b].nonzero().squeeze(-1)
        nonempty_idx = patch_nonempty[b].nonzero().squeeze(-1)

        num_P = min(len(empty_idx), num_mask_total // 2)
        num_Q = min(len(nonempty_idx), num_mask_total - num_P)

        if num_P > 0:
            perm = torch.randperm(len(empty_idx), device=device)[:num_P]
            mask_empty[b, empty_idx[perm]] = True

        if num_Q > 0:
            perm = torch.randperm(len(nonempty_idx), device=device)[:num_Q]
            mask_nonempty[b, nonempty_idx[perm]] = True

    mask_any = mask_empty | mask_nonempty

    # -----------------------------------------------------
    # 4. Upsample back to pixel mask
    # -----------------------------------------------------
    mask_grid = mask_any.view(B, 1, ph, pw).float()
    mask_pixel = F.interpolate(mask_grid, size=(H, W), mode="nearest").bool()

    return mask_empty, mask_nonempty, mask_any, mask_pixel, ph, pw

def create_non_empty_mask(
        imgfile: str,
        patch_size=PATCH_SIZE,
        mask_ratio_nonempty=MASK_RATIO,   # % of non-empty patches to mask
        mask_ratio_empty=0.0,      # % of empty patches to mask (optional)
        is_visualize=False
    ):
    # -------------------------------------------------------------
    # 1. Analyze colors
    # -------------------------------------------------------------
    color_analysis = analyze_bev_colors(imgfile, top_k=100)

    img = Image.open(imgfile).convert("RGB")
    arr = np.array(img)

    # remove background gray
    target_colors = [
      rgb for rgb, _ in color_analysis
      if rgb not in ([255, 255, 255], [0, 0, 0])
    ]

    # -------------------------------------------------------------
    # 2. Point sampling
    # -------------------------------------------------------------
    all_coords = []

    for r, g, b in target_colors:
        ys, xs = np.where(
            (arr[:, :, 0] == r) &
            (arr[:, :, 1] == g) &
            (arr[:, :, 2] == b)
        )

        if len(xs) == 0:
            continue

        idx = np.random.choice(len(xs), size=min(100, len(xs)), replace=False)
        xs_s = xs[idx]
        ys_s = ys[idx]

        coords = np.vstack([xs_s, ys_s]).T
        all_coords.append(coords)

    if len(all_coords) == 0:
        return None

    all_coords = np.vstack(all_coords)

    # -------------------------------------------------------------
    # 3. Spatial hierarchical clustering
    # -------------------------------------------------------------
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=40,
        linkage="ward"
    )
    labels = model.fit_predict(all_coords)
    clusters = [all_coords[labels == lbl] for lbl in np.unique(labels)]

    # -------------------------------------------------------------
    # 4. Create pixel-level circular mask
    # -------------------------------------------------------------
    H, W = arr.shape[:2]
    mask_pixel = np.zeros((H, W), dtype=np.uint8)

    for pts in clusters:
        xs, ys = pts[:, 0], pts[:, 1]
        cx, cy = int(xs.mean()), int(ys.mean())

        dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        radius = int(dists.max() * 1.25)

        yy, xx = np.ogrid[:H, :W]
        circle_mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
        mask_pixel[circle_mask] = 1

    # =============================================================
    # 5. Patchify version
    # =============================================================
    mask_torch = torch.tensor(mask_pixel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    tokens, ph, pw = patchify(mask_torch, patch_size)
    patch_sums = tokens.sum(dim=-1)          # (1, N)
    patch_nonempty = (patch_sums > 0)        # (1, N)
    patch_empty = ~patch_nonempty            # (1, N)

    # =============================================================
    # 6. Random Sampling Masking (NEW — JEPA-style partial masking)
    # =============================================================
    B, N = patch_nonempty.shape

    # --- Non-empty patches (object regions)
    K_idx = torch.where(patch_nonempty[0])[0]
    num_K_mask = int(mask_ratio_nonempty * len(K_idx))
    perm_K = torch.randperm(len(K_idx))
    K_mask_idx = K_idx[perm_K[:num_K_mask]]

    # --- Empty patches (optional)
    E_idx = torch.where(patch_empty[0])[0]
    num_E_mask = int(mask_ratio_empty * len(E_idx))
    perm_E = torch.randperm(len(E_idx))
    E_mask_idx = E_idx[perm_E[:num_E_mask]]

    # --- Final mask over patches
    patch_mask = torch.zeros_like(patch_nonempty)
    patch_mask[0, K_mask_idx] = 1
    patch_mask[0, E_mask_idx] = 1

    # Expand for unpatchify
    token_dim = patch_size * patch_size
    patch_mask_tokens = patch_mask.float().unsqueeze(-1).repeat(1, 1, token_dim)

    mask_pixel_restored = unpatchify(patch_mask_tokens, ph, pw, patch_size)
    mask_pixel_restored = mask_pixel_restored[0,0].cpu().numpy().astype(np.uint8)

    # -------------------------------------------------------------
    # 7. Visualization
    # -------------------------------------------------------------
    if is_visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(arr)
        plt.axis("off")
        plt.title("Cluster Circles (Pixel-Space)")

        for pts in clusters:
            xs, ys = pts[:, 0], pts[:, 1]
            cx, cy = int(xs.mean()), int(ys.mean())
            radius = int(np.sqrt((xs - cx)**2 + (ys - cy)**2).max() * 1.25)
            circ = plt.Circle((cx, cy), radius, edgecolor="cyan", fill=False, linewidth=2, alpha=0.8)
            plt.gca().add_patch(circ)

        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(mask_pixel_restored, cmap="gray")
        plt.title(f"Partial Non-Empty Mask (Q-set, {mask_ratio_nonempty * 100}% \n of obj been masked)")
        plt.colorbar()
        plt.show()

    # -------------------------------------------------------------
    # 8. Return final outputs
    # -------------------------------------------------------------
    return {
        "mask_pixel": mask_pixel,
        "patch_nonempty": patch_nonempty.cpu(),
        "patch_mask": patch_mask.cpu(),
        "mask_pixel_restored": mask_pixel_restored,
        "ph": ph,
        "pw": pw
    }

def masking(image_file: str, visualize=False, empty_mask_ratio = 0.25):
  # Load the user's BEV image
  img = Image.open(image_file).convert("RGB") # Take the
  arr = np.array(img)  # shape (H, W, 3)

  H, W, C = arr.shape # 512 x 512 x 3

  bev = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()

  # mask = 1 (WHITE), Not mask = 0 (DARK)
  mask_emp, _, _, _, ph, pw =create_bev_mask_grid_occupancy(bev, mask_ratio=empty_mask_ratio) # (1024,)

  mask_non_empty_dict = create_non_empty_mask(image_file, is_visualize=visualize )

  if mask_non_empty_dict is None:
    return None

  mask_non_emp_np = mask_non_empty_dict["patch_nonempty"][0].cpu().numpy()   # (1024,) - Q set
  mask_emp_np = mask_emp.cpu().numpy()[0]                    # (1024,) - P set

  # union of both P U Q
  mask_union = mask_non_emp_np | mask_emp_np                  # (1024,)

  # Convert to patch grid
  ph, pw = mask_non_empty_dict["ph"], mask_non_empty_dict["pw"]     # the height & width for the patch - H / PATCH_SIZE = pH, W / PATCH_SIZE = pW
  mask_grid = mask_union.reshape(ph, pw)

  if visualize:
    plt.figure(figsize=(6,6))
    plt.imshow(mask_emp[0].view(ph,pw).cpu().numpy(), cmap='gray')
    plt.title(f"Masked Empty (P-Set) {empty_mask_ratio * 100} % of Empty Region \n been masked")
    plt.axis('off')

    plt.figure(figsize=(6,6))
    plt.imshow(mask_grid, cmap='gray')
    plt.title("Patch-Level Mask Union Non-Empty & Empty (P U Q)")
    plt.show()

  transform = transforms.ToTensor()
  img = img = transform(img)                # <---- IMPORTANT
  return mask_emp_np, mask_non_emp_np, mask_union, ph, pw, bev, img

def apply_mask(bev, mask_emp_np, mask_non_emp_np, mask_any_np, visualize = False):
    # 1. Patchify BEV
    tokens, ph, pw = patchify(bev, patch_size=PATCH_SIZE)

    # 2. Convert mask to tensor
    mask_any = torch.tensor(mask_any_np, dtype=torch.bool)
    mask_emp = torch.tensor(mask_emp_np, dtype=torch.bool)
    mask_non_emp = torch.tensor(mask_non_emp_np, dtype=torch.bool)


    # 3. Apply mask in token space
    tokens_masked_emp = tokens.clone()
    tokens_masked_non_emp = tokens.clone()
    token_masked_any = tokens.clone()

    tokens_masked_emp[0, mask_emp] = 0
    tokens_masked_non_emp[0, mask_non_emp] = 0
    token_masked_any[0, mask_any] = 0


    bev_masked_emp = unpatchify(tokens_masked_emp, ph, pw, patch_size=PATCH_SIZE)
    bev_masked_non_emp  = unpatchify(tokens_masked_non_emp, ph, pw, patch_size=PATCH_SIZE)
    bev_masked_any = unpatchify(token_masked_any, ph, pw, patch_size=PATCH_SIZE)

    if visualize:
      img_emp = bev_masked_emp[0].permute(1,2,0).cpu().numpy().astype("uint8")
      img_non_emp = bev_masked_non_emp[0].permute(1,2,0).cpu().numpy().astype("uint8")
      img_any = bev_masked_any[0].permute(1,2,0).cpu().numpy().astype("uint8")

      plt.figure(figsize=(6,6))
      plt.imshow(img_emp)
      plt.axis("off")

      plt.figure(figsize=(6,6))
      plt.imshow(img_non_emp)
      plt.axis("off")


      plt.figure(figsize=(6,6))
      plt.imshow(img_any)
      plt.axis("off")

    return bev_masked_emp, bev_masked_non_emp, bev_masked_any

from PIL import ImageDraw

def patch_mask_to_token_mask(patch_mask, has_cls=False):
    """
    Spatial-only V-JEPA-2 masking (T = 1)

    patch_mask: [B, Hp, Wp]  (True = masked)
    returns:    [B, N]       where N = Hp * Wp
    """
    B, Hp, Wp = patch_mask.shape

    token_mask = patch_mask.reshape(B, Hp * Wp).bool()

    if has_cls:
        cls = torch.zeros(B, 1, dtype=torch.bool, device=patch_mask.device)
        token_mask = torch.cat([cls, token_mask], dim=1)

    return token_mask

def boolean_to_index_masks(bool_mask, pad_value=0):
    """
    bool_mask: [B, N] True = masked
    """
    B, N = bool_mask.shape
    context_list, target_list = [], []

    for b in range(B):
        target = torch.nonzero(bool_mask[b], as_tuple=False).flatten()
        context = torch.nonzero(~bool_mask[b], as_tuple=False).flatten()
        context_list.append(context)
        target_list.append(target)

    def pad(idxs):
        max_len = max(i.numel() for i in idxs)
        max_len = max(max_len, 1)  # avoid zero-length
        padded, valid = [], []

        for i in idxs:
            L = i.numel()
            pad_len = max_len - L
            padded.append(
                torch.cat([i, torch.full((pad_len,), pad_value, device=i.device)])
            )
            valid.append(
                torch.cat([torch.ones(L, dtype=torch.bool, device=i.device),
                           torch.zeros(pad_len, dtype=torch.bool, device=i.device)])
            )
        return torch.stack(padded), torch.stack(valid)

    context_idx, context_valid = pad(context_list)
    target_idx, target_valid   = pad(target_list)

    # safety clamp
    context_idx = context_idx.clamp(0, N - 1)
    target_idx  = target_idx.clamp(0, N - 1)

    return context_idx.long(), target_idx.long(), context_valid, target_valid


def mask_image(img, patch_mask, patch_size=16):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    Hp, Wp = patch_mask.shape

    for y in range(Hp):
        for x in range(Wp):
            if patch_mask[y, x]:
                x0, y0 = x * patch_size, y * patch_size
                draw.rectangle(
                    [x0, y0, x0 + patch_size, y0 + patch_size],
                    fill=(127,127,127)
                )
    return img