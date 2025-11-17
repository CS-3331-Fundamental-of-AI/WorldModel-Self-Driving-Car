import cv2
import numpy as np

# === Load the image ===
img = cv2.imread("image.png")
H, W = img.shape[:2]

# Convert to RGBA for transparency
img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# === Grid setup ===
rows, cols = 4, 6
cell_h, cell_w = H // rows, W // cols

# === Define context masking pattern ===
# 1 = visible region (keep)
# 0 = empty region (mask)
context_pattern = np.array([
    [0, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1],
])

# Target = complement of context
target_pattern = 1 - context_pattern

# === Masking function ===
def apply_mask(img_rgba, pattern):
    masked = img_rgba.copy()
    alpha = np.ones((H, W), dtype=np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            if pattern[i, j] == 0:
                # transparent (empty)
                alpha[y0:y1, x0:x1] = 40
    masked[..., 3] = alpha
    return masked

# === Generate and save complementary masks ===
context_masked = apply_mask(img_rgba, context_pattern)
target_masked = apply_mask(img_rgba, target_pattern)

cv2.imwrite("context_mask_complement.png", context_masked)
cv2.imwrite("target_mask_complement.png", target_masked)

print("✅ Saved complementary context & target masks.")