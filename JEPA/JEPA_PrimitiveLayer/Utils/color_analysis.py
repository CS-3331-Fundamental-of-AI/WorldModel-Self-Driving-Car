from PIL import Image
import numpy as np

def analyze_bev_colors(image_path, top_k=100):
    """
    Analyze the unique RGB colors in a BEV image and return
    (color, count) sorted by frequency.

    Args:
        image_path (str): Path to the BEV image.
        top_k (int): Number of most frequent colors to show.

    Returns:
        List of (RGB, count) tuples sorted by descending count.
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)                       # (H,W,3)

    # Flatten pixel array to shape (N,3)
    pixels = arr.reshape(-1, 3)

    # Unique colors + counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # Sort by count (descending)
    sorted_idx = np.argsort(-counts)
    unique_colors_sorted = unique_colors[sorted_idx]
    counts_sorted = counts[sorted_idx]

    # Return top colors
    return [(unique_colors_sorted[i].tolist(), int(counts_sorted[i]))
            for i in range(min(top_k, len(unique_colors_sorted)))]