import os
import pandas as pd
import kagglehub
from dotenv import load_dotenv

load_dotenv()


# ----------------------------------------------------
# 1. DOWNLOAD DATASET (KaggleHub)
# ----------------------------------------------------
def download_dataset(ret_path=True, already_download=True):
    """
    Download dataset using KaggleHub.
    Uses .env variables whenever available.
    """

    if already_download:
        print("📦 Dataset already downloaded — skipping.")
        return os.getenv("DATASET_PATH", None) or 1

    print("⬇️  Downloading dataset via KaggleHub...")

    try:
        path = kagglehub.dataset_download("min1124/a-crude-data-set-converted-from-nuscene")
        print("📁 Dataset downloaded at:", path)
    except Exception as e:
        print("❌ KaggleHub Download Failed:", e)
        return None

    if ret_path:
        return path
    return 0


# ----------------------------------------------------
# 2. PREPARE CSV LIST OF MAP FILES
# ----------------------------------------------------
def prep_map_csv(
    dataset_path=None,
    output_csv="map_files.csv"
):
    """
    Scan a directory for BEV .png files and save them to a CSV.

    dataset_path:
        If None → load from MAP_ROOT in .env
        Else → use provided path
    """

    # Load path from .env if not given
    if dataset_path is None:
        dataset_path = os.getenv("MAP_ROOT", "./5/exported_maps/local_maps")

    # Ensure path normalized
    dataset_path = os.path.abspath(dataset_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset path does not exist: {dataset_path}")

    print(f"📌 Scanning dataset folder:\n   {dataset_path}")

    # List only PNG files
    png_files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(".png")])

    print(f"🖼 Found {len(png_files)} .png images.")
    print("🔍 First few:", png_files[:10])

    # Build dataframe
    df = pd.DataFrame({"filename": png_files})

    # Save to CSV (where your training scripts read it)
    df.to_csv(output_csv, index=False)
    print(f"📄 CSV saved: {output_csv}")

    # Quick verification
    df_loaded = pd.read_csv(output_csv)
    print("📄 CSV preview:")
    print(df_loaded.head())

    return df_loaded