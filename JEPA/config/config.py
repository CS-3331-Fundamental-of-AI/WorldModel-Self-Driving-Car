# config/config.py
from pathlib import Path
import torch
import os

# -------------------------
# Environment
# -------------------------
ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# -------------------------
# Training
# -------------------------
EPOCHS = 5 if IS_KAGGLE else 200
BATCH_SIZE = 8 if IS_KAGGLE else 16
NUM_WORKERS = 2 if IS_KAGGLE else 4

LR = 3e-4
WEIGHT_DECAY = 1e-2
CLIP_NORM = 1.0
ACCUM_STEPS = 1

USE_BF16 = False

# -------------------------
# EMA
# -------------------------
EMA_JEPA1 = 0.996
EMA_JEPA2 = 0.995
EMA_JEPA3 = 0.995

# -------------------------
# Loss weights
# -------------------------
LAMBDA_JEPA1 = 1.0
LAMBDA_JEPA2 = 1.0
LAMBDA_INV = 0.25
LAMBDA_GLOB = 1.0

# JEPA-1 loss params & config
ALPHA_0 = 1.0
ALPHA_1 = 1.0
BETA_1 = 1.0
BETA_2 = 1.0
GAMMA = 1.0
LAMBDA_JEPA = 1.0
LAMBDA_REG = 0.05

MASK_RATIO = 0.5
PATCH_SIZE = 16

