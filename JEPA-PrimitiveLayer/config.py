import torch
EPOCH = 5
ACCUM_STEPS = 4  # effective_batch = batch_size * ACCUM_STEPS
USE_BF16 = torch.backends.mps.is_available()
EMBED_DIM = 128
PATCH_SIZE = 16
IMAGE_H = 32
IMAGE_W = 32
TOKEN_DIM = 3 * PATCH_SIZE * PATCH_SIZE
ACTION_DIM = 4
MASK_RATIO = 0.5
VICREG_WEIGHT = 0.1
DRIFT_WEIGHT = 0.05
JEPA_WEIGHT = 1.0
EMA_DECAY = 0.95 # Initially 0.996
BATCH_SIZE = 8
NUM_STEPS = 50
LR = 1e-3
LAMBDA_JEPA = 1.0
LAMBDA_REG = 10.0
ALPHA_0 = 0.25
ALPHA_1 = 0.75
BETA_1 = 1.0
BETA_2 = 1.0
GAMMA = 1.0
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DATA_ROOT = "/kaggle/input/test1t/exported_maps"
