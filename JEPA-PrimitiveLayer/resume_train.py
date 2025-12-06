import os
import json 
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from comet_ml import Experiment
from tqdm import tqdm

# ============================================================
#  IMPORT MODELS + DATASET + LOSS
# ============================================================

from JEPA_PrimitiveLayer.bev_jepa import BEVJEPAEncoder2D
from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer
from Utils.dataset import MapDataset
from Utils.losses import compute_jepa_loss

from config import (
    LAMBDA_JEPA, LAMBDA_REG,
    ALPHA_0, ALPHA_1,
    BETA_1, BETA_2,
    GAMMA,
    USE_BF16,
    EPOCH,      # number of new epochs to train on Kaggle
    BATCH_SIZE,
    LR
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

# ============================================================
#  DETECT KAGGLE + CHECKPOINT PATHS
# ============================================================

IS_KAGGLE = os.path.exists("/kaggle/input")

CKPT_BEVDINO = "/kaggle/input/jepa-1-checkpoint-6-dec-25/pytorch/1default/1/bev_mobilenet_dino_init-240.pt"
CKPT_PRIMITIVE = "/kaggle/input/jepa-1-checkpoint-6-dec-25/pytorch/1default/1/primitive_layer-1499.pt"

PREV_FINISHED_EPOCH = 5

def comet_safe_save(model, epoch, tag="latest"):
    """
    Fail-safe saving: always try to log a checkpoint to Comet.
    Never crashes the main training process.
    """
    try:
        save_path = f"primitive_layer-epoch{epoch}-{tag}.pt"
        torch.save(
            {
                "version": 2,
                "epoch": epoch,
                "state": model.state_dict(),
            },
            save_path
        )
        experiment.log_model(
            name=f"primitive_layer_{tag}",
            file_or_folder=save_path
        )
        print(f"💾 Comet fail-safe: Checkpoint uploaded ({tag})")
    except Exception as e:
        print(f"❌ Comet fail-safe error: {e}")


# ============================================================
#  VERSION-SAFE WEIGHT LOADER
# ============================================================

def load_checkpoint_version_safe(model, ckpt_path, key=None, device="cpu"):
    print(f"\n🔍 Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and key is not None and key in ckpt:
        print(f"🔢 Checkpoint version: {ckpt.get('version', 'unknown')}")
        state = ckpt[key]
    else:
        state = ckpt

    model_state = model.state_dict()
    loaded, skipped = 0, 0

    for name, param in state.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_state, strict=False)

    print(f"   ✔️ Loaded {loaded} params")
    print(f"   ⚠️ Skipped {skipped} params (mismatch or new layer)")

    return loaded, skipped


# ============================================================
#  EXPERIMENT LOGGER (COMET)
# ============================================================

experiment = Experiment(
    api_key=os.getenv("COMET_API_KEY", "dummy"),
    project_name="jepa-tier-2",
    workspace="dtj-tran"
)
experiment.set_name("JEPA-Resume-Train")
experiment.log_parameters({
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "epochs": EPOCH,
    "lambda_jepa": LAMBDA_JEPA,
    "lambda_reg": LAMBDA_REG
})


# ============================================================
#  BUILD MODELS
# ============================================================

bev_backbone = BEVJEPAEncoder2D().to(device)

primitive_layer = PrimitiveLayer(
    bev_encoder=bev_backbone,
    dim=128,
    alpha0=ALPHA_0,
    alpha1=ALPHA_1,
    beta1=BETA_1,
    beta2=BETA_2,
    gamma=GAMMA
).to(device)


# ============================================================
#  LOAD CHECKPOINTS IF ON KAGGLE
# ============================================================

if IS_KAGGLE:

    if os.path.exists(CKPT_BEVDINO):
        load_checkpoint_version_safe(
            bev_backbone,
            CKPT_BEVDINO,
            key=None,
            device=device
        )
    else:
        print("⚠️ BEV checkpoint missing — training from scratch.")

    if os.path.exists(CKPT_PRIMITIVE):
        load_checkpoint_version_safe(
            primitive_layer,
            CKPT_PRIMITIVE,
            key=None,
            device=device
        )
        print(f"✅ Resuming PrimitiveLayer from epoch {PREV_FINISHED_EPOCH}")
    else:
        print("⚠️ PrimitiveLayer checkpoint missing — training from scratch.")


# ============================================================
#  OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    primitive_layer.parameters(),
    lr=LR,
    weight_decay=0.01
)


# ============================================================
#  DATASET + LOADER
# ============================================================

dataset = MapDataset(split="train")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# ============================================================
#  TRAINING LOOP (FULL, NO SKIP, CONTINUATION) + WITH FAIL-SAVE
# ============================================================

try:
    for local_epoch in range(EPOCH):

        # Calculate correct epoch number
        if IS_KAGGLE:
            global_epoch = PREV_FINISHED_EPOCH + local_epoch + 1
            print(f"\n🚀 Epoch {global_epoch} (continuation {local_epoch+1}/{EPOCH})")
        else:
            global_epoch = local_epoch + 1
            print(f"\n🚀 Epoch {global_epoch}/{EPOCH}")

        primitive_layer.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Training Epoch {global_epoch}")

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = primitive_layer(batch)
            loss_jepa, reg_loss = compute_jepa_loss(
                out,
                lambda_jepa=LAMBDA_JEPA,
                lambda_reg=LAMBDA_REG
            )

            loss = loss_jepa + reg_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(loader)
        print(f"✅ Epoch {global_epoch} complete. Avg Loss = {avg_loss:.6f}")

        experiment.log_metric("loss", avg_loss, step=global_epoch)

        # 🔥 FAIL-SAFE: push checkpoint every epoch
        comet_safe_save(primitive_layer, global_epoch, tag="epoch")

except Exception as e:
    print("\n❌ TRAINING FAILED with exception:", e)
    print("⚠️ Uploading emergency checkpoint to Comet...")

    # 🔥 Emergency checkpoint on failure
    comet_safe_save(primitive_layer, global_epoch, tag="fail")

    raise e  # Optional re-throw for debugging

finally:
    print("\n🏁 TRAINING FINISHED — running final fail-safe save")

    # 🔥 Final success checkpoint
    comet_safe_save(primitive_layer, global_epoch, tag="final")



# # ============================================================
# #  SAVE CHECKPOINT AT END
# # ============================================================

# SAVE_DIR = Path("./checkpoints")
# SAVE_DIR.mkdir(exist_ok=True)

# save_path = SAVE_DIR / f"primitive_layer-continue-{global_epoch}.pt"

# torch.save(
#     {
#         "version": 2,
#         "epoch": global_epoch,
#         "state": primitive_layer.state_dict(),
#     },
#     save_path
# )

# print(f"\n💾 Saved continuation checkpoint → {save_path}")