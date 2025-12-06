import os
import json 
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from comet_ml import Experiment
from tqdm import tqdm
from contextlib import nullcontext
# ============================================================
#  IMPORT MODELS + DATASET + LOSS
# ============================================================

from JEPA_PrimitiveLayer.bev_jepa import BEVJEPAEncoder2D
from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer
from Utils.dataset import MapDataset
from Utils.losses import compute_jepa_loss
from Utils.ema_buffer import ema_update

from config import (
    LAMBDA_JEPA, LAMBDA_REG,
    ALPHA_0, ALPHA_1,
    BETA_1, BETA_2,
    GAMMA,
    USE_BF16,
    EPOCH,      # number of new epochs to train on Kaggle
    BATCH_SIZE,
    LR,
    EMA_DECAY,
    ACCUM_STEPS
)
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
NUM_WORKERS = 2 # if IS_KAGGLE else  0
MAX_STEPS = 300 if IS_KAGGLE else 999999

# ----------------------------
# Auto Device Selection
# ----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple Silicon MPS backend")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU detected — using CPU")

print(f"👉 Final device used for training: {device}")


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
        torch.save(model.state_dict(), "primitive_layer.pt")
        experiment.log_asset("primitive_layer.pt")
        experiment.end()
        print("Log asset and end experiment !")

# ----------------------------
# Upsample Helper
# ----------------------------
def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)



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
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE"),
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

primitive_layer = PrimitiveLayer(embed_dim=128,distilled_path=CKPT_BEVDINO).to(device)


# ============================================================
#  LOAD CHECKPOINTS IF ON KAGGLE
# ============================================================

if IS_KAGGLE:

    # if os.path.exists(CKPT_BEVDINO):
    #     load_checkpoint_version_safe(
    #         bev_backbone,
    #         CKPT_BEVDINO,
    #         key=None,
    #         device=device
    #     )
    # else:
    #     print("⚠️ BEV checkpoint missing — training from scratch.")

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

map_ds = MapDataset(map_csv_file=os.getenv("MAP_CSV"))
loader = DataLoader(
            map_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            persistent_workers=False,
            pin_memory=True if device.type == "cuda" else False
        )

# Mixed precision context
autocast_ctx = (
    torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if USE_BF16 else nullcontext()
)


# ============================================================
#  TRAINING LOOP (FULL, NO SKIP, CONTINUATION) + WITH FAIL-SAVE
# ============================================================

try:
    primitive_layer.train()
    global_step = 0
    for local_epoch in range(EPOCH):

        # Calculate correct epoch number
        if IS_KAGGLE:
            global_epoch = PREV_FINISHED_EPOCH + local_epoch + 1
            print(f"\n🚀 Epoch {global_epoch} (continuation {local_epoch+1}/{EPOCH})")
        else:
            global_epoch = local_epoch + 1
            print(f"\n🚀 Epoch {global_epoch}/{EPOCH}")

        # EMA schedule
        t = local_epoch / max(1, EPOCH - 1)
        primitive_layer.ema_decay = float(EMA_DECAY + (1.0 - EMA_DECAY) * t)

        print(f"\n🚀 Epoch {local_epoch + 1}/{EPOCH}")
        pbar = tqdm(loader, mininterval=1.0)

        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Training Epoch {global_epoch}")

       
        # ----------------------------
        # Batch Loop
        # ----------------------------
        for step_idx, batch in enumerate(pbar):

                if IS_KAGGLE and step_idx >= MAX_STEPS:
                    print(f"⏹ Early stop: step {step_idx}/{MAX_STEPS}")
                    break

                (
                    bev, mask_emp, mask_non_emp, mask_union,
                    mask_emp_np, mask_non_emp_np, mask_union_np,
                    ph, pw, img
                ) = batch

                B = bev.shape[0]
                bev = bev.squeeze(1).to(device, non_blocking=True)

                # mask handling
                mask_emp_grid = mask_emp_np.to(device).view(B,1,32,32).bool()
                mask_non_grid = mask_non_emp_np.to(device).view(B,1,32,32).bool()
                mask_any_grid = mask_union_np.to(device).view(B,1,32,32).bool()

                mask_emp_up = up2(mask_emp_grid)
                mask_non_up = up2(mask_non_grid)
                mask_any_up = up2(mask_any_grid)

                mask_emp_flat = mask_emp_up.view(B, -1)
                mask_non_flat = mask_non_up.view(B, -1)

                # JEPA forward
                with autocast_ctx:
                    z_c, s_c, z_t = primitive_layer(
                        mask_emp.squeeze(1),
                        mask_non_emp.squeeze(1),
                        mask_emp_up,
                        mask_non_up,
                        mask_any_up
                    )

                    z_c = F.normalize(z_c, dim=-1)
                    s_c = F.normalize(s_c, dim=-1)
                    z_t = F.normalize(z_t, dim=-1)

                    losses = compute_jepa_loss(
                        s_c=s_c, s_t=z_t, z_c=z_c,
                        mask_empty=mask_emp_flat,
                        mask_nonempty=mask_non_flat,
                        alpha0=ALPHA_0, alpha1=ALPHA_1,
                        beta1=BETA_1, beta2=BETA_2,
                        lambda_jepa=LAMBDA_JEPA,
                        lambda_reg=LAMBDA_REG,
                        gamma=GAMMA
                    )

                loss = losses["loss_total"] / ACCUM_STEPS
                loss.backward()

                if (step_idx + 1) % ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(primitive_layer.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    ema_update(
                        primitive_layer.context_encoder,
                        primitive_layer.target_encoder,
                        primitive_layer.ema_decay
                    )

                pbar.set_postfix({
                    "loss": f"{losses['loss_total'].item():.4f}",
                    "jepa": f"{losses['loss_jepa'].item():.4f}"
                })

                if global_step % 10 == 0:
                    experiment.log_metric("loss", losses["loss_total"].item(), step=global_step)
                    experiment.log_metric("jepa_loss", losses["loss_jepa"].item(), step=global_step)

                global_step += 1

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