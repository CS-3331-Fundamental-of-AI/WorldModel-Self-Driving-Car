# ------------------------------------------------------------
# train_adjustment.py (Kaggle-optimized version)
# ------------------------------------------------------------
import os
from dotenv import load_dotenv
from comet_ml import Experiment
from torch.utils.data import DataLoader
from Utils.dataset import MapDataset
from config import (
    LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1,
    BETA_1, BETA_2, GAMMA, EPOCH,
    EMA_DECAY, ACCUM_STEPS, USE_BF16
)
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from JEPA.JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer, load_dino_resnet50
from Utils.losses import compute_jepa_loss
from Utils.ema_buffer import ema_update
from tqdm import tqdm
import traceback

# ------------------------------------------------------------
# Kaggle detection
# ------------------------------------------------------------
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
NUM_WORKERS = 2 # if IS_KAGGLE else  0
BATCH_SIZE = 8 if IS_KAGGLE else 16
MAX_STEPS = 300 if IS_KAGGLE else 999999

load_dotenv()
os.environ["COMET_LOG_PACKAGES"] = "0"


# ----------------------------
# Upsample Helper
# ----------------------------
def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)


# ----------------------------
# Training Loop
# ----------------------------
def main():

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

    # Load DINO teacher (Kaggle-safe path)
    teacher = load_dino_resnet50(device)

    # Mixed precision context
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if USE_BF16 else nullcontext()
    )

    # ----------------------------
    # Comet Init
    # ----------------------------
    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE"),
    )
    experiment.set_name("JEPA-PrimitiveLayer-v1-Kaggle")

    try:
        # ----------------------------
        # Dataset + DataLoader
        # ----------------------------
        map_ds = MapDataset(map_csv_file=os.getenv("MAP_CSV"))
        dataloader = DataLoader(
            map_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            persistent_workers=False,
            pin_memory=True if device.type == "cuda" else False
        )

        # ----------------------------
        # Model
        # ----------------------------
        primitive_layer = PrimitiveLayer(
            embed_dim=128,
            distilled_path="bev_mobilenet_dino_init.pt"
        ).to(device)

        optimizer = torch.optim.Adam(
            primitive_layer.predictor.parameters(),
            lr=3e-4,
            weight_decay=0.01
        )

        global_step = 0
        num_epochs = int(EPOCH)

        # ----------------------------
        # Epoch Loop
        # ----------------------------
        for epoch_idx in range(num_epochs):

            # EMA schedule
            t = epoch_idx / max(1, num_epochs - 1)
            primitive_layer.ema_decay = float(EMA_DECAY + (1.0 - EMA_DECAY) * t)

            print(f"\n🚀 Epoch {epoch_idx + 1}/{num_epochs}")
            pbar = tqdm(dataloader, mininterval=1.0)

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

        # ----------------------------
        # Save final model
        # ----------------------------
        torch.save(primitive_layer.state_dict(), "primitive_layer.pt")
        experiment.log_asset("primitive_layer.pt")
        experiment.end()

    # ----------------------------
    # FAIL-SAFE Handling
    # ----------------------------
    except Exception as e:
        print("\n❌ TRAINING FAILED — fail-safe mode\n")

        fail_path = "primitive_layer_fail.pt"
        try:
            torch.save(primitive_layer.state_dict(), fail_path)
            print(f"💾 Saved fail-safe model → {fail_path}")
        except:
            print("⚠️ Could not save fail-safe model")

        with open("training_error_log.txt", "w") as f:
            f.write(traceback.format_exc())

        experiment.log_asset("training_error_log.txt")
        experiment.end()

        raise


if __name__ == "__main__":
    main()