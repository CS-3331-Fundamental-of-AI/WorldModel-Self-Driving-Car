import os
from dotenv import load_dotenv
from comet_ml import Experiment
from torch.utils.data import DataLoader
from Utils.dataset import MapDataset
from config import LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1, BETA_1, BETA_2, GAMMA, EPOCH, EMA_DECAY, ACCUM_STEPS, USE_BF16
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer, load_dino_resnet50
from Utils.losses import compute_jepa_loss
from Utils.ema_buffer import ema_update
from torchvision.models import resnet50
from tqdm import tqdm
import traceback

# ----------------------------
# Environment Setup
# ----------------------------
load_dotenv()
os.environ["COMET_LOG_PACKAGES"] = "0"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


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
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon MPS backend")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU detected — using CPU (very slow!)")

    print(f"👉 Final device used for training: {device}")

    teacher = load_dino_resnet50()

    # Mixed precision context
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if USE_BF16 else nullcontext()
    )

    # Comet Experiment (initialized early for fail-safe)
    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE"),
    )
    experiment.set_name("JEPA-PrimitiveLayer-v1")

    try:
        # ----------------------------
        # Dataset
        # ----------------------------
        map_ds = MapDataset(map_csv_file=os.getenv("MAP_CSV"))
        dataloader = DataLoader(map_ds, batch_size=16, num_workers=2)

        # ----------------------------
        # Model
        # ----------------------------
        primitive_layer = PrimitiveLayer(embed_dim=128,distilled_path="bev_mobilenet_dino_init.pt",).to(device) 
        # from pretrain_distill.py


        optimizer = torch.optim.Adam(
            primitive_layer.predictor.parameters(),
            lr=3e-4,
            weight_decay=0.01,
        )

        global_step = 0
        num_epochs = int(EPOCH)

        # ----------------------------
        # Epoch Loop
        # ----------------------------
        for epoch_idx in range(num_epochs):

            t = epoch_idx / max(1, num_epochs - 1)
            primitive_layer.ema_decay = float(EMA_DECAY + (1.0 - EMA_DECAY) * t)

            pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{num_epochs}")

            # ----------------------------
            # Batch Loop
            # ----------------------------
            for step_idx, batch in enumerate(pbar):
                (
                    bev, mask_emp, mask_non_emp, mask_union,
                    mask_emp_np, mask_non_emp_np, mask_union_np,
                    ph, pw, img
                ) = batch

                B = bev.shape[0]
                bev = bev.squeeze(1).to(device)

                mask_emp_grid = mask_emp_np.to(device).view(B,1,32,32).bool()
                mask_non_grid = mask_non_emp_np.to(device).view(B,1,32,32).bool()
                mask_any_grid = mask_union_np.to(device).view(B,1,32,32).bool()

                mask_emp_up = up2(mask_emp_grid)
                mask_non_up = up2(mask_non_grid)
                mask_any_up = up2(mask_any_grid)

                mask_emp_flat = mask_emp_up.view(B, -1)
                mask_non_flat = mask_non_up.view(B, -1)

                # JEPA forward pass
                with autocast_ctx:
                    z_c, s_c, z_t = primitive_layer(
                        mask_emp.squeeze(1),
                        mask_non_emp.squeeze(1),
                        mask_emp_up,
                        mask_non_up,
                        mask_any_up
                    )

                    z_c_norm = F.normalize(z_c, dim=-1)
                    s_c_norm = F.normalize(s_c, dim=-1)
                    z_t_norm = F.normalize(z_t, dim=-1)

                    losses = compute_jepa_loss(
                        s_c=s_c_norm, s_t=z_t_norm, z_c=z_c_norm,
                        mask_empty=mask_emp_flat,
                        mask_nonempty=mask_non_flat,
                        alpha0=ALPHA_0, alpha1=ALPHA_1,
                        beta1=BETA_1, beta2=BETA_2,
                        lambda_jepa=LAMBDA_JEPA,
                        lambda_reg=LAMBDA_REG,
                        gamma=GAMMA,
                    )

                # Backprop
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

                # TQDM live logging
                pbar.set_postfix({
                    "loss": f"{losses['loss_total'].item():.4f}",
                    "jepa": f"{losses['loss_jepa'].item():.4f}"
                })

                # Comet logging (frequent)
                if global_step % 5 == 0:
                    experiment.log_metric("loss_total_step", losses["loss_total"].item(), step=global_step)
                    experiment.log_metric("loss_jepa_step", losses["loss_jepa"].item(), step=global_step)
                    experiment.log_metric("loss_empty_step", losses["loss_P_empty"].item(), step=global_step)
                    experiment.log_metric("loss_nonempty_step", losses["loss_Q_nonempty"].item(), step=global_step)
                    experiment.log_metric("loss_reg_step", losses["loss_reg"].item(), step=global_step)

                    tqdm.write(
                        f"[step {global_step}] "
                        f"loss={losses['loss_total'].item():.4f} | "
                        f"jepa={losses['loss_jepa'].item():.4f}"
                    )

                global_step += 1

        # Normal save
        torch.save(primitive_layer.state_dict(), "primitive_layer.pt")
        experiment.log_asset("primitive_layer.pt")
        experiment.end()

    # ----------------------------
    # FAIL-SAFE: Handle Crashes
    # ----------------------------
    except Exception as e:
        print("\n❌ TRAINING FAILED — entering fail-safe mode...\n")

        # Save emergency model checkpoint
        fail_path = "primitive_layer_fail.pt"
        try:
            torch.save(primitive_layer.state_dict(), fail_path)
            print(f"💾 Saved fail-safe checkpoint → {fail_path}")
            experiment.log_asset(fail_path)
        except:
            print("⚠️ Could not save fail-safe model.")

        # Save traceback
        err_msg = traceback.format_exc()
        with open("training_error_log.txt", "w") as f:
            f.write(err_msg)

        print("📝 Error log written to training_error_log.txt")
        experiment.log_asset("training_error_log.txt")

        print("⛔ Training ended due to error:")
        print(err_msg)

        experiment.end()


if __name__ == "__main__":
    main()