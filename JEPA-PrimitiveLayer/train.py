from dotenv import load_dotenv
from comet_ml import Experiment
from torch.utils.data import DataLoader
from Utils.dataset import MapDataset
from config import DEVICE, LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1, BETA_1, BETA_2, GAMMA, EPOCH, EMA_DECAY, ACCUM_STEPS, USE_BF16, EPOCH
import torch
import torch.nn as nn
from contextlib import nullcontext
from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer
from tqdm import tqdm
import torch.nn.functional as F
from Utils.losses import compute_jepa_loss
from Utils.ema_buffer import ema_update
import os
from ultralytics import YOLO

load_dotenv()

os.environ["COMET_LOG_PACKAGES"] = "0"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def transfer_yolo_weights(primitive_layer):
    print("🔄 Loading YOLOv8 backbone for weight transfer...")

    yolo = YOLO(os.getenv("YOLO_PATH"))

    # YOLOv8 v8.3+ full model is a giant Sequential stored in .model
    full = yolo.model.model  

    # Backbone is the first 10 layers (0–9) ALWAYS for YOLOv8
    backbone = full[:10]  # This is a Sequential

    # Extract the Conv layers we want:
    # layer 0 = Conv(3→32)
    # layer 1 = Conv(32→64)
    # layer 3 = Conv(64→128)
    yolo_s1 = backbone[0].conv
    yolo_s2 = backbone[1].conv
    yolo_s3 = backbone[3].conv

    def slice_and_load(my_conv, yo_conv, in_ch, out_ch):
        with torch.no_grad():
            my_conv.weight[:] = yo_conv.weight[:out_ch, :in_ch]
            if my_conv.bias is not None:
                my_conv.bias.zero_()

    # === CONTEXT encoder ===
    ctx = primitive_layer.context_encoder
    slice_and_load(ctx.s1[0].block[0], yolo_s1, 3, 16)
    slice_and_load(ctx.s2[0],          yolo_s2, 16, 32)
    slice_and_load(ctx.s3[0],          yolo_s3, 32, 64)

    # s4: perfect match → directly copy
    with torch.no_grad():
        ctx.s4[0].weight[:] = yolo_s3.weight
        if ctx.s4[0].bias is not None:
            ctx.s4[0].bias.zero_()

    # === TARGET encoder ===
    tgt = primitive_layer.target_encoder
    slice_and_load(tgt.s1[0].block[0], yolo_s1, 3, 16)
    slice_and_load(tgt.s2[0],          yolo_s2, 16, 32)
    slice_and_load(tgt.s3[0],          yolo_s3, 32, 64)
    with torch.no_grad():
        tgt.s4[0].weight[:] = yolo_s3.weight
        if tgt.s4[0].bias is not None:
            tgt.s4[0].bias.zero_()

    print("✅ YOLO → JEPA weight transfer complete.")

    # Freeze encoders
    for p in ctx.parameters():
        p.requires_grad = False
    for p in tgt.parameters():
        p.requires_grad = False

    print("🔒 Encoders frozen. Predictor will train.")

    return primitive_layer

def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

def main():
    # ----------------------------
    # MPS speed/memory settings
    # ----------------------------

    autocast_ctx = (
        torch.autocast(device_type="mps", dtype=torch.bfloat16)
        if USE_BF16 else nullcontext()
    )

    map_ds = MapDataset(map_csv_file=os.getenv("MAP_CSV"))
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    dataloader = DataLoader(
        map_ds,
        batch_size=16,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2 if 0 > 0 else None,
    )

    device = torch.device(DEVICE)
    primitive_layer = PrimitiveLayer(embed_dim=128).to(device)
    primitive_layer = transfer_yolo_weights(primitive_layer)
    optimizer = torch.optim.Adam(primitive_layer.predictor.parameters(), lr=3e-4, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)

    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE"),
    )

    experiment.set_name("JEPA-PrimitiveLayer-v1")
    experiment.add_tag("jepa")
    experiment.add_tag("primitive-layer")
    experiment.add_tag("masked-tokens")

    experiment.log_parameters({
        "lambda_jepa": LAMBDA_JEPA,
        "lambda_reg": LAMBDA_REG,
        "alpha0": ALPHA_0,
        "alpha1": ALPHA_1,
        "beta1": BETA_1,
        "beta2": BETA_2,
        "gamma": GAMMA,
        "lr": optimizer.param_groups[0]["lr"],
    })

    loss_history = {
        "total": [],
        "jepa": [],
        "empty": [],
        "nonempty": [],
        "reg": []
    }

    primitive_layer.train()

    global_step = 0
    num_epochs = int(EPOCH)
    global_step = 0
    for epoch_idx in range(EPOCH):
        t = epoch_idx / max(1, num_epochs - 1)
        primitive_layer.ema_decay = float(EMA_DECAY + (1.0 - EMA_DECAY) * t)

        for step_idx, batch in enumerate(tqdm(dataloader, desc="Training JEPA")):
            (
                bev, mask_emp, mask_non_emp, mask_union,
                mask_emp_np, mask_non_emp_np, mask_union_np,
                ph, pw, img
            ) = batch

            B = bev.shape[0]

            bev = bev.squeeze(1).to(device, non_blocking=True)

            # grids as bool
            mask_emp_grid = mask_emp_np.to(device, non_blocking=True).view(B,1,32,32).bool()
            mask_non_grid = mask_non_emp_np.to(device, non_blocking=True).view(B,1,32,32).bool()
            mask_any_grid = mask_union_np.to(device, non_blocking=True).view(B,1,32,32).bool()

            # fast upsample (32->64)
            mask_emp_up = up2(mask_emp_grid)
            mask_non_up = up2(mask_non_grid)
            mask_any_up = up2(mask_any_grid)

            mask_emp_flat = mask_emp_up.view(B, -1)
            mask_non_flat = mask_non_up.view(B, -1)

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

            # log less often
            if global_step % 20 == 0:
                lt = losses["loss_total"].item()
                experiment.log_metric("loss_total", lt, step=global_step)
                experiment.log_metric("loss_total", losses["loss_total"].item())
                experiment.log_metric("loss_jepa", losses["loss_jepa"].item())
                experiment.log_metric("loss_empty", losses["loss_P_empty"].item())
                experiment.log_metric("loss_nonempty", losses["loss_Q_nonempty"].item())
                experiment.log_metric("loss_reg", losses["loss_reg"].item())

                loss_history["total"].append(lt)

            global_step += 1
    # model saving (checkpoint)
    torch.save(primitive_layer.state_dict(), "primitive_layer.pt")
    experiment.log_asset("primitive_layer.pt")

    experiment.log_metric("final_loss_total", losses["loss_total"].item())
    experiment.end() # END EXPERIMENT

    # go to this for checking the process: https://www.comet.com/YOUR_WORKSPACE/jepa-training


if __name__ == "__main__":
    main()
