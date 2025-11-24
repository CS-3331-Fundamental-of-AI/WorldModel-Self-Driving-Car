import os
import torch
import torch.nn as nn
from comet_ml import Experiment
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Utils.dataset import MapDataset
from config import (DEVICE, LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1,
                    BETA_1, BETA_2, GAMMA, EPOCH, EMA_DECAY,
                    ACCUM_STEPS, USE_BF16)
from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer
from Utils.losses import compute_jepa_loss
from Utils.ema_buffer import ema_update
from ultralytics import YOLO

from dotenv import load_dotenv
load_dotenv()

os.environ["COMET_LOG_PACKAGES"] = "0"

def transfer_yolo_weights(primitive_layer):
    print("🔄 Loading YOLOv8 backbone for weight transfer...")

    yolo = YOLO(os.getenv("YOLO_PATH"))
    full = yolo.model.model
    backbone = full[:10]

    yolo_s1 = backbone[0].conv
    yolo_s2 = backbone[1].conv
    yolo_s3 = backbone[3].conv

    def slice_and_load(my_conv, yo_conv, in_ch, out_ch):
        with torch.no_grad():
            my_conv.weight[:] = yo_conv.weight[:out_ch, :in_ch]
            if my_conv.bias is not None:
                my_conv.bias.zero_()

    ctx = primitive_layer.context_encoder
    slice_and_load(ctx.s1[0].block[0], yolo_s1, 3, 16)
    slice_and_load(ctx.s2[0],          yolo_s2, 16, 32)
    slice_and_load(ctx.s3[0],          yolo_s3, 32, 64)
    with torch.no_grad():
        ctx.s4[0].weight[:] = yolo_s3.weight
        if ctx.s4[0].bias is not None:
            ctx.s4[0].bias.zero_()

    tgt = primitive_layer.target_encoder
    slice_and_load(tgt.s1[0].block[0], yolo_s1, 3, 16)
    slice_and_load(tgt.s2[0],          yolo_s2, 16, 32)
    slice_and_load(tgt.s3[0],          yolo_s3, 32, 64)
    with torch.no_grad():
        tgt.s4[0].weight[:] = yolo_s3.weight
        if tgt.s4[0].bias is not None:
            tgt.s4[0].bias.zero_()

    print("✅ YOLO → JEPA weight transfer complete.")

    for p in ctx.parameters():
        p.requires_grad = False
    for p in tgt.parameters():
        p.requires_grad = False

    print("🔒 Encoders frozen. Predictor will train.")

    return primitive_layer

def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.backends.cudnn.benchmark = True

    # Mixed-precision setup
    use_fp16 = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    map_ds = MapDataset(map_csv_file=os.getenv("MAP_CSV"))
    dataloader = DataLoader(
        map_ds,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    primitive_layer = PrimitiveLayer(embed_dim=128).to(device)
    primitive_layer = transfer_yolo_weights(primitive_layer)

    optimizer = torch.optim.Adam(
        primitive_layer.predictor.parameters(),
        lr=3e-4,
        weight_decay=0.01
    )
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

    loss_history = {"total": [], "jepa": [], "empty": [], "nonempty": [], "reg": []}
    primitive_layer.train()

    global_step = 0
    for epoch_idx in range(EPOCH):
        t = epoch_idx / max(1, EPOCH - 1)
        primitive_layer.ema_decay = float(EMA_DECAY + (1.0 - EMA_DECAY) * t)

        for step_idx, batch in enumerate(dataloader):
            bev, mask_emp, mask_non_emp, mask_union, \
                mask_emp_np, mask_non_emp_np, mask_union_np, \
                ph, pw, img = batch

            B = bev.shape[0]
            bev = bev.squeeze(1).to(device, non_blocking=True)

            mask_emp_grid     = mask_emp_np.to(device, non_blocking=True).view(B,1,32,32).bool()
            mask_non_grid     = mask_non_emp_np.to(device, non_blocking=True).view(B,1,32,32).bool()
            mask_any_grid     = mask_union_np.to(device, non_blocking=True).view(B,1,32,32).bool()

            mask_emp_up = up2(mask_emp_grid)
            mask_non_up = up2(mask_non_grid)
            mask_any_up = up2(mask_any_grid)

            mask_emp_flat = mask_emp_up.view(B, -1)
            mask_non_flat = mask_non_up.view(B, -1)

            # Move masks to device dtype if needed
            mask_emp_up = mask_emp_up.to(device)
            mask_non_up = mask_non_up.to(device)
            mask_any_up = mask_any_up.to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16, device_type="cuda"):
                z_c, s_c, z_t = primitive_layer(
                    bev,
                    mask_emp.squeeze(1).to(device),
                    mask_non_emp.squeeze(1).to(device),
                    mask_emp_up,
                    mask_non_up,
                    mask_any_up
                )

                z_c_norm = F.normalize(z_c, dim=-1)
                s_c_norm = F.normalize(s_c, dim=-1)
                z_t_norm = F.normalize(z_t, dim=-1)

                losses = compute_jepa_loss(
                    s_c=s_c_norm, s_t=z_t_norm, z_c=z_c_norm,
                    mask_empty=mask_emp_flat.to(device),
                    mask_nonempty=mask_non_flat.to(device),
                    alpha0=ALPHA_0, alpha1=ALPHA_1,
                    beta1=BETA_1, beta2=BETA_2,
                    lambda_jepa=LAMBDA_JEPA, lambda_reg=LAMBDA_REG,
                    gamma=GAMMA,
                )

            loss = losses["loss_total"] / ACCUM_STEPS
            scaler.scale(loss).backward()

            if (step_idx + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(primitive_layer.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                ema_update(
                    primitive_layer.context_encoder,
                    primitive_layer.target_encoder,
                    primitive_layer.ema_decay
                )

            if global_step % 20 == 0:
                lt = losses["loss_total"].item()
                experiment.log_metric("loss_total", lt, step=global_step)
                experiment.log_metric("loss_jepa", losses["loss_jepa"].item(), step=global_step)
                experiment.log_metric("loss_empty", losses["loss_P_empty"].item(), step=global_step)
                experiment.log_metric("loss_nonempty", losses["loss_Q_nonempty"].item(), step=global_step)
                experiment.log_metric("loss_reg", losses["loss_reg"].item(), step=global_step)
                loss_history["total"].append(lt)

            global_step += 1

    torch.save(primitive_layer.state_dict(), "primitive_layer_cuda.pt")
    experiment.log_asset("primitive_layer_cuda.pt")
    experiment.log_metric("final_loss_total", losses["loss_total"].item())
    experiment.end()

if __name__ == "__main__":
    main()