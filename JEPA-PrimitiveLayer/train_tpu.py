

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext
from dotenv import load_dotenv
from comet_ml import Experiment
from ultralytics import YOLO
from tqdm import tqdm

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from Utils.dataset import MapDataset
from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer
from Utils.losses import compute_jepa_loss
from Utils.ema_buffer import ema_update
from config import (
    LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1,
    BETA_1, BETA_2, GAMMA, EPOCH, EMA_DECAY,
    ACCUM_STEPS, USE_BF16
)

load_dotenv()
os.environ["COMET_LOG_PACKAGES"] = "0"


def transfer_yolo_weights(primitive_layer):
    print("🔄 Loading YOLOv8 backbone for weight transfer...")
    yolo_path = os.getenv("YOLO_PATH", "yolov8s.pt")
    yolo = YOLO(yolo_path)

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


def _train_fn(index):
    # TPU device
    device = xm.xla_device()

    # Dataset / loader (CPU side)
    map_csv = os.getenv("MAP_CSV", "./map_files.csv")
    map_ds = MapDataset(map_csv_file=map_csv)

    # On TPU, keep workers modest; XLA parallel loader handles device feeding
    dataloader = DataLoader(
        map_ds,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # Wrap with XLA mp device loader
    para_loader = pl.MpDeviceLoader(dataloader, device)

    # Model
    primitive_layer = PrimitiveLayer(embed_dim=128).to(device)
    primitive_layer = transfer_yolo_weights(primitive_layer)

    optimizer = torch.optim.Adam(
        primitive_layer.predictor.parameters(),
        lr=3e-4,
        weight_decay=0.01
    )
    optimizer.zero_grad(set_to_none=True)

    # Mixed precision on TPU: BF16 is best if available
    autocast_ctx = (
        torch.autocast(device_type="xla", dtype=torch.bfloat16)
        if USE_BF16 else nullcontext()
    )

    # Comet: only log from master process to avoid duplicates
    is_master = xm.is_master_ordinal()
    experiment = None
    if is_master:
        experiment = Experiment(
            api_key=os.getenv("API_KEY"),
            project_name=os.getenv("PROJECT_NAME"),
            workspace=os.getenv("WORK_SPACE"),
        )
        experiment.set_name("JEPA-PrimitiveLayer-TPU")
        experiment.add_tag("tpu")
        experiment.add_tag("jepa")
        experiment.add_tag("primitive-layer")
        experiment.log_parameters({
            "lambda_jepa": LAMBDA_JEPA,
            "lambda_reg": LAMBDA_REG,
            "alpha0": ALPHA_0,
            "alpha1": ALPHA_1,
            "beta1": BETA_1,
            "beta2": BETA_2,
            "gamma": GAMMA,
            "lr": optimizer.param_groups[0]["lr"],
            "accum_steps": ACCUM_STEPS,
        })

    primitive_layer.train()
    global_step = 0
    num_epochs = int(EPOCH)

    for epoch_idx in range(num_epochs):
        # EMA warm-up
        t = epoch_idx / max(1, num_epochs - 1)
        primitive_layer.ema_decay = float(EMA_DECAY + (1.0 - EMA_DECAY) * t)

        prog = tqdm(para_loader, desc=f"TPU Training JEPA (epoch {epoch_idx+1}/{num_epochs})") if is_master else para_loader

        for step_idx, batch in enumerate(prog):
            (
                bev, mask_emp, mask_non_emp, mask_union,
                mask_emp_np, mask_non_emp_np, mask_union_np,
                ph, pw, img
            ) = batch

            B = bev.shape[0]
            bev = bev.squeeze(1)  # already on device via MpDeviceLoader

            mask_emp_grid = mask_emp_np.view(B, 1, 32, 32).bool()
            mask_non_grid = mask_non_emp_np.view(B, 1, 32, 32).bool()
            mask_any_grid = mask_union_np.view(B, 1, 32, 32).bool()

            mask_emp_up = up2(mask_emp_grid)
            mask_non_up = up2(mask_non_grid)
            mask_any_up = up2(mask_any_grid)

            mask_emp_flat = mask_emp_up.view(B, -1)
            mask_non_flat = mask_non_up.view(B, -1)

            with autocast_ctx:
                z_c, s_c, z_t = primitive_layer(
                    bev,
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
                # XLA optimizer step
                xm.clip_grad_norm_(primitive_layer.parameters(), 1.0)
                xm.optimizer_step(optimizer, barrier=True)
                optimizer.zero_grad(set_to_none=True)

                ema_update(
                    primitive_layer.context_encoder,
                    primitive_layer.target_encoder,
                    primitive_layer.ema_decay
                )

            if is_master and (global_step % 20 == 0):
                experiment.log_metric("loss_total", losses["loss_total"].item(), step=global_step)
                experiment.log_metric("loss_jepa", losses["loss_jepa"].item(), step=global_step)
                experiment.log_metric("loss_empty", losses["loss_P_empty"].item(), step=global_step)
                experiment.log_metric("loss_nonempty", losses["loss_Q_nonempty"].item(), step=global_step)
                experiment.log_metric("loss_reg", losses["loss_reg"].item(), step=global_step)

            global_step += 1

        # mark end of epoch for TPU sync
        xm.mark_step()

    # Save only on master
    if is_master:
        torch.save(primitive_layer.state_dict(), "primitive_layer_tpu.pt")
        experiment.log_asset("primitive_layer_tpu.pt")
        experiment.log_metric("final_loss_total", losses["loss_total"].item())
        experiment.end()


def main():
    # Spawn one process per TPU core
    xmp.spawn(_train_fn, nprocs=8, start_method="fork")


if __name__ == "__main__":
    main()