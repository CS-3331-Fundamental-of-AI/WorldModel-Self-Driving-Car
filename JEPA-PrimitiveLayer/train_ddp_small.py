import os, math, random, time, traceback
from dotenv import load_dotenv
from pathlib import Path

# ✅ Robust env load (Kaggle/Colab-safe)
load_dotenv(".env")
os.environ["COMET_LOG_PACKAGES"] = "0"
from comet_ml import Experiment
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from Utils.dataset import MapDataset
from JEPA_PrimitiveLayer.jepa_1 import PrimitiveLayer
from Utils.losses import compute_jepa_loss
from Utils.ema_buffer import ema_update
from ultralytics import YOLO
from config import (
    LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1,
    BETA_1, BETA_2, GAMMA, EMA_DECAY,
    ACCUM_STEPS, USE_BF16
)

# -----------------------
# Optional small dataset
# -----------------------
EPOCH = 2
SMALL_N = 8000  # int(os.getenv("SMALL_N", "0"))  # 0 = use full dataset

# -----------------------
# Checkpoint settings
# -----------------------
CKPT_DIR = Path("checkpoints_ddp")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_EVERY_STEPS = 200   # save "latest" every N optimizer steps
KEEP_LAST_K = 3          # keep last K latest checkpoints


def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def save_checkpoint(tag, primitive_layer_ddp, optimizer, scaler, epoch_idx, step_idx, global_step, losses=None):
    """
    DDP-safe checkpoint save. Only rank 0 writes.

    tag: "latest", "error", "final", "best"...
    """
    if not is_rank0():
        return None

    model = primitive_layer_ddp.module if hasattr(primitive_layer_ddp, "module") else primitive_layer_ddp
    ckpt_path = CKPT_DIR / f"primitive_layer_{tag}.pt"

    payload = {
        "tag": tag,
        "epoch": epoch_idx,
        "step_idx": step_idx,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "ema_decay": float(getattr(model, "ema_decay", EMA_DECAY)),
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        },
        "losses": {k: float(v.item()) for k, v in losses.items()} if losses is not None else None,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(payload, ckpt_path)

    # keep rolling backups for "latest"
    if tag == "latest":
        ts_path = CKPT_DIR / f"primitive_layer_latest_step{global_step}.pt"
        torch.save(payload, ts_path)

        # cleanup old latest backups
        latest_files = sorted(CKPT_DIR.glob("primitive_layer_latest_step*.pt"), key=os.path.getmtime)
        if len(latest_files) > KEEP_LAST_K:
            for f in latest_files[:-KEEP_LAST_K]:
                try:
                    f.unlink()
                except:
                    pass

    print(f"💾 Saved checkpoint: {ckpt_path}")
    return str(ckpt_path)


# -----------------------
# YOLO weight transfer
# -----------------------
def transfer_yolo_weights(primitive_layer):
    if dist.get_rank() == 0:
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

    # Train policy:
    for p in ctx.parameters(): p.requires_grad = True
    for p in tgt.parameters(): p.requires_grad = False
    for p in primitive_layer.predictor.parameters(): p.requires_grad = True

    if dist.get_rank() == 0:
        print("✅ YOLO → JEPA weight transfer complete.")
        print("🔒 Target frozen. Context + Predictor train.")
    return primitive_layer


def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)


def get_amp_policy():
    use_amp = True
    amp_dtype = (
        torch.bfloat16
        if (USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )
    return use_amp, amp_dtype, scaler


# -----------------------
# DDP worker
# -----------------------
def ddp_worker(rank, world_size):
    # ---- init process group ----
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.backends.cudnn.benchmark = True

    # AMP
    use_amp, amp_dtype, scaler = get_amp_policy()

    # Dataset
    map_csv = os.getenv("MAP_CSV")
    full_ds = MapDataset(map_csv_file=map_csv)

    if SMALL_N > 0:
        idx = list(range(min(SMALL_N, len(full_ds))))
        full_ds = Subset(full_ds, idx)
        if rank == 0:
            print(f"📌 Using SMALL dataset: {len(full_ds)} samples")

    sampler = DistributedSampler(
        full_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    dataloader = DataLoader(
        full_ds,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    # Model
    primitive_layer = PrimitiveLayer(embed_dim=128).to(device)
    primitive_layer = transfer_yolo_weights(primitive_layer)

    primitive_layer = torch.nn.parallel.DistributedDataParallel(
        primitive_layer,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    optimizer = torch.optim.Adam(
        list(primitive_layer.module.context_encoder.parameters()) +
        list(primitive_layer.module.predictor.parameters()),
        lr=3e-4,
        weight_decay=0.01
    )
    optimizer.zero_grad(set_to_none=True)

    # Comet only on rank 0
    experiment = None
    if rank == 0:
        experiment = Experiment(
            api_key=os.getenv("API_KEY"),
            project_name=os.getenv("PROJECT_NAME"),
            workspace=os.getenv("WORK_SPACE"),
        )
        experiment.set_name("JEPA-PrimitiveLayer-DDP-Small")
        experiment.add_tags(["jepa", "primitive-layer", "ddp", "small-ds"])
        experiment.log_parameters({
            "lambda_jepa": LAMBDA_JEPA,
            "lambda_reg": LAMBDA_REG,
            "alpha0": ALPHA_0,
            "alpha1": ALPHA_1,
            "beta1": BETA_1,
            "beta2": BETA_2,
            "gamma": GAMMA,
            "lr": optimizer.param_groups[0]["lr"],
            "amp_dtype": str(amp_dtype),
            "small_n": SMALL_N,
            "world_size": world_size,
        })

    primitive_layer.train()
    global_step = 0
    last_losses = None  # keep last losses for emergency save

    # -----------------------
    # TRAIN WITH FAIL-SAFE CKPT
    # -----------------------
    try:
        for epoch_idx in range(int(EPOCH)):
            sampler.set_epoch(epoch_idx)
            t = epoch_idx / max(1, int(EPOCH) - 1)
            primitive_layer.module.ema_decay = float(EMA_DECAY + (1.0 - EMA_DECAY) * t)

            pbar = dataloader
            if rank == 0:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{EPOCH}")

            for step_idx, batch in enumerate(pbar):
                bev, mask_emp, mask_non_emp, mask_union, \
                mask_emp_np, mask_non_emp_np, mask_union_np, \
                ph, pw, img = batch

                B = bev.shape[0]
                masked_img   = mask_emp.squeeze(1).to(device, non_blocking=True)
                unmasked_img = mask_non_emp.squeeze(1).to(device, non_blocking=True)

                mask_emp_grid = mask_emp_np.to(device, non_blocking=True).view(B,1,32,32).bool()
                mask_non_grid = mask_non_emp_np.to(device, non_blocking=True).view(B,1,32,32).bool()
                mask_any_grid = mask_union_np.to(device, non_blocking=True).view(B,1,32,32).bool()

                mask_emp_up = up2(mask_emp_grid)
                mask_non_up = up2(mask_non_grid)
                mask_any_up = up2(mask_any_grid)

                mask_emp_flat = mask_emp_up.view(B, -1)
                mask_non_flat = mask_non_up.view(B, -1)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    z_c, s_c, z_t = primitive_layer(
                        masked_img, unmasked_img,
                        mask_emp_up, mask_non_up, mask_any_up
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
                        lambda_jepa=LAMBDA_JEPA, lambda_reg=LAMBDA_REG,
                        gamma=GAMMA,
                    )

                last_losses = losses
                loss = losses["loss_total"] / ACCUM_STEPS

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step_idx + 1) % ACCUM_STEPS == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        list(primitive_layer.module.context_encoder.parameters()) +
                        list(primitive_layer.module.predictor.parameters()),
                        1.0
                    )

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    ema_update(
                        primitive_layer.module.context_encoder,
                        primitive_layer.module.target_encoder,
                        primitive_layer.module.ema_decay
                    )

                    # periodic latest checkpoint (rank 0 only)
                    if global_step % SAVE_EVERY_STEPS == 0:
                        save_checkpoint(
                            "latest", primitive_layer, optimizer, scaler,
                            epoch_idx, step_idx, global_step, losses=last_losses
                        )

                if experiment is not None and global_step % 20 == 0:
                    experiment.log_metric("loss_total", losses["loss_total"].item(), step=global_step)
                    experiment.log_metric("loss_jepa", losses["loss_jepa"].item(), step=global_step)
                    experiment.log_metric("loss_empty", losses["loss_P_empty"].item(), step=global_step)
                    experiment.log_metric("loss_nonempty", losses["loss_Q_nonempty"].item(), step=global_step)
                    experiment.log_metric("loss_reg", losses["loss_reg"].item(), step=global_step)

                global_step += 1

    except KeyboardInterrupt:
        if rank == 0:
            print("\n🛑 KeyboardInterrupt — saving emergency checkpoint...")
            save_checkpoint(
                "error", primitive_layer, optimizer, scaler,
                epoch_idx, step_idx, global_step, losses=last_losses
            )
        raise

    except Exception as e:
        if rank == 0:
            print("\n❌ Training crashed — saving emergency checkpoint...")
            print("Error:", repr(e))
            traceback.print_exc()
            save_checkpoint(
                "error", primitive_layer, optimizer, scaler,
                epoch_idx, step_idx, global_step, losses=last_losses
            )
        raise

    finally:
        # always save some final state on rank 0
        if rank == 0:
            print("\n💾 Saving final checkpoint...")
            save_checkpoint(
                "final", primitive_layer, optimizer, scaler,
                epoch_idx if "epoch_idx" in locals() else 0,
                step_idx if "step_idx" in locals() else 0,
                global_step,
                losses=last_losses
            )

    # save state + comet asset on rank 0
    if rank == 0:
        state_path = "primitive_layer_ddp.pt"
        torch.save(primitive_layer.module.state_dict(), state_path)
        experiment.log_asset(state_path)
        if last_losses is not None:
            experiment.log_metric("final_loss_total", last_losses["loss_total"].item())
        experiment.end()

    dist.destroy_process_group()


# -----------------------
# Launcher
# -----------------------
def main():
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Need >=2 GPUs, found {world_size}"

    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")

    mp.spawn(ddp_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()