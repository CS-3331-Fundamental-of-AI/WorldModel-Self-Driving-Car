#!/usr/bin/env python3
# train_jepa.py
"""
Unified JEPA-1 / JEPA-2 / JEPA-3 trainer.

Assumptions (adapt dataset keys if needed; see TODOs below):
- PrimitiveLayer (JEPA-1) defined in JEPA-PrimitiveLayer/.../jepa_1.py
- Tier2Module (JEPA-2) defined in JEPA-SecondLayer/jepaTier2.py
- JEPA_Tier3_InverseAffordance, JEPA_Tier3_GlobalEncoding, losses in JEPA-ThirdLayer
- Utility losses: compute_jepa_loss (JEPA-1), total_tokenizer_loss_fsq (JEPA-2)
- ema_update helper exists (used for PrimitiveLayer target) and EMAHelper included below.

Run: python train_jepa.py
"""
import os
import sys
import copy
import math
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from comet_ml import Experiment
import traceback

load_dotenv()
os.environ["COMET_LOG_PACKAGES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext
from tqdm import tqdm
from config import (
    LAMBDA_JEPA, LAMBDA_REG, ALPHA_0, ALPHA_1,
    BETA_1, BETA_2, GAMMA, EPOCH,
    EMA_DECAY, ACCUM_STEPS, USE_BF16
)

# -------------------------
# CONFIG / HYPERPARAMETERS
# -------------------------
ROOT = Path(__file__).resolve().parent
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

BATCH_SIZE = 8 if IS_KAGGLE else 16
NUM_WORKERS = 2 if IS_KAGGLE else 4
EPOCHS = 10 if IS_KAGGLE else 200
LR = 3e-4
WEIGHT_DECAY = 1e-2
ACCUM_STEPS = 1
CLIP_NORM = 1.0

EMA_M_JEPA1 = 0.999   # primitive layer target momentum (you had EMA_DECAY)
EMA_M_JEPA2 = 0.995   # tier2 teacher shadow decay
USE_BF16 = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss weights
LAMBDA_JEPA1 = 1.0
LAMBDA_T2 = 1.0
LAMBDA_INV = 1.0
LAMBDA_GLOB = 1.0

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# JEPA-1
from JEPA_PrimitiveLayer import (
    PrimitiveLayer,
    load_dino_resnet50,
    compute_jepa_loss,
)

# JEPA-2
from JEPA_SecondLayer import (
    Tier2Module,
    total_tokenizer_loss_fsq,
)

# optional Tier-2 loss helper
try:
    from JEPA_SecondLayer import compute_tier2_loss
    HAVE_T2_LOSS = True
except ImportError:
    HAVE_T2_LOSS = False

# JEPA-3
from JEPA_ThirdLayer import (
    JEPA_Tier3_InverseAffordance,
    JEPA_Tier3_GlobalEncoding,
    inverse_affordance_losses,
    global_encoding_losses,
)

# Dataset
from Utils.dataset import MapDataset

# -------------------------
# EMA helper (shadow) for JEPA-2
# -------------------------
class EMAHelper:
    def __init__(self, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}

    def register(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model: nn.Module):
        # update shadow as EMA of model params
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                new = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                self.shadow[name] = new.clone()

    def assign_to(self, target: nn.Module):
        # push shadow values into target model
        for name, p in target.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

# primitive_layer target update helper (you already used this in your codebase)
def ema_update(online: nn.Module, target: nn.Module, m: float):
    with torch.no_grad():
        o_params = dict(online.named_parameters())
        t_params = dict(target.named_parameters())
        for k, t in t_params.items():
            if k in o_params:
                t.data.mul_(m).add_(o_params[k].data, alpha=1.0 - m)

# -------------------------
# small util helpers
# -------------------------
def up2(x: torch.Tensor) -> torch.Tensor:
    # expects [B,1,H,W] -> upsample by 2
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

def maybe_to_device(x, device):
    if x is None:
        return None
    return x.to(device)

# -------------------------
# Build models & optimizers
# -------------------------
def build_models(device: torch.device):
    jepa1 = PrimitiveLayer().to(device)
    jepa2 = Tier2Module().to(device)
    jepa3_inv = JEPA_Tier3_InverseAffordance().to(device)
    jepa3_glob = JEPA_Tier3_GlobalEncoding().to(device)

    # teacher copies (EMA) for stability (deep copies + freeze)
    #jepa1_tgt = copy.deepcopy(jepa1)
    # jepa1_tgt is only for checkpointing/frozen reference; JEPA-1 uses DINO teacher (z_t) for loss.

    jepa2_tgt = copy.deepcopy(jepa2)
    jepa3_inv_tgt = copy.deepcopy(jepa3_inv)
    jepa3_glob_tgt = copy.deepcopy(jepa3_glob)

    for m in (jepa2_tgt, jepa3_inv_tgt, jepa3_glob_tgt):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    return {
        "jepa1": jepa1,
        "jepa2": jepa2,
        "jepa3_inv": jepa3_inv,
        "jepa3_glob": jepa3_glob,
        "jepa2_tgt": jepa2_tgt,
        "jepa3_inv_tgt": jepa3_inv_tgt,
        "jepa3_glob_tgt": jepa3_glob_tgt,
    }

def build_optimizers(models: Dict[str, nn.Module]):
    # common pattern: train predictor of jepa1 (like your working script) but make whole-module optional
    opt_j1 = torch.optim.Adam(models["jepa1"].predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt_j2 = torch.optim.Adam(models["jepa2"].parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    params_j3 = list(models["jepa3_inv"].parameters()) + list(models["jepa3_glob"].parameters())
    opt_j3 = torch.optim.Adam(params_j3, lr=LR, weight_decay=WEIGHT_DECAY)
    return opt_j1, opt_j2, opt_j3

# -------------------------
# Per-layer step functions
# -------------------------
def step_jepa1(
    jepa1: nn.Module,
    opt: torch.optim.Optimizer,
    batch_masks,
    device: torch.device,
    autocast_ctx=nullcontext(),
    accum_steps: int = ACCUM_STEPS,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run JEPA-1 forward and optimizer step.
    - Uses z_t from DINO teacher as s_t
    - Mixed precision via autocast_ctx
    - Gradient accumulation and clipping
    - EMA update per step
    """

    mask_emp_np, mask_non_np, mask_union_np = batch_masks
    B = mask_emp_np.shape[0]

    mask_emp_up = up2(mask_emp_np.view(B, 1, 32, 32))
    mask_non_up = up2(mask_non_np.view(B, 1, 32, 32))
    mask_any_up = up2(mask_union_np.view(B, 1, 32, 32))
    mask_emp_flat = mask_emp_up.view(B, -1)
    mask_non_flat = mask_non_up.view(B, -1)

    with autocast_ctx:
        # JEPA-1 forward
        z_c, s_c, z_t = jepa1(
            mask_emp_np.squeeze(1),
            mask_non_np.squeeze(1),
            mask_emp_up,
            mask_non_up,
            mask_any_up,
        )
        z_c = F.normalize(z_c, dim=-1)
        s_c = F.normalize(s_c, dim=-1)
        z_t = F.normalize(z_t, dim=-1)

        # Compute JEPA-1 loss using z_t as teacher 
        jepa1_loss_dict = compute_jepa_loss(
            s_c=s_c,
            s_t=z_t,            
            z_c=z_c,
            mask_empty=mask_emp_flat,
            mask_nonempty=mask_non_flat,
            alpha0=ALPHA_0,
            alpha1=ALPHA_1,
            beta1=BETA_1,
            beta2=BETA_2,
            lambda_jepa=LAMBDA_JEPA,
            lambda_reg=LAMBDA_REG,
            gamma=GAMMA
        )

        loss = jepa1_loss_dict["loss_total"] / accum_steps
        loss.backward()

    # Gradient clipping and optimizer step
    torch.nn.utils.clip_grad_norm_(jepa1.parameters(), CLIP_NORM)
    opt.step()
    opt.zero_grad(set_to_none=True)

    # EMA update (teacher target)
    ema_update(jepa1.context_encoder, jepa1.target_encoder, jepa1.ema_decay)

    return loss.detach(), jepa1_loss_dict, s_c.detach(), z_c.detach(), z_t.detach()


def step_jepa2(
    jepa2: nn.Module,
    opt: torch.optim.Optimizer,
    batch_traj,
    batch_graph,
    ema_j2: EMAHelper,
    jepa2_tgt: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Run JEPA-2 (online) and its optimizer step.
    Returns loss (possibly zero if no tokenizer loss provided), loss-dict, and s_tg (fusion).
    """
    traj, traj_mask = batch_traj
    x_graph, graph_adj, graph_mask = batch_graph

    tier2_out = jepa2(traj, graph_adj, x_graph, traj_mask, graph_mask)
    s_tg = tier2_out.get("fusion", None)   # [B, D]
    # Try to compute tokenizer loss if module returns out_clean/out_aug
    out_clean = tier2_out.get("out_clean", None)
    out_aug = tier2_out.get("out_aug", None)

    if out_clean is not None and out_aug is not None:
        losses_j2 = total_tokenizer_loss_fsq(
            out_clean=out_clean,
            out_aug=out_aug,
            target_traj=traj,
            fsq_levels=16,  # adapt to your config
            lambda_recon=1.0, lambda_smooth=0.1,
        )
        loss = losses_j2["loss_total"]
    else:
        # If no tokenizer losses are available yet, set zero and user should implement recon inside Tier2
        loss = torch.tensor(0.0, device=traj.device)
        losses_j2 = {"loss_total": loss}

    opt.zero_grad(set_to_none=True)
    (loss / ACCUM_STEPS).backward()
    torch.nn.utils.clip_grad_norm_(jepa2.parameters(), CLIP_NORM)
    opt.step()

    # update EMA shadow and assign to frozen target model
    ema_j2.update(jepa2)
    ema_j2.assign_to(jepa2_tgt)

    return loss.detach(), losses_j2, s_tg.detach() if s_tg is not None else None

def step_jepa3(
    jepa3_inv: nn.Module,
    jepa3_glob: nn.Module,
    opt: torch.optim.Optimizer,
    action: torch.Tensor,
    spatial_x: torch.Tensor,
    s_c_for_glob: torch.Tensor,
    s_tg_target: Optional[torch.Tensor],
    global_nodes: Optional[torch.Tensor],
    global_adj: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run JEPA-3 inv + global and update their parameters.
    s_c_for_glob should be detached (we detach outside).
    s_tg_target: detached s_tg from JEPA-2 (used only in inverse-affordance loss; detached)
    """
    # inv: produces s_tg_hat etc.
    inv_out = jepa3_inv(action=action, spatial_x=spatial_x)
    inv_out["s_a_detached"] = inv_out.get("s_a_detached", inv_out.get("s_a", None))

    # global uses inv_out["s_tg_hat"] and s_c_for_glob (we pass detached s_c_for_glob)
    glob_out = jepa3_glob(
        s_tg_hat=inv_out["s_tg_hat"],
        s_c=s_c_for_glob,
        global_nodes=global_nodes,
        global_adj=global_adj,
        tokens_final=inv_out.get("tokens_final", None),
    )

    # normalize if desired (loss functions in your code call normalize in cosine_distance)
    # compute losses: inverse_affordance_losses expects inv_out and target_s_tg (detached)
    losses_inv = inverse_affordance_losses(inv_out, target_s_tg=s_tg_target)
    losses_glob = global_encoding_losses(glob_out, s_tar_target=None)

    loss = losses_inv["total"] + losses_glob["total"]

    opt.zero_grad(set_to_none=True)
    (loss / ACCUM_STEPS).backward()
    torch.nn.utils.clip_grad_norm_(jepa3_inv.parameters(), CLIP_NORM)
    torch.nn.utils.clip_grad_norm_(jepa3_glob.parameters(), CLIP_NORM)
    opt.step()

    return loss.detach(), {"inv": losses_inv, "glob": losses_glob}, inv_out, glob_out

# -------------------------
# Training loop (main)
# -------------------------
def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    device: torch.device = DEVICE,
):
    # -------------------------
    # Comet Init
    # -------------------------
    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE"),
    )

    experiment.set_name("JEPA-FULL-3L-Kaggle")

    experiment.log_parameters({
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "lambda_jepa1": LAMBDA_JEPA1,
        "lambda_jepa2": LAMBDA_T2,
        "lambda_inv": LAMBDA_INV,
        "lambda_glob": LAMBDA_GLOB,
        "ema_jepa1": EMA_M_JEPA1,
        "ema_jepa2": EMA_M_JEPA2,
    })

    print("Device:", device)
    ds = MapDataset(map_csv_file=os.environ.get("MAP_CSV", "maps.csv"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type=="cuda"))

    # build models + teachers
    models = build_models(device)
    jepa1 = models["jepa1"]
    jepa2 = models["jepa2"]
    jepa3_inv = models["jepa3_inv"]
    jepa3_glob = models["jepa3_glob"]
    # jepa1_tgt = models["jepa1_tgt"]
    jepa2_tgt = models["jepa2_tgt"]
    jepa3_inv_tgt = models["jepa3_inv_tgt"]
    jepa3_glob_tgt = models["jepa3_glob_tgt"]

    # optimizers (separate)
    opt_j1, opt_j2, opt_j3 = build_optimizers(models)

    # EMA helper for jepa2 (shadow)
    ema_j2 = EMAHelper(decay=EMA_M_JEPA2)
    ema_j2.register(jepa2)

    # optionally register jepa3 EMA shadow if you want to keep shadow copies (not required)
    ema_j3_inv = EMAHelper(decay=EMA_M_JEPA3)
    ema_j3_inv.register(jepa3_inv)
    ema_j3_glob = EMAHelper(decay=EMA_M_JEPA3)
    ema_j3_glob.register(jepa3_glob)

    # training mode
    jepa1.train(); jepa2.train(); jepa3_inv.train(); jepa3_glob.train()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and USE_BF16))
    autocast_ctx = (torch.autocast(device_type=device.type, dtype=torch.bfloat16) if USE_BF16 else nullcontext())

    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0

        for batch in pbar:
            # -------------------------
            # Unpack dataset (adapt keys if dataset different)
            # -------------------------
            # Expected keys from MapDataset (adapt if needed)
            masked_img = maybe_to_device(batch.get("masked_img", None), device)
            unmasked_img = maybe_to_device(batch.get("unmasked_img", None), device)
            mask_emp_np = maybe_to_device(batch.get("mask_emp_np", batch.get("mask_emp", None)), device)
            mask_non_np = maybe_to_device(batch.get("mask_non_emp_np", batch.get("mask_non", None)), device)
            mask_union_np = maybe_to_device(batch.get("mask_union_np", batch.get("mask_union", None)), device)

            traj = maybe_to_device(batch.get("deltas", batch.get("traj", None)), device)
            traj_mask = maybe_to_device(batch.get("mask", batch.get("traj_mask", None)), device)
            x_graph = maybe_to_device(batch.get("graph_feats", None), device)
            graph_adj = maybe_to_device(batch.get("graph_adj", None), device)
            graph_mask = maybe_to_device(batch.get("graph_mask", None), device)

            action = maybe_to_device(batch.get("action", None), device)
            spatial_x = maybe_to_device(batch.get("spatial_x", None), device)
            global_nodes = maybe_to_device(batch.get("global_nodes", None), device)
            global_adj = maybe_to_device(batch.get("global_adj", None), device)

            # ensure batch masks present
            if mask_emp_np is None or mask_non_np is None or mask_union_np is None:
                raise RuntimeError("Dataset must provide mask_emp_np / mask_non_emp_np / mask_union_np for JEPA-1")

            # -------------------------
            # Forward & Steps (Phase A: parallel training; we run per-layer steps sequentially but avoid cross-grad by detach)
            # -------------------------
            # 1) JEPA-1 forward+step: we optionally pass inv_s_tg_hat (None on first iter)
            inv_s_tg_hat_for_j1 = None  # we will compute jepa3 inv later in this iteration; for simplicity we use last iteration's value = None
            # (To use current iteration inv prediction in jepa1 loss you'd need to compute jepa3.inv first;
            #  that would require ordering. The previous working scripts used jepa3 outputs in jepa1 loss —
            #  if you want that, move jepa3 forward above and pass prediction here. Current design keeps separation.)
            loss_j1, losses_j1_dict, s_c, z_c, z_t = step_jepa1(
                jepa1=jepa1,
                opt=opt_j1,
                batch_masks=(mask_emp_np, mask_non_np, mask_union_np),
                inv_s_tg_hat=inv_s_tg_hat_for_j1,
                device=device,
            )

            # 2) JEPA-2 forward+step
            loss_j2, losses_j2_dict, s_tg = step_jepa2(
                jepa2=jepa2,
                opt=opt_j2,
                batch_traj=(traj, traj_mask),
                batch_graph=(x_graph, graph_adj, graph_mask),
                ema_j2=ema_j2,
                jepa2_tgt=jepa2_tgt,
                device=device,
            )

            # 3) JEPA-3 forward+step (use detached s_c for spatial context; use s_tg.detach() as target for inverse affordance)
            s_c_for_j3 = s_c.detach() if s_c is not None else None
            s_tg_target = s_tg.detach() if s_tg is not None else None

            loss_j3, losses_j3_dict, inv_out, glob_out = step_jepa3(
                jepa3_inv=jepa3_inv,
                jepa3_glob=jepa3_glob,
                opt=opt_j3,
                action=action,
                spatial_x=spatial_x,
                s_c_for_glob=s_c_for_j3,
                s_tg_target=s_tg_target,
                global_nodes=global_nodes,
                global_adj=global_adj,
                device=device,
            )

            # 4) Optionally update JEPA-3 EMA shadow (keeps teacher copies stable)
            ema_j3_inv.update(jepa3_inv)
            ema_j3_glob.update(jepa3_glob)
            ema_j3_inv.assign_to(jepa3_inv_tgt)
            ema_j3_glob.assign_to(jepa3_glob_tgt)

            # 5) Jepa1 could also benefit from using current inv prediction as s_t in its loss.
            #    If you want that (as in older script), you can recompute jepa1 loss here using inv_out["s_tg_hat"]
            #    and take an extra optimization step for jepa1. For clarity we left that out; uncomment & adapt if desired.

            # -------------------------
            # Logging / checkpointing / bookkeeping
            # -------------------------
            loss_total_val = float((loss_j1 + loss_j2 + loss_j3).detach().cpu().item())
            epoch_loss += loss_total_val
            global_step += 1
            
            if global_step % 10 == 0:
                experiment.log_metric("loss/total", loss_total_val, step=global_step)
                experiment.log_metric("loss/jepa1", float(loss_j1), step=global_step)
                experiment.log_metric("loss/jepa2", float(loss_j2), step=global_step)
                experiment.log_metric("loss/jepa3", float(loss_j3), step=global_step)


            pbar.set_postfix({
                "L_total": f"{loss_total_val:.4f}",
                "L1": f"{float(loss_j1):.4f}",
                "L2": f"{float(loss_j2):.4f}",
                "L3": f"{float(loss_j3):.4f}",
            })

            # periodic checkpoint
            if global_step % 500 == 0:
                ckpt = {
                    "step": global_step,
                    "epoch": epoch,
                    "jepa1_state": jepa1.state_dict(),
                    "jepa2_state": jepa2.state_dict(),
                    "jepa2_tgt_shadow": ema_j2.shadow,
                    "jepa3_inv_state": jepa3_inv.state_dict(),
                    "jepa3_glob_state": jepa3_glob.state_dict(),
                }
                torch.save(ckpt, CHECKPOINT_DIR / f"jepa_world_ckpt_step{global_step}.pt")

        # end epoch
        avg_loss = epoch_loss / (len(loader) + 1e-12)
        print(f"Epoch {epoch+1} done — avg loss {avg_loss:.6f}")
        experiment.log_metric("epoch/avg_loss", avg_loss, step=epoch + 1)

        # save epoch checkpoint
        torch.save({
            "epoch": epoch + 1,
            "jepa1_state": jepa1.state_dict(),
            "jepa2_state": jepa2.state_dict(),
            "jepa2_tgt_shadow": ema_j2.shadow,
            "jepa3_inv_state": jepa3_inv.state_dict(),
            "jepa3_glob_state": jepa3_glob.state_dict(),
        }, CHECKPOINT_DIR / f"jepa_world_epoch{epoch+1}.pt")

    print("Training finished.")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print("\n❌ TRAINING FAILED — fail-safe mode\n")

        fail_path = "jepa_full_fail.pt"
        try:
            torch.save({
                "jepa1": jepa1.state_dict(),
                "jepa2": jepa2.state_dict(),
                "jepa3_inv": jepa3_inv.state_dict(),
                "jepa3_glob": jepa3_glob.state_dict(),
            }, fail_path)
            experiment.log_asset(fail_path)
        except:
            print("⚠️ Could not save fail-safe checkpoint")

        with open("training_error_log.txt", "w") as f:
            f.write(traceback.format_exc())

        experiment.log_asset("training_error_log.txt")
        experiment.end()
        raise
