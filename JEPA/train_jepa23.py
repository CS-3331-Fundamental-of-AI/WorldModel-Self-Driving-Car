#!/usr/bin/env python3
"""
Stage-2: Train JEPA-2 + JEPA-3 with frozen JEPA-1 (V-JEPA-2).

- JEPA-1 provides s_c (no gradients)
- JEPA-2 trains with EMA target
- JEPA-3 trains inverse + global heads
"""

import os
import traceback
from pathlib import Path
from contextlib import nullcontext
from tqdm import tqdm
from dotenv import load_dotenv
from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader

def atomic_torch_save(obj, final_path: Path):
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(final_path)

# ============================================================
# CHECKPOINT HELPERS (Stage-2)
# ============================================================

def save_jepa2_checkpoint(t2, step: int, tag: str):
    path = CKPT_DIR / f"jepa2_{tag}_step{step}.pt"
    atomic_torch_save(
        {
            "version": 1,
            "step": step,
            "state": {
                "pa": t2.pa.state_dict(),
                "ia": t2.ia.state_dict(),
            },
        },
        path,
    )
    print(f"💾 Saved JEPA-2 checkpoint → {path}")


def save_jepa3_checkpoint(t3, step: int, tag: str):
    path = CKPT_DIR / f"jepa3_{tag}_step{step}.pt"
    atomic_torch_save(
        {
            "version": 1,
            "step": step,
            "state": t3.glob.state_dict(),   # ONLINE ONLY
        },
        path,
    )
    print(f"💾 Saved JEPA-3 checkpoint → {path}")

# -------------------------
# Config
# -------------------------
from config.config import (
    DEVICE, EPOCHS, BATCH_SIZE, NUM_WORKERS,
    CKPT_DIR, USE_BF16, ROOT
)

# --------------------------------------------------
# Device
# --------------------------------------------------
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

# ==================================================
# JEPA-1 checkpoint path
# ==================================================

JEPA1_CKPT = "/kaggle/input/jepa1-step7684/pytorch/default/1/final_step7684.pt"
assert os.path.exists(JEPA1_CKPT), f"JEPA-1 checkpoint not found: {JEPA1_CKPT}"
print(f"✅ Using JEPA-1 checkpoint: {JEPA1_CKPT}")

GCN_CKPT = "/kaggle/input/gcn-pretrained/pytorch/default/2/gcn_pretrain_step8215.pt"
assert os.path.exists(GCN_CKPT), f"GCN checkpoint not found: {GCN_CKPT}"
print(f"✅ Using pretrained GCN: {GCN_CKPT}")


# ==================================================
# Models
# ==================================================
from transformers import AutoModel
from JEPA_PrimitiveLayer.vjepa.model import PrimitiveLayerJEPA

from JEPA_SecondLayer import JEPA_Tier2_InverseAffordance, JEPA_Tier2_PhysicalAffordance
from JEPA_ThirdLayer import JEPA_Tier3_GlobalEncoding

# ==================================================
# Trainers & Pipeline
# ==================================================
from trainers.trainer_jepa1vjepa2 import JEPA1VJEPATrainer
from trainers.trainer_jepa2 import JEPA2Trainer
from trainers.trainer_jepa3 import JEPA3Trainer

from pipeline.jepa_pipeline import JEPAPipeline
from pipeline.jepa_adapter import JEPAInputAdapter

# ==================================================
# Dataset
# ==================================================
from Utils.unified_dataset import UnifiedDataset, unified_collate_fn
from Utils.jepa1vjepadata import MapDataset, build_map_dataframe
from Utils.jepa2data import Tier2Dataset, DATASET_PATH, get_scene_map
from Utils.jepa3data import Tier3Dataset
from Utils.utilities import to_float

# ==================================================
# Build all modules
# ==================================================
load_dotenv()
os.environ["COMET_LOG_PACKAGES"] = "0"

def build_all(device):
    # --------------------------------------------------
    # JEPA-1 (Frozen V-JEPA-2)
    # --------------------------------------------------
    backbone = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        dtype=torch.float16,
        device_map=device.type,
    )

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    jepa1 = PrimitiveLayerJEPA(
        encoder=backbone.encoder,
        grid_h=16,
        grid_w=16,
        enc_dim=1024,
        prim_dim=128,
    ).to(device)

    ckpt = torch.load(JEPA1_CKPT, map_location=device)
    jepa1.load_state_dict(ckpt["state"], strict=True)
    
    jepa1 = jepa1.to(device)
    for p in jepa1.parameters():
        p.requires_grad = False
    jepa1.eval()

    t1 = JEPA1VJEPATrainer(
        model=jepa1,
        optimizer=None,          # ❌ no optimizer
        device=device,
        trainable=False,         # IMPORTANT
    )

    # --------------------------------------------------
    # JEPA-2 (Student + EMA)
    # --------------------------------------------------
    jepa2_pa = JEPA_Tier2_PhysicalAffordance().to(device)
    jepa2_ia = JEPA_Tier2_InverseAffordance().to(device)
    jepa2_ia_ema = JEPA_Tier2_InverseAffordance().to(device)

    jepa2_ia_ema.load_state_dict(jepa2_ia.state_dict())
    for p in jepa2_ia_ema.parameters():
        p.requires_grad = False
    jepa2_ia_ema.eval()

    opt_j2 = torch.optim.AdamW(
        list(jepa2_pa.parameters()) + list(jepa2_ia.parameters()),
        lr=3e-4,
        weight_decay=1e-2,
    )

    t2 = JEPA2Trainer(
        pa_model=jepa2_pa,
        ia_model=jepa2_ia,
        ia_ema_model=jepa2_ia_ema,
        optimizer=opt_j2,
    )

    # --------------------------------------------------
    # JEPA-3
    # --------------------------------------------------
    jepa3_glob = JEPA_Tier3_GlobalEncoding(s_c_dim=128).to(device)

    jepa3_glob.load_pretrained_gcn(
        ckpt_path=GCN_CKPT,
        freeze_gcn=True,
    )

    jepa3_glob.init_ema()

    opt_j3 = torch.optim.AdamW(
        jepa3_glob.parameters(),
        lr=3e-4,
        weight_decay=1e-2,
    )

    t3 = JEPA3Trainer(
        glob=jepa3_glob,
        optimizer=opt_j3,
    )

    # -------------------------
    # Graph vocab for JEPA-2 (needed by adapter)
    # -------------------------
    TYPE_VALUES = ['agent', 'road']
    CATEGORY_VALUES = [
        'animal','human.pedestrian.adult','human.pedestrian.child',
        'human.pedestrian.construction_worker','human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer','human.pedestrian.stroller',
        'human.pedestrian.wheelchair','movable_object.barrier','movable_object.debris',
        'movable_object.pushable_pullable','movable_object.trafficcone',
        'static_object.bicycle_rack','vehicle.bicycle','vehicle.bus.bendy',
        'vehicle.bus.rigid','vehicle.car','vehicle.construction',
        'vehicle.emergency.ambulance','vehicle.emergency.police','vehicle.motorcycle',
        'vehicle.trailer','vehicle.truck'
    ]
    LAYER_VALUES = ['lane']

    type2id = {v: i for i, v in enumerate(TYPE_VALUES)}
    category2id = {v: i for i, v in enumerate(CATEGORY_VALUES)}
    layer2id = {v: i for i, v in enumerate(LAYER_VALUES)}

    # -------------------------
    # Adapter + Pipeline
    # -------------------------
    adapter = JEPAInputAdapter(
        device=device,
        type2id=type2id,
        category2id=category2id,
        layer2id=layer2id,
    )
    pipeline = JEPAPipeline(t1, t2, t3, adapter)

    return pipeline, {
        "jepa1": jepa1,
        "jepa2_pa": jepa2_pa,
        "jepa2_ia": jepa2_ia,
        "jepa3": jepa3_glob,
    }
    
def export_rssm_checkpoint(pipeline, path: Path):
    """
    Export JEPA-2 / JEPA-3 representation state for RSSM.
    EMA (teacher) weights only.
    """
    ckpt = {
        "version": 1,

        "jepa2": {
            "state": pipeline.t2.ia_ema_model.state_dict(),
        },

        "jepa3": {
            "state": pipeline.t3.glob.ema_state_dict(),
        },
    }

    atomic_torch_save(ckpt, path)
    print(f"💾 RSSM checkpoint saved to: {path}")

# ==================================================
# Training
# ==================================================
def train():
    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME", "JEPA"),
        workspace=os.getenv("WORK_SPACE"),
    )
    experiment.set_name("JEPA-Stage2-JEPA23")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    MAP_ROOT = os.getenv("MAP_ROOT")
    assert MAP_ROOT is not None, "MAP_ROOT must be set in .env"

    df_maps = build_map_dataframe(MAP_ROOT)

    unified_dataset = UnifiedDataset(
        jepa1_dataset=MapDataset(
            df=df_maps,
            mask_ratio=0.0   # no masking for context
        ),
        jepa2_dataset=Tier2Dataset(
            scene_map=get_scene_map(),
            dataset_path=DATASET_PATH,
            augment=True,
        ),
        jepa3_dataset=Tier3Dataset(
            scene_map=get_scene_map()
        ),
    )

    loader = DataLoader(
        unified_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=unified_collate_fn,
    )

    pipeline, models = build_all(DEVICE)

    autocast_ctx = (
        torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16)
        if USE_BF16
        else nullcontext()
    )

    global_step = 0

    try:
        for epoch in range(EPOCHS):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            epoch_loss = 0.0

            for batch in pbar:
                global_step += 1

                with autocast_ctx:
                    out = pipeline.step(batch)

                # -----------------------------
                # Scalar extraction
                # -----------------------------
                loss_total = to_float(out.get("loss", 0.0))
                epoch_loss += loss_total

                loss_j2     = to_float(out.get("loss_j2", 0.0))
                loss_j2_pa  = to_float(out.get("loss_j2_pa", 0.0))
                loss_j2_ia  = to_float(out.get("loss_j2_ia", 0.0))

                loss_j3     = to_float(out.get("loss_j3", 0.0))
                loss_j3_cos = to_float(out.get("loss_j3_cos", 0.0))
                loss_j3_vic = to_float(out.get("loss_j3_vic", 0.0))

                grad_j2_pa = to_float(out.get("grad_j2_pa", 0.0))
                grad_j2_ia = to_float(out.get("grad_j2_ia", 0.0))

                vic_j2_pa_var = to_float(out.get("vic_j2_pa_var", 0.0))
                vic_j2_ia_var = to_float(out.get("vic_j2_ia_var", 0.0))

                # -----------------------------
                # Logging
                # -----------------------------
                if global_step % 100 == 0:
                    experiment.log_metrics(
                        {"total": loss_total},
                        step=global_step,
                        prefix="loss"
                    )

                    experiment.log_metrics(
                        {"total": loss_j2, "pa": loss_j2_pa, "ia": loss_j2_ia},
                        step=global_step,
                        prefix="loss/jepa2"
                    )

                    experiment.log_metrics(
                        {"pa": grad_j2_pa, "ia": grad_j2_ia},
                        step=global_step,
                        prefix="grad/jepa2"
                    )

                    experiment.log_metrics(
                        {"pa_var": vic_j2_pa_var, "ia_var": vic_j2_ia_var},
                        step=global_step,
                        prefix="vicreg/jepa2"
                    )

                    experiment.log_metrics(
                        {
                            "total": loss_j3,
                            "cos_pred_tar": loss_j3_cos,
                            "vic_pred_tar": loss_j3_vic,
                        },
                        step=global_step,
                        prefix="loss/jepa3"
                    )

                pbar.set_postfix({
                    "L": f"{loss_total:.4f}",
                    "L2": f"{loss_j2:.4f}",
                    "L3": f"{loss_j3:.4f}",
                })

                if global_step % 500 == 0:
                    save_jepa2_checkpoint(pipeline.t2, global_step, tag="auto")
                    save_jepa3_checkpoint(pipeline.t3, global_step, tag="auto")

            avg_loss = epoch_loss / max(len(loader), 1)
            experiment.log_metric("epoch/avg_loss", avg_loss, step=epoch + 1)
            print(f"Epoch {epoch+1} done — avg loss {avg_loss:.6f}")

    except KeyboardInterrupt:
        print("\n⛔ KeyboardInterrupt detected")
        save_jepa2_checkpoint(pipeline.t2, global_step, tag="interrupt")
        save_jepa3_checkpoint(pipeline.t3, global_step, tag="interrupt")
        raise

    except Exception:
        print("\n❌ Training crashed")
        save_jepa2_checkpoint(pipeline.t2, global_step, tag="crash")
        save_jepa3_checkpoint(pipeline.t3, global_step, tag="crash")
        raise

    else:
        print("\n🎉 Training completed successfully")
        save_jepa2_checkpoint(pipeline.t2, global_step, tag="final")
        save_jepa3_checkpoint(pipeline.t3, global_step, tag="final")

    finally:
        experiment.end()
        print("🧹 Experiment closed")


# ==================================================
# Entrypoint
# ==================================================
if __name__ == "__main__":
    try:
        train()
    except Exception:
        print("\n❌ TRAINING FAILED\n")
        with open("jepa23_error.txt", "w") as f:
            f.write(traceback.format_exc())
        raise
