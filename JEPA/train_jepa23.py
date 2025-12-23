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

# -------------------------
# Config
# -------------------------
from config.config import (
    DEVICE, EPOCHS, BATCH_SIZE, NUM_WORKERS,
    CKPT_DIR, USE_BF16, ROOT
)

# ==================================================
# Paths
# ==================================================

JEPA1_DIR = Path("/kaggle/output/checkpoints/jepa1_vjepa")

if "JEPA1_CKPT" in os.environ:
    JEPA1_CKPT = os.environ["JEPA1_CKPT"]
else:
    ckpts = sorted(
        JEPA1_DIR.glob("jepa1_step*.pt"),
        key=lambda p: int(p.stem.split("step")[-1])
    )
    assert len(ckpts) > 0, f"No JEPA-1 checkpoints found in {JEPA1_DIR}"
    JEPA1_CKPT = str(ckpts[-1])   # latest checkpoint

print(f"✅ Using JEPA-1 checkpoint: {JEPA1_CKPT}")



# ==================================================
# Models
# ==================================================
from transformers import AutoModel
from JEPA_PrimitiveLayer.vjepa.model import PrimitiveLayerJEPA

from JEPA_SecondLayer import Tier2Module
from JEPA_ThirdLayer import (
    JEPA_Tier3_InverseAffordance,
    JEPA_Tier3_GlobalEncoding,
)

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
        torch_dtype=torch.float16,
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

    ckpt = torch.load(JEPA1_CKPT, map_location="cpu")
    jepa1.load_state_dict(ckpt["model"], strict=True)
    
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
    jepa2 = Tier2Module().to(device)
    jepa2_tgt = Tier2Module().to(device)

    jepa2_tgt.load_state_dict(jepa2.state_dict())
    for p in jepa2_tgt.parameters():
        p.requires_grad = False
    jepa2_tgt.eval()

    opt_j2 = torch.optim.AdamW(
        jepa2.parameters(),
        lr=3e-4,
        weight_decay=1e-2,
    )

    t2 = JEPA2Trainer(jepa2, jepa2_tgt, opt_j2)

    # --------------------------------------------------
    # JEPA-3
    # --------------------------------------------------
    jepa3_inv = JEPA_Tier3_InverseAffordance().to(device)
    jepa3_glob = JEPA_Tier3_GlobalEncoding(s_c_dim=128).to(device)

    opt_j3 = torch.optim.AdamW(
        list(jepa3_inv.parameters()) +
        list(jepa3_glob.parameters()),
        lr=3e-4,
        weight_decay=1e-2,
    )

    t3 = JEPA3Trainer(
        jepa3_inv,      # inverse affordance
        jepa3_glob,     # global encoding 
        opt_j3          # optimizer
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
        "jepa2": jepa2,
        "jepa3_inv": jepa3_inv,
        "jepa3_glob": jepa3_glob,
    }

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

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0

        for batch in pbar:
            global_step += 1

            with autocast_ctx:
                out = pipeline.step(batch)

            loss = float(out["loss"])
            epoch_loss += loss

            if global_step % 100 == 0:
                experiment.log_metrics({
                    # =====================
                    # Global
                    # =====================
                    "loss/total": loss,

                    # =====================
                    # JEPA-2 (VICReg, SSL)
                    # =====================
                    "loss/jepa2/total": out["loss_j2"],
                    "loss/jepa2/vic_inv": out["loss_j2_inv"],
                    "loss/jepa2/vic_var": out["loss_j2_var"],
                    "loss/jepa2/vic_cov": out["loss_j2_cov"],

                    # =====================
                    # JEPA-3 (Task)
                    # =====================
                    "loss/jepa3/total": out["loss_j3"],
                    "loss/jepa3/inv_total": out["loss_j3_inv"],
                    "loss/jepa3/glob_total": out["loss_j3_glob"],
                }, step=global_step)


            pbar.set_postfix({"loss": f"{loss:.4f}"})

            if global_step % 500 == 0:
                torch.save(
                    {
                        "step": global_step,
                        **{k: v.state_dict() for k, v in models.items()},
                    },
                    CKPT_DIR / f"jepa23_step{global_step}.pt",
                )

        avg_loss = epoch_loss / max(len(loader), 1)
        experiment.log_metric("epoch/avg_loss", avg_loss, step=epoch + 1)
        print(f"Epoch {epoch+1} done — avg loss {avg_loss:.6f}")

    experiment.end()
    print("✅ Stage-2 training finished")

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
