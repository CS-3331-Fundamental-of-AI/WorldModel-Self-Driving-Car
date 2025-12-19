#!/usr/bin/env python3
"""
Unified JEPA-1 / JEPA-2 / JEPA-3 training (single steady phase).

- Trainers handle backward + EMA
- Pipeline handles stop-gradient
- Train loop handles AMP, logging, checkpointing
"""

import os
import traceback
from dotenv import load_dotenv
from comet_ml import Experiment
from tqdm import tqdm
from contextlib import nullcontext
from Utils.logging import log_metrics

import torch
from torch.utils.data import DataLoader

# -------------------------
# Config
# -------------------------
from config.config import (
    DEVICE, EPOCHS, BATCH_SIZE, NUM_WORKERS,
    CKPT_DIR, USE_BF16
)

# -------------------------
# Models
# -------------------------
from JEPA_PrimitiveLayer import PrimitiveLayer
from JEPA_SecondLayer import Tier2Module
from JEPA_ThirdLayer import (
    JEPA_Tier3_InverseAffordance,
    JEPA_Tier3_GlobalEncoding,
)

# -------------------------
# Trainers & Pipeline
# -------------------------
from trainers.trainer_jepa1 import JEPA1Trainer
from trainers.trainer_jepa2 import JEPA2Trainer
from trainers.trainer_jepa3 import JEPA3Trainer
from pipeline.jepa_pipeline import JEPAPipeline

# -------------------------
# Dataset
# -------------------------
from Utils.jepa1data import MapDataset
from Utils.jepa2data import Tier2Dataset, tier2_collate_fn, DATASET_PATH, get_scene_map
from Utils.unified_dataset import UnifiedDataset, unified_collate_fn
from Utils.utilities import move_j1_to_device, build_graph_batch

# -------------------------
# Environment
# -------------------------
load_dotenv()
os.environ["COMET_LOG_PACKAGES"] = "0"


# ============================================================
# Build everything
# ============================================================
def build_all(device):
    # ---- models ----
    jepa1 = PrimitiveLayer().to(device)
    jepa2 = Tier2Module().to(device)
    jepa3_inv = JEPA_Tier3_InverseAffordance().to(device)
    jepa3_glob = JEPA_Tier3_GlobalEncoding().to(device)

    # ---- EMA teachers ----
    jepa2_tgt = Tier2Module().to(device)
    jepa3_inv_tgt = JEPA_Tier3_InverseAffordance().to(device)
    jepa3_glob_tgt = JEPA_Tier3_GlobalEncoding().to(device)

    for tgt, src in [
        (jepa2_tgt, jepa2),
        (jepa3_inv_tgt, jepa3_inv),
        (jepa3_glob_tgt, jepa3_glob),
    ]:
        tgt.load_state_dict(src.state_dict())
        for p in tgt.parameters():
            p.requires_grad = False
        tgt.eval()

    # ---- optimizers ----
    opt_j1 = torch.optim.Adam(
        jepa1.predictor.parameters(), lr=3e-4, weight_decay=1e-2
    )
    opt_j2 = torch.optim.Adam(
        jepa2.parameters(), lr=3e-4, weight_decay=1e-2
    )
    opt_j3 = torch.optim.Adam(
        list(jepa3_inv.parameters()) + list(jepa3_glob.parameters()),
        lr=3e-4, weight_decay=1e-2
    )

    # ---- trainers ----
    t1 = JEPA1Trainer(jepa1, opt_j1)
    t2 = JEPA2Trainer(jepa2, jepa2_tgt, opt_j2)
    t3 = JEPA3Trainer(jepa3_inv, jepa3_glob, opt_j3)

    return JEPAPipeline(t1, t2, t3), {
        "jepa1": jepa1,
        "jepa2": jepa2,
        "jepa3_inv": jepa3_inv,
        "jepa3_glob": jepa3_glob,
    }


# ============================================================
# Training
# ============================================================
def train():
    experiment = Experiment(
        api_key=os.getenv("API_KEY"),
        project_name=os.getenv("PROJECT_NAME"),
        workspace=os.getenv("WORK_SPACE"),
    )
    experiment.set_name("JEPA-FULL-3L-STEADY")

    # --------------------------
    # JEPA-1 dataset
    # --------------------------
    dataset_j1 = MapDataset(map_csv_file=os.getenv("MAP_CSV", "maps.csv"))
    
    # --------------------------
    # JEPA-2 dataset
    # --------------------------
    dataset_j2 = Tier2Dataset(
        scene_map=get_scene_map(),
        dataset_path=DATASET_PATH,
        augment=True
    )
    
    # -------------------------
    # JEPA-2 graph vocab
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

    
    # --------------------------
    # Unified dataset
    # --------------------------
    unified_dataset = UnifiedDataset(jepa1_dataset=dataset_j1, jepa2_dataset=dataset_j2)
    
    unified_loader = DataLoader(
        unified_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=unified_collate_fn  # unified collate handles jepa-2
    )

    pipeline, models = build_all(DEVICE)

    autocast_ctx = (
        torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16)
        if USE_BF16
        else nullcontext()
    )

    global_step = 0

    for epoch in range(EPOCHS):
        pbar = tqdm(unified_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0

        for batch in pbar:
            # Each batch is a dict with keys "j1" and/or "j2"
            j1_batch = batch.get("j1", None)
            j2_batch = batch.get("j2", None)

            # Move JEPA-1 batch to device if exists
            if j1_batch is not None:
                batch["j1"] = move_j1_to_device(j1_batch, DEVICE)


            # Move JEPA-2 tensors to device if exists
            # -------------------------------------------------
            # JEPA-2 preprocessing
            # -------------------------------------------------
            if j2_batch is not None:
                graphs = j2_batch["graphs"]

                # Build graph tensors HERE (explicit & safe)
                graph_feats, graph_adj = build_graph_batch(
                    graphs,
                    type2id,
                    category2id,
                    layer2id
                )

                # Move to device
                j2_batch["graph_feats"] = graph_feats.to(DEVICE)
                j2_batch["graph_adj"] = graph_adj.to(DEVICE)

                # Move remaining tensors
                for key in ["graph_mask", "clean_deltas", "aug_deltas", "traj_mask"]:
                    j2_batch[key] = j2_batch[key].to(DEVICE)

                # Safety check (kills 8 vs 13 bug immediately)
                assert j2_batch["graph_feats"].shape[-1] == 13, \
                    f"Expected 13 graph features, got {j2_batch['graph_feats'].shape[-1]}"

                batch["j2"] = j2_batch


            with autocast_ctx:
                out = pipeline.step(batch)

            loss_total = out["loss"]
            loss_j1 = out["loss_j1"]
            loss_j2 = out["loss_j2"]
            loss_j3 = out["loss_j3"]

            loss_total_val = float(loss_total.detach().cpu())
            epoch_loss += loss_total_val
            global_step += 1

            # ============================================================
            # LOGGING (Comet)
            # ============================================================

            if global_step % 10 == 0:
                # ---- global loss ----
                log_metrics(
                    experiment,
                    {"total": loss_total},
                    prefix="loss",
                    step=global_step,
                )

                # ---- JEPA-1 ----
                log_metrics(
                    experiment,
                    {"total": loss_j1},
                    prefix="loss/jepa1",
                    step=global_step,
                )

                # ---- JEPA-2 ----
                log_metrics(
                    experiment,
                    {"total": loss_j2},
                    prefix="loss/jepa2",
                    step=global_step,
                )

                # ---- JEPA-3 ----
                log_metrics(
                    experiment,
                    {"total": loss_j3},
                    prefix="loss/jepa3",
                    step=global_step,
                )


            pbar.set_postfix({
                "L": f"{loss_total_val:.4f}",
                "L1": f"{float(loss_j1):.4f}",
                "L2": f"{float(loss_j2):.4f}",
                "L3": f"{float(loss_j3):.4f}",
            })


            if global_step % 500 == 0:
                torch.save(
                    {
                        "step": global_step,
                        **{k: v.state_dict() for k, v in models.items()}
                    },
                    CKPT_DIR / f"jepa_step{global_step}.pt",
                )

        avg_loss = epoch_loss / max(len(loader), 1)
        experiment.log_metric("epoch/avg_loss", avg_loss, step=epoch + 1)
        print(f"Epoch {epoch+1} done — avg loss {avg_loss:.6f}")

    experiment.end()
    print("✅ Training finished")


# ============================================================
# Entrypoint
# ============================================================
# ============================================================
# Entrypoint (Fail-safe protected)
# ============================================================
if __name__ == "__main__":
    experiment = None
    models = None

    try:
        result = train()
        experiment = result["experiment"]
        models = result["models"]

    except Exception:
        print("\n❌ TRAINING FAILED — fail-safe mode\n")

        # -------------------------
        # Save emergency checkpoint
        # -------------------------
        fail_path = "jepa_full_fail.pt"
        try:
            if models is not None:
                torch.save(
                    {k: v.state_dict() for k, v in models.items()},
                    fail_path,
                )
                print(f"💾 Fail-safe checkpoint saved: {fail_path}")

                if experiment is not None:
                    experiment.log_asset(fail_path)
        except Exception:
            print("⚠️ Could not save fail-safe checkpoint")

        # -------------------------
        # Save traceback
        # -------------------------
        with open("training_error_log.txt", "w") as f:
            f.write(traceback.format_exc())

        try:
            if experiment is not None:
                experiment.log_asset("training_error_log.txt")
                experiment.end()
        except Exception:
            pass

        raise

