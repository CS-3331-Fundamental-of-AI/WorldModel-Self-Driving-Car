#!/usr/bin/env python3
"""
Stage-1: JEPA-1 pretraining using frozen V-JEPA-2 encoder.
"""

import os
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from comet_ml import Experiment

# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()
os.environ["COMET_LOG_PACKAGES"] = "0"

# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints" / "jepa1_vjepa"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Device
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Dataset
# --------------------------------------------------
from Utils.jepa1vjepadata import (
    MapDataset,
    collate_maps,
    build_map_dataframe,
)

MAP_ROOT = os.getenv(
    "MAP_ROOT",
    "/kaggle/input/a-crude-data-set-converted-from-nuscene/local_maps",
)

assert os.path.exists(MAP_ROOT), f"MAP_ROOT does not exist: {MAP_ROOT}"


# --------------------------------------------------
# Model
# --------------------------------------------------
from transformers import AutoModel
from JEPA_PrimitiveLayer.vjepa.model import PrimitiveLayerJEPA

# --------------------------------------------------
# Trainer
# --------------------------------------------------
from trainers.trainer_jepa1vjepa2 import (
    JEPA1VJEPATrainer
)

# ==================================================
# Build
# ==================================================
def build():
    backbone = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        torch_dtype=torch.float16,
        device_map=DEVICE.type,
    )

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    model = PrimitiveLayerJEPA(
        encoder=backbone.encoder,
        grid_h=16,
        grid_w=16,
        enc_dim=1024,
        prim_dim=128,
    ).to(DEVICE)

    model.predictor.to(dtype=torch.float32)
    model.train()

    optimizer = torch.optim.AdamW(
        model.predictor.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    trainer = JEPA1VJEPATrainer(
        model=model,
        optimizer=optimizer,
        device=DEVICE,
    )

    return trainer


# ==================================================
# Train
# ==================================================
def train():
    experiment = Experiment(
    api_key="QyLnRuj1INQl4us5SoyYulpae",
    project_name="vjepa-2",
    workspace="c-nguy-n-3886",
    auto_param_logging=False,
    auto_metric_logging=False,
    )
    experiment.log_parameters({
        "model": "vjepa2-vitl-fpc64-256",
        "encoder_frozen": True,
        "primitive_dim": 128,
        "grid_h": 16,
        "grid_w": 16,
        "loss": "L1",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "precision": "encoder_fp16_predictor_fp32",
    })

    experiment.set_name("JEPA1-VJEPA2-PRETRAIN")

    trainer = build()

    # --------------------------------------------------
    # Data (IDENTICAL to Kaggle)
    # --------------------------------------------------
    df = build_map_dataframe(MAP_ROOT)

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True,
    )

    train_loader = DataLoader(
        MapDataset(train_df, mask_ratio=0.3),
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_maps,
        pin_memory=(DEVICE.type == "cuda"),
    )

    global_step = 0
    EPOCHS = int(os.getenv("EPOCHS", 10))

    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            global_step += 1
            out = trainer.step(batch)

            loss = out["loss"].item()
            stats = out["stats"]

            if global_step % 10 == 0:
                experiment.log_metrics({
                    "loss/total": loss,
                    "loss/align": stats["loss_align"].item(),
                    "loss/var":   stats["loss_var"].item(),
                    "loss/cov":   stats["loss_cov"].item(),
                    "stats/z_std": stats["z_hat_std"].item(),
                }, step=global_step)

            pbar.set_postfix({"loss": f"{loss:.4f}"})

            if global_step % 500 == 0:
                ckpt = CKPT_DIR / f"jepa1_step{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "model": trainer.model.state_dict(),
                    },
                    ckpt,
                )
                experiment.log_asset(str(ckpt))

        print(f"Epoch {epoch+1} finished")

    experiment.end()
    print("✅ JEPA-1 pretraining finished")


if __name__ == "__main__":
    train()
