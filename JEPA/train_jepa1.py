#!/usr/bin/env python3
"""
Stage-1: JEPA-1 pretraining using frozen V-JEPA-2 encoder.
"""

from comet_ml import Experiment
import signal
import sys
import os
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv


# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()
os.environ["COMET_LOG_PACKAGES"] = "0"

# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = Path("/kaggle/input/jepa1-checkpoint/pytorch/default/1")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
EPOCHS = int(os.getenv("EPOCHS", 2))
BATCH_SIZE = 8
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

       
# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def atomic_torch_save(obj, final_path: Path):
    tmp_path = final_path.with_suffix(".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(final_path)
    
def cleanup_old_checkpoints(ckpt_dir: Path, keep_last: int = 2):
    ckpts = sorted(
        ckpt_dir.glob("jepa1_step*.pt"),
        key=lambda p: int(p.stem.split("step")[-1])
    )
    for p in ckpts[:-keep_last]:
        try:
            p.unlink()
        except Exception as e:
            print(f"⚠️ Failed to delete old checkpoint {p}: {e}")

def safe_save(trainer, step, experiment=None, tag="auto"):
    print(f"\n💾 Saving checkpoint ({tag})")

    try:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        path = CKPT_DIR / f"{tag}_step{step}.pt"

        atomic_torch_save(
            {
                "version": 2,
                "step": step,
                "state": trainer.model.state_dict(),   #  must be "state"
            },
            path,
        )

        if experiment is not None:
            try:
                experiment.log_model(
                    file_or_folder=str(path),
                    name=f"jepa1_{tag}"
                )
                print("📤 Checkpoint uploaded as Kaggle model")
            except Exception as e:
                print(f"⚠️ Comet upload failed: {e}")

    except Exception as e:
        print(f"❌ Safe save failed: {e}")


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
        device_map=device.type,
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
    ).to(device)

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
        device=device,
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
        pin_memory=(device.type == "cuda"),
    )

    global_step = 0

    try:
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
                        "stats/s_c_std": stats["s_c_std"].item(),
                        "stats/s_c_mean": stats["s_c_mean"].item(),
                    }, step=global_step)

                pbar.set_postfix({"loss": f"{loss:.4f}"})

                # -------------------------------
                # SAFE CHECKPOINT
                # -------------------------------
                if global_step % 1000 == 0:
                    try:
                        safe_save(
                            trainer,
                            global_step,
                            experiment=experiment,
                            tag="checkpoint",
                        )                           
                        cleanup_old_checkpoints(CKPT_DIR, keep_last=2)

                    except Exception as e:
                        print(f"❌ Checkpoint save failed at step {global_step}: {e}")
                        safe_save(
                            trainer,
                            global_step,
                            experiment=experiment,
                            tag="checkpoint_io_failure",
                        )

            experiment.log_metric("epoch", epoch + 1, step=global_step)
            print(f"Epoch {epoch+1} finished")

    # -------------------------------
    # INTERRUPT / CRASH HANDLING
    # -------------------------------
    except KeyboardInterrupt:
        print("\n⛔ KeyboardInterrupt detected")
        safe_save(trainer, global_step, experiment, tag="interrupt")
        raise

    except Exception as e:
        print("\n❌ Training crashed:", e)
        safe_save(trainer, global_step, experiment, tag="crash")
        raise

    else:
        # ✅ ONLY runs if training finished successfully
        print("\n🎉 Training completed successfully")
        safe_save(trainer, global_step, experiment, tag="final")

    finally:
        experiment.end()


if __name__ == "__main__":
    train()
