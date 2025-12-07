import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from comet_ml import Experiment

# ================================
# Project modules
# ================================
from dataset import Tier2Dataset, tier2_collate_fn
from LatentCollector import LatentCollector
from TrajTokenizer import TrajectoryTokenizerFSQ
from EDAIN import EDAINLayer
from losses import total_tokenizer_loss_fsq
from kinematic import build_scene_mapping_parallel


# ============================================================
# 0. COMET ENVIRONMENT SETUP
# ============================================================

os.environ["COMET_DISABLE_MULTI_PROCESSING"] = "1"

experiment = Experiment(
    api_key=os.getenv("API_KEY"),
    project_name=os.getenv("PROJECT_NAME"),
    workspace=os.getenv("WORK_SPACE"),
)

experiment.set_name("TrajectoryTokenizer-T2-with-EDAIN+FSQ+VICReg")
experiment.log_parameters({
    "batch_size": 8,
    "lr": 1e-4,
    "epochs": 3,
    "traj_dim": 6,
    "smooth_lambda": 0.1,
    "vic_inv": 1.0,
    "vic_var": 1.0,
    "vic_cov": 0.01,
    "normalization": "EDAIN",
})


# ============================================================
# 1. DATASET + LOADER
# ============================================================

print("Building dataset...")
scene_map = build_scene_mapping_parallel(
    "/kaggle/input/a-crude-data-set-converted-from-nuscene/metadata"
)

dataset = Tier2Dataset(
    scene_map=scene_map,
    dataset_path="/kaggle/input/a-crude-data-set-converted-from-nuscene",
    augment=True
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=tier2_collate_fn,
    num_workers=0,
    pin_memory=True
)

latent_probe = LatentCollector(latent_dim=128, max_batches=200)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. MODEL SETUP
# ============================================================

train_traj_tokenizer = TrajectoryTokenizerFSQ(
    traj_dim=6,
    T=8,
    enc_latent_dim=128,
    d_q=6,
    fsq_levels=7,
    enc_layers=3,
    dec_hidden=64,
    use_layernorm=True
).to(device)

edain = EDAINLayer(D=6).to(device)

optimizer = optim.Adam(
    list(train_traj_tokenizer.parameters()) +
    list(edain.parameters()),
    lr=1e-4
)

EPOCHS = 3
global_step = 0


# ============================================================
# 3. TRAINING FUNCTION
# ============================================================

def train_one_epoch(epoch):
    global global_step

    train_traj_tokenizer.train()
    edain.train()

    bar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)

    epoch_stats = {
        "loss": 0, "recon": 0, "smooth": 0,
        "vic_inv": 0, "vic_var": 0, "vic_cov": 0,
        "usage": 0
    }

    for batch in bar:

        traj_clean = batch["clean_deltas"].to(device)
        traj_aug   = batch["aug_deltas"].to(device)

        # EDAIN normalization
        traj_clean = edain(traj_clean)
        traj_aug   = edain(traj_aug)

        # Forward pass
        out_clean = train_traj_tokenizer(traj_clean)
        out_aug   = train_traj_tokenizer(traj_aug)

        z0 = out_clean["z0"]
        latent_probe.add_batch(z0)

        # Loss
        loss_dict = total_tokenizer_loss_fsq(
            out_clean=out_clean,
            out_aug=out_aug,
            target_traj=traj_clean,
            fsq_levels=train_traj_tokenizer.fsq_levels,
            lambda_recon=1.0,
            lambda_smooth=0.1,
            lambda_inv=1.0,
            lambda_var=1.0,
            lambda_cov=0.01,
            lambda_usage=0.2
        )

        loss = loss_dict["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging to COMET every 50 steps
        if global_step % 50 == 0:
            experiment.log_metrics({
                "train/loss": loss.item(),
                "train/recon": loss_dict["loss_recon"],
                "train/smooth": loss_dict["loss_smooth"],
                "vic/inv": loss_dict["vic_inv"],
                "vic/var": loss_dict["vic_var"],
                "vic/cov": loss_dict["vic_cov"],
                "fsq/usage": loss_dict["fsq_usage"],
            }, step=global_step)

        bar.set_postfix(loss=f"{loss.item():.4f}")

        # accumulate
        epoch_stats["loss"]    += loss.item()
        epoch_stats["recon"]   += loss_dict["loss_recon"]
        epoch_stats["smooth"]  += loss_dict["loss_smooth"]
        epoch_stats["vic_inv"] += loss_dict["vic_inv"]
        epoch_stats["vic_var"] += loss_dict["vic_var"]
        epoch_stats["vic_cov"] += loss_dict["vic_cov"]
        epoch_stats["usage"]   += loss_dict["fsq_usage"]

        global_step += 1

    # average
    N = len(loader)
    for k in epoch_stats:
        epoch_stats[k] /= N

    experiment.log_metrics({
        f"epoch/{k}": v for k, v in epoch_stats.items()
    }, epoch=epoch)

    print(f"✓ Finished epoch {epoch}")
    return epoch_stats


# ============================================================
# 4. VALIDATION FUNCTION
# ============================================================

def run_validation():
    print("\nRunning validation...")

    train_traj_tokenizer.eval()
    edain.eval()

    # Load test dataset
    test_scene = build_scene_mapping_parallel(
        "/kaggle/input/test-dataset/metadata"
    )

    test_dataset = Tier2Dataset(
        scene_map=test_scene,
        dataset_path="/kaggle/input/test-dataset",
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=tier2_collate_fn
    )

    L = train_traj_tokenizer.fsq_levels
    token_hist = torch.zeros(L, dtype=torch.long)

    stats = {"loss": 0, "recon": 0, "smooth": 0, "usage": 0}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[VALIDATION]"):

            traj = edain(batch["clean_deltas"].to(device))
            out  = train_traj_tokenizer(traj)

            # Count tokens
            tokens = train_traj_tokenizer.encode_tokens(traj).view(-1).cpu()
            token_hist.scatter_add_(0, tokens, torch.ones_like(tokens))

            # Compute loss
            loss_dict = total_tokenizer_loss_fsq(
                out_clean=out, out_aug=out,
                target_traj=traj,
                fsq_levels=train_traj_tokenizer.fsq_levels,
                lambda_recon=1.0,
                lambda_smooth=0.1,
                lambda_usage=0.1,
                lambda_inv=0,
                lambda_var=0,
                lambda_cov=0,
            )

            stats["loss"]   += loss_dict["loss_total"]
            stats["recon"]  += loss_dict["loss_recon"]
            stats["smooth"] += loss_dict["loss_smooth"]
            stats["usage"]  += loss_dict["fsq_usage"]

    # Normalize
    N = len(test_loader)
    for k in stats:
        stats[k] /= N

    print("\n====== VALIDATION SUMMARY ======")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    print("\nToken histogram:", token_hist)

    experiment.log_metrics({
        "val/loss": stats["loss"],
        "val/recon": stats["recon"],
        "val/smooth": stats["smooth"],
        "val/usage": stats["usage"],
    })

    for level in range(L):
        experiment.log_metric(f"val/token_level_{level}", token_hist[level].item())


# ============================================================
# 5. MAIN TRAINING LOOP
# ============================================================

try:
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)

except Exception as e:
    print("\n❌ ERROR — saving FAILSAFE checkpoint")
    torch.save(train_traj_tokenizer.state_dict(), "failsafe_tokenizer.pth")
    torch.save(edain.state_dict(), "failsafe_edain.pth")
    raise e

finally:
    print("\n✔ Saving final checkpoint")
    torch.save(train_traj_tokenizer.state_dict(), "tokenizer_final.pth")
    torch.save(edain.state_dict(), "edain_final.pth")
    experiment.log_model("tokenizer_final", "tokenizer_final.pth")
    experiment.log_model("edain_final", "edain_final.pth")


# ============================================================
# 6. RUN VALIDATION AFTER TRAINING
# ============================================================

run_validation()