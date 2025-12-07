# -------------------------------------------------------------
# Tier-2 Module: Training Script with Comet Logging + Fail-Safe
# -------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from comet_ml import Experiment
from torch.utils.data import DataLoader
from tqdm import tqdm

from jepaTier2 import Tier2Module
from utils import count_params, build_graph_batch
from dataset import Tier2Dataset, tier2_collate_fn
from kinematic import build_scene_mapping_parallel

# -------------------------------------------------------------
# 0. LOAD ENVIRONMENT
# -------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
PROJECT_NAME = os.getenv("PROJECT_NAME", "tier2-default")
WORKSPACE = os.getenv("WORK_SPACE", "dtj-tran")

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. COMET EXPERIMENT
# -------------------------------------------------------------
experiment = Experiment(
    api_key=API_KEY,
    project_name=PROJECT_NAME,
    workspace=WORKSPACE,
    auto_param_logging=True,
    auto_metric_logging=True,
    auto_output_logging=False,
)

experiment.set_name("Tier2-Training")

# -------------------------------------------------------------
# 2. CONFIG
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "batch_size": 8,
    "lr": 1e-4,
    "epochs": 5,
    "tau": 0.5,
    "loss_type": "MSE(fusion, traj_emb)"
}
experiment.log_parameters(config)

# -------------------------------------------------------------
# 3. DATASET
# -------------------------------------------------------------
dataroot = "/kaggle/input/a-crude-data-set-converted-from-nuscene/metadata"
dataset_path = "/kaggle/input/a-crude-data-set-converted-from-nuscene"

scene_map = build_scene_mapping_parallel(dataroot)
train_dataset = Tier2Dataset(scene_map=scene_map, dataset_path=dataset_path, augment=True)

loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    collate_fn=tier2_collate_fn,
    num_workers=0,
    pin_memory=True
)

# -------------------------------------------------------------
# 4. MODEL + OPTIMIZER
# -------------------------------------------------------------
model = Tier2Module().to(device)
count_params(model)

optimizer = optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.MSELoss()

# Static dictionaries
type_values = ['agent', 'road']
category_values = [
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
layer_values = ['lane']

type2id = {v:i for i,v in enumerate(type_values)}
category2id = {v:i for i,v in enumerate(category_values)}
layer2id = {v:i for i,v in enumerate(layer_values)}

# -------------------------------------------------------------
# 5. FAIL-SAFE CHECKPOINT FUNCTION
# -------------------------------------------------------------
def save_checkpoint_safely(epoch, model, optimizer, experiment):
    ckpt_path = f"{CHECKPOINT_DIR}/tier2_epoch_{epoch}.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"💾 Saved checkpoint → {ckpt_path}")

    # Upload to comet (fail-safe)
    try:
        experiment.log_asset(ckpt_path, overwrite=True)
        print("☁️ Uploaded checkpoint to Comet.")
    except Exception as e:
        print("❗ Comet upload failed:", e)

# -------------------------------------------------------------
# 6. TRAINING LOOP
# -------------------------------------------------------------
print("🚀 Starting Tier-2 Training...")

try:
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config['epochs']}")

        for batch in pbar:
            graphs = batch["graphs"]
            graph_mask = batch["graph_mask"].to(device)
            traj = batch["clean_deltas"].to(device)
            traj_mask = batch["traj_mask"].to(device)

            # Build graph tensors
            x_batch, adj_batch = build_graph_batch(graphs, type2id, category2id, layer2id)
            x_batch = x_batch.to(device)
            adj_batch = adj_batch.to(device)

            # Forward pass
            out = model(
                traj=traj,
                adj=adj_batch,
                x_graph=x_batch,
                traj_mask=traj_mask,
                graph_mask=graph_mask,
                tau=config["tau"],
            )

            # Simple JEPA-like latent alignment
            loss = criterion(out["fusion"], out["traj_emb"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            experiment.log_metric("batch_loss", loss.item())

        avg_loss = epoch_loss / len(loader)
        experiment.log_metric("epoch_loss", avg_loss)
        print(f"🔥 Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

        # Save and upload checkpoint at each epoch
        save_checkpoint_safely(epoch, model, optimizer, experiment)

except Exception as e:
    print("❗ TRAINING CRASHED! Saving fail-safe checkpoint...")
    save_checkpoint_safely("FAILSAFE", model, optimizer, experiment)
    raise e

print("🎉 Training completed successfully!")