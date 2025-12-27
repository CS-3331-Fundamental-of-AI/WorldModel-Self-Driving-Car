#!/usr/bin/env python3
"""
Stage-0: Global GCN self-supervised pretraining
(using Tier3Dataset global graphs)

- Mask node positions
- Reconstruct masked nodes
- Save checkpoint in JEPA-compatible format
"""

import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from comet_ml import Experiment

# --------------------------------------------------
# Env
# --------------------------------------------------
load_dotenv()  # load .env
os.environ["COMET_LOG_PACKAGES"] = "0"

# --------------------------------------------------
# Device
# --------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"👉 Using device: {device}")

# --------------------------------------------------
# Paths / Checkpoints
# --------------------------------------------------
CKPT_DIR = Path("/kaggle/working/checkpoints/gcn_global")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
from Utils.jepa2data import get_scene_map
from Utils.jepa3data import Tier3Dataset, tier3_collate_fn

scene_map = get_scene_map()
dataset = Tier3Dataset(scene_map)
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    collate_fn=tier3_collate_fn,
    pin_memory=(device.type == "cuda"),
)

# --------------------------------------------------
# Model
# --------------------------------------------------
from JEPA_ThirdLayer.gcn_pretrain import GCNPretrainModel
from Utils.gcn_utils import edges_to_adj, mask_nodes

model = GCNPretrainModel(node_dim=3, hidden=128, out_dim=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# --------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------
def atomic_torch_save(obj, final_path: Path):
    tmp = final_path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(final_path)

def safe_save(model, step, tag="auto"):
    path = CKPT_DIR / f"{tag}_step{step}.pt"
    print(f"\n💾 Saving checkpoint: {path.name}")
    atomic_torch_save({"version": 1, "step": step, "state": model.state_dict()}, path)

# --------------------------------------------------
# Comet log
# --------------------------------------------------
experiment = Experiment(
    api_key=os.getenv("API_KEY"),
    project_name=os.getenv("PROJECT_NAME", "JEPA"),
    workspace=os.getenv("WORK_SPACE"),
)
experiment.set_name("GCN-Stage0-Pretrain")

# --------------------------------------------------
# Training
# --------------------------------------------------
EPOCHS = 4
MASK_RATIO = 0.2  # lower mask ratio for stability
global_step = 0

try:
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            global_step += 1

            global_nodes = [n.to(device) for n in batch["global_nodes"]]
            global_edges = [e.to(device) for e in batch["global_edges"]]

            losses = []

            for nodes, edges in zip(global_nodes, global_edges):
                # -------------------
                # Normalize per graph
                # -------------------
                nodes_mean = nodes.mean(dim=0, keepdim=True)
                nodes_std = nodes.std(dim=0, keepdim=True) + 1e-6
                nodes_norm = (nodes - nodes_mean) / nodes_std

                N = nodes.shape[0]
                adj = edges_to_adj(edges, N).to(device)

                nodes_norm = nodes_norm.unsqueeze(0)  # [1, N, 3]
                adj = adj.unsqueeze(0)                # [1, N, N]

                masked_nodes, mask = mask_nodes(nodes_norm, MASK_RATIO)

                recon, _ = model(masked_nodes, adj)

                node_loss = F.mse_loss(recon, nodes_norm, reduction="none")
                node_loss = node_loss.mean(dim=-1)
                loss = (node_loss * mask.float()).sum() / mask.sum().clamp(min=1)
                losses.append(loss)

            loss = torch.stack(losses).mean()
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # ----------------- Comet logging -----------------
            experiment.log_metric("loss_step", loss.item(), step=global_step)
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_loss = epoch_loss / max(len(loader), 1)
        experiment.log_metric("loss_epoch_avg", avg_loss, step=epoch+1)
        print(f"Epoch {epoch+1} finished — avg loss: {avg_loss:.5f}")

except KeyboardInterrupt:
    print("\n⛔ Interrupted")
    safe_save(model, global_step, tag="interrupt")
    raise

except Exception as e:
    print("\n❌ Training crashed:", e)
    safe_save(model, global_step, tag="crash")
    raise

else:
    print("\n🎉 Pretraining completed")
    safe_save(model, global_step, tag="gcn_global_final")
